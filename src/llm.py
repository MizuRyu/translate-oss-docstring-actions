from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import date
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import JsonSchemaFormat, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from pydantic import BaseModel, ConfigDict, ValidationError

from src.util import count_tokens, logger

MAX_TOKENS = 3000
TOKEN_BUFFER = 500
MAX_REQUEST_TOKENS = MAX_TOKENS - TOKEN_BUFFER
REQUEST_OVERHEAD_TOKENS = 10
TEMPERATURE = 0.2
TOP_P = 1.0
PRIMARY_TIMEOUT_SECONDS = 100.0
FALLBACK_TIMEOUT_SECONDS = 100.0
AZURE_TPM_LIMIT = 100000  # Azure OpenAI: 1分あたり10万トークン


class TranslationEntry(BaseModel):
    index: int
    translation: str

    model_config = ConfigDict(extra="forbid")


class TranslationResponse(BaseModel):
    translations: List[TranslationEntry]

    model_config = ConfigDict(extra="forbid")


RESPONSE_FORMAT = JsonSchemaFormat(
    name="translation_response",
    schema=TranslationResponse.model_json_schema(),
    strict=True,
)


class TranslationRequestItem(BaseModel):
    index: int
    original: str


class TranslationRequest(BaseModel):
    items: List[TranslationRequestItem]


class GithubModelPolicy(NamedTuple):
    rpm: Optional[int]
    rpd: Optional[int]
    concurrency: Optional[int]


_GITHUB_MODEL_POLICIES: Dict[str, GithubModelPolicy] = {
    "openai/gpt-4.1": GithubModelPolicy(rpm=10, rpd=150, concurrency=2),
    "openai/gpt-4.1-mini": GithubModelPolicy(rpm=15, rpd=150, concurrency=5),
}

GITHUB_MODEL_SEQUENCE = ["openai/gpt-4.1", "openai/gpt-4.1-mini"]


class ConcurrencyLimiter:
    """GitHubモデルの同時実行数を管理するセマフォラッパー"""

    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self.limit = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def acquire(self) -> None:
        await self._semaphore.acquire()

    def release(self) -> None:
        self._semaphore.release()

    @asynccontextmanager
    async def slot(self):
        await self.acquire()
        try:
            yield
        finally:
            self.release()


_CONCURRENCY_LIMITERS: Dict[Tuple[str, asyncio.AbstractEventLoop], ConcurrencyLimiter] = {}


def _init_stats() -> Dict[str, int]:
    """翻訳バッチで利用する統計情報の初期値を生成する"""

    return {
        "primary_requests": 0,
        "fallback_requests": 0,
        "rate_limit_hits": 0,
        "timeouts": 0,
    }


def _get_github_model_policy(model_name: str) -> GithubModelPolicy:
    """GitHubモデルごとのレート上限と同時実行数を返す"""

    try:
        return _GITHUB_MODEL_POLICIES[model_name]
    except KeyError as e:
        raise TranslationRequestError(f"未対応のGitHubモデルです model={model_name}") from e


def _get_concurrency_limiter(
    model_name: str, limit: Optional[int]
) -> Optional[ConcurrencyLimiter]:
    """モデル単位の並列数を管理するリミッタを取得する"""

    if not limit or limit <= 0:
        return None
    loop = asyncio.get_running_loop()
    key = (model_name, loop)
    limiter = _CONCURRENCY_LIMITERS.get(key)
    if limiter is None or limiter.limit != limit:
        limiter = ConcurrencyLimiter(limit)
        _CONCURRENCY_LIMITERS[key] = limiter
    return limiter

class GHModelsRateLimiter:
    """GitHubエンドポイント向けの単純なレートリミッタ。"""

    def __init__(self, per_minute: Optional[int], per_day: Optional[int]) -> None:
        self.per_minute = per_minute
        self.per_day = per_day
        self.events: deque[float] = deque()
        self.current_day: date = date.today()
        self.daily_count = 0

    def _reset_if_needed(self) -> None:
        today = date.today()
        if today != self.current_day:
            self.current_day = today
            self.daily_count = 0

    async def acquire(self) -> None:
        self._reset_if_needed()
        if self.per_day and self.daily_count >= self.per_day:
            raise DailyLimitError("GitHub daily request limit reached")
        if not self.per_minute:
            self.daily_count += 1
            return
        while True:
            now = time.monotonic()
            while self.events and now - self.events[0] >= 60:
                self.events.popleft()
            if len(self.events) < self.per_minute:
                self.events.append(now)
                self.daily_count += 1
                return
            sleep_for = 60 - (now - self.events[0])
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)


class AOAIRateLimiter:
    """Azure OpenAI向けのトークンベースレートリミッタ（TPM制限対応）"""

    def __init__(self, tokens_per_minute: int) -> None:
        self.tokens_per_minute = tokens_per_minute
        self.token_events: deque[Tuple[float, int]] = deque()  # (timestamp, token_count)

    async def acquire(self, token_count: int) -> None:
        """指定トークン数を消費可能になるまで待機する
        
        Args:
            token_count: 消費するトークン数
        """
        while True:
            now = time.monotonic()
            # 1分以上前のイベントを削除
            while self.token_events and now - self.token_events[0][0] >= 60:
                self.token_events.popleft()
            
            # 現在の1分間のトークン使用量を計算
            current_tokens = sum(tokens for _, tokens in self.token_events)
            
            # トークン数が制限内なら許可
            if current_tokens + token_count <= self.tokens_per_minute:
                self.token_events.append((now, token_count))
                logger.debug(
                    "Azure TPM: acquired %d tokens (current: %d/%d)",
                    token_count,
                    current_tokens + token_count,
                    self.tokens_per_minute,
                )
                return
            
            # 制限を超える場合は待機
            if self.token_events:
                oldest_timestamp = self.token_events[0][0]
                sleep_for = 60 - (now - oldest_timestamp) + 0.1  # 少し余裕を持たせる
            else:
                sleep_for = 1.0
            
            logger.warning(
                "Azure TPM limit reached: waiting %.1fs (current: %d, needed: %d, limit: %d)",
                sleep_for,
                current_tokens,
                token_count,
                self.tokens_per_minute,
            )
            await asyncio.sleep(sleep_for)


_azure_token_limiter: Optional[AOAIRateLimiter] = None


def _get_azure_token_limiter() -> AOAIRateLimiter:
    """Azure OpenAI用のトークンレートリミッタを取得する（シングルトン）"""
    global _azure_token_limiter
    if _azure_token_limiter is None:
        _azure_token_limiter = AOAIRateLimiter(AZURE_TPM_LIMIT)
    return _azure_token_limiter


async def translate_batch(
    system_prompt: str,
    entries: Sequence[Dict[str, Any]],
    *,
    is_mock: bool = False,
    azure_only: bool = False,
) -> Tuple[Optional[List[str]], Optional[Exception], Dict[str, int]]:
    """翻訳を1バッチ分実行し、結果と統計値を返す
    
    Args:
        system_prompt: システムプロンプト
        entries: 翻訳対象エントリのリスト
        is_mock: モックモード（LLMを呼ばずダミー翻訳を返す）
        azure_only: Azure AI Inferenceのみを使用（GitHub Modelsをスキップ）
    
    Returns:
        (翻訳結果リスト, エラー, 統計情報)
    """

    stats = _init_stats()

    texts = [entry["text"] for entry in entries]

    logger.debug("translate_batch entries=%d is_mock=%s azure_only=%s", len(texts), is_mock, azure_only)

    if is_mock:
        translations = await _mock_request(system_prompt, texts)
        return translations, None, stats

    # Azure onlyモードの場合、GitHub Modelsをスキップ
    if not azure_only:
        primary_client = get_github_client()

        for model_name in GITHUB_MODEL_SEQUENCE:
            policy = _get_github_model_policy(model_name)
            concurrency_limiter = _get_concurrency_limiter(model_name, policy.concurrency)
            if concurrency_limiter is None:
                raise TranslationRequestError(
                    f"GitHubモデルの同時実行数が未設定です model={model_name}"
                )

            request_limiter = GHModelsRateLimiter(policy.rpm, policy.rpd)
            primary_params = _get_model_config(model_name)

            try:
                async with concurrency_limiter.slot():
                    await request_limiter.acquire()
                    translations = await asyncio.wait_for(
                        _invoke_client(primary_client, primary_params, system_prompt, texts),
                        timeout=PRIMARY_TIMEOUT_SECONDS,
                    )
                stats["primary_requests"] += 1
                logger.debug("GitHub model used model=%s", model_name)
                return translations, None, stats
            except asyncio.TimeoutError:
                logger.warning(
                    "GitHub model timed out model=%s limit=%.1f entries=%d",
                    model_name,
                    PRIMARY_TIMEOUT_SECONDS,
                    len(texts),
                )
                stats["timeouts"] += 1
                continue
            except DailyLimitError:
                logger.info("GitHub model daily limit reached model=%s", model_name)
                stats["rate_limit_hits"] += 1
                continue
            except RateLimitError as rate_error:
                logger.debug("GitHub rate limit model=%s error=%s", model_name, rate_error)
                stats["rate_limit_hits"] += 1
                continue
            except (TranslationRequestError, TranslationParseError) as primary_error:
                logger.debug("GitHub request failed model=%s error=%s", model_name, primary_error)
                break
    else:
        logger.info("Azure-only mode: skipping GitHub Models")

    # Azure AI Inference (fallback)
    fallback_client = get_azure_client()
    fallback_params = _get_model_config(os.getenv("AZURE_INFERENCE_MODEL", "gpt-4.1-mini"))

    # Azure TPM制限を適用
    token_limiter = _get_azure_token_limiter()
    # システムプロンプト + ユーザーメッセージのトークン数を概算
    total_text = system_prompt + "\n".join(texts)
    estimated_tokens = count_tokens(total_text)
    # 出力トークンも考慮（入力の50%程度と仮定）
    estimated_total_tokens = int(estimated_tokens * 1.5)
    
    await token_limiter.acquire(estimated_total_tokens)

    try:
        translations = await asyncio.wait_for(
            _invoke_client(fallback_client, fallback_params, system_prompt, texts),
            timeout=FALLBACK_TIMEOUT_SECONDS,
        )
        stats["fallback_requests"] += 1
        return translations, None, stats
    except asyncio.TimeoutError:
        logger.warning(
            "Azure fallback timed out after %.1f sec entries=%d",
            FALLBACK_TIMEOUT_SECONDS,
            len(texts),
        )
        stats["timeouts"] += 1
        return None, RateLimitError("Azure fallback timed out"), stats
    except (TranslationRequestError, TranslationParseError) as fallback_error:
        logger.debug("fallback failed error=%s", fallback_error)
        if isinstance(fallback_error, (RateLimitError, DailyLimitError)):
            stats["rate_limit_hits"] += 1
        return None, fallback_error, stats


async def _mock_request(_: str, texts: Sequence[str]) -> List[str]:
    """モックモードの翻訳結果を生成する"""

    return [f"{text} (mock)" for text in texts]


async def _invoke_client(
    client: ChatCompletionsClient,
    params: Dict[str, Any],
    system_prompt: str,
    texts: Sequence[str],
) -> List[str]:
    """LLMクライアントを呼び出しレスポンスを解析する"""

    payload = _build_payload(texts)

    def _call() -> Any:
        return client.complete(
            messages=[SystemMessage(system_prompt), UserMessage(payload)],
            **params,
        )

    try:
        response = await asyncio.to_thread(_call)
    except HttpResponseError as e:
        status = getattr(e, "status_code", None)
        message = str(e)
        if status == 429:
            raise RateLimitError(message) from e
        raise TranslationRequestError(message) from e
    except RateLimitError:
        raise
    except Exception as e:  # pragma: no cover
        raise TranslationRequestError(str(e)) from e

    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = _extract_usage_value(usage, "prompt_tokens")
        completion_tokens = _extract_usage_value(usage, "completion_tokens")
        total_tokens = _extract_usage_value(usage, "total_tokens")
        logger.debug(
            "LLM usage prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

    content = _extract_content(response)
    return _parse(content)


def _build_payload(texts: Sequence[str]) -> str:
    """バッチ要求用のJSON文字列を構築する"""

    request = TranslationRequest(
        items=[
            TranslationRequestItem(index=index, original=text)
            for index, text in enumerate(texts)
        ]
    )
    return request.model_dump_json(ensure_ascii=False)


def _extract_content(response: Any) -> str:
    """レスポンスオブジェクトから文字列を取り出す"""

    choice = response.choices[0]
    content = choice.message.content
    if isinstance(content, list):
        return "".join(getattr(fragment, "text", "") for fragment in content)
    if isinstance(content, str):
        return content
    raise TranslationParseError("LLMレスポンスが想定外の形式です")


def _parse(response_text: str) -> List[str]:
    """JSONレスポンスを翻訳リストへ変換する"""

    try:
        model = TranslationResponse.model_validate_json(response_text)
        translations = sorted(model.translations, key=lambda item: item.index)
        _validate_indexes(translations)
        return [entry.translation for entry in translations]
    except ValidationError:
        pass

    try:
        raw = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise TranslationParseError(str(error)) from error

    if isinstance(raw, list):
        converted = {
            "translations": [
                {"index": index, "translation": item if isinstance(item, str) else item.get("translation", "")}
                for index, item in enumerate(raw)
            ]
        }
    elif isinstance(raw, dict) and "translations" in raw:
        converted = {
            "translations": [
                {
                    "index": item.get("index", idx),
                    "translation": item.get("translation", ""),
                }
                for idx, item in enumerate(raw["translations"])
            ]
        }
    else:
        raise TranslationParseError("翻訳結果の形式が不正です")

    try:
        model = TranslationResponse.model_validate(converted)
    except ValidationError as error:
        raise TranslationParseError(str(error)) from error
    translations = sorted(model.translations, key=lambda item: item.index)
    _validate_indexes(translations)
    return [entry.translation for entry in translations]


def _extract_usage_value(usage: Any, key: str) -> Optional[int]:
    """usageオブジェクトから指定キーの数値を取得する"""

    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def _validate_indexes(entries: Sequence[TranslationEntry]) -> None:
    """インデックスが欠番なく並んでいるか検証する"""

    expected = list(range(len(entries)))
    actual = [entry.index for entry in entries]
    if actual != expected:
        raise TranslationParseError(f"翻訳結果のindexが不正です expected={expected} actual={actual}")


def _get_model_config(model_name: str) -> Dict[str, Any]:
    """翻訳リクエスト用のパラメータを組み立てる"""
    # Codexさんへ。冗長かもですが、可読性を意識しているのでこのままにしてください。
    params = {}
    params["model"] = model_name
    params["temperature"] = TEMPERATURE
    params["top_p"] = TOP_P
    params["response_format"] = RESPONSE_FORMAT

    return params


def get_github_client() -> ChatCompletionsClient:
    """GitHub Models向けクライアントを生成する"""

    endpoint = os.getenv("GH_MODELS_ENDPOINT")
    token = os.getenv("GH_TOKEN")
    api_version = os.getenv("API_VERSION", "2024-10-01-preview")
    if not endpoint or not token:
        raise TranslationRequestError("GitHub Models の接続情報が不足しています")

    return ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
        api_version=api_version,
    )

def get_azure_client() -> ChatCompletionsClient:
    """Azure Inference向けクライアントを生成する"""

    endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
    credential = os.getenv("AZURE_INFERENCE_CREDENTIAL")
    api_version = os.getenv("API_VERSION", "2024-10-01-preview")
    if not endpoint or not credential:
        raise TranslationRequestError("Azure Inference の接続情報が不足しています")

    return ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(credential),
        api_version=api_version,
    )



class TranslationError(Exception):
    """翻訳処理で発生する基底例外。"""


class TranslationRequestError(TranslationError):
    """LLMへのリクエストで失敗した場合の例外。"""


class TranslationParseError(TranslationError):
    """LLMレスポンスの解析に失敗した場合の例外。"""


class TokenLimitError(TranslationError):
    """トークン制限によりリクエストを構成できなかった場合の例外。"""


class RateLimitError(TranslationRequestError):
    """GitHub側でレートリミットに到達した場合の例外。"""


class DailyLimitError(TranslationRequestError):
    """GitHubの日次リミットに達した場合の例外。"""
