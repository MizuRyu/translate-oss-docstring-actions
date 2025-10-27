from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import JsonSchemaFormat, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from pydantic import BaseModel, ConfigDict, ValidationError

from src.util import count_tokens, logger

MAX_TOKENS = 8000
TOKEN_BUFFER = 500
MAX_REQUEST_TOKENS = MAX_TOKENS - TOKEN_BUFFER
GITHUB_MODELS_RPM = 15
GITHUB_MODELS_DAILY = 150
REQUEST_OVERHEAD_TOKENS = 10
TEMPERATURE = 0.2
TOP_P = 1.0
PRIMARY_TIMEOUT_SECONDS = 100.0
FALLBACK_TIMEOUT_SECONDS = 100.0


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


def _init_stats() -> Dict[str, int]:
    """翻訳バッチで利用する統計情報の初期値を生成する"""

    return {
        "primary_requests": 0,
        "fallback_requests": 0,
        "rate_limit_hits": 0,
        "timeouts": 0,
    }

class RequestRateLimiter:
    """GitHubエンドポイント向けの単純なレートリミッタ。"""

    def __init__(self, per_minute: int, per_day: int) -> None:
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


async def translate_batch(
    system_prompt: str,
    entries: Sequence[Dict[str, Any]],
    *,
    mock_mode: bool = False,
) -> Tuple[Optional[List[str]], Optional[Exception], Dict[str, int]]:
    stats = _init_stats()

    texts = [entry["text"] for entry in entries]

    logger.debug("translate_batch entries=%d mock_mode=%s", len(texts), mock_mode)

    if mock_mode:
        translations = await _mock_request(system_prompt, texts)
        return translations, None, stats

    primary_client = get_github_client()
    primary_params = _get_model_config(os.getenv("GITHUB_MODELS_MODEL", "openai/gpt-4.1-mini"))
    limiter = RequestRateLimiter(GITHUB_MODELS_RPM, GITHUB_MODELS_DAILY)

    try:
        await limiter.acquire()
        translations = await asyncio.wait_for(
            _invoke_client(primary_client, primary_params, system_prompt, texts),
            timeout=PRIMARY_TIMEOUT_SECONDS,
        )
        stats["primary_requests"] += 1
        return translations, None, stats
    except asyncio.TimeoutError:
        logger.warning(
            "GitHub primary timed out after %.1f sec entries=%d",
            PRIMARY_TIMEOUT_SECONDS,
            len(texts),
        )
        stats["timeouts"] += 1
    except (RateLimitError, DailyLimitError) as rate_error:
        logger.debug("primary rate limit error=%s", rate_error)
        stats["rate_limit_hits"] += 1
    except (TranslationRequestError, TranslationParseError) as primary_error:
        logger.debug("primary failed error=%s", primary_error)
        stats["rate_limit_hits"] += 1

    fallback_client = get_azure_client()
    fallback_params = _get_model_config(os.getenv("AZURE_INFERENCE_MODEL", "gpt-4.1-mini"))

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
    return [f"{text} (mock)" for text in texts]

async def _invoke_client(
    client: ChatCompletionsClient,
    params: Dict[str, Any],
    system_prompt: str,
    texts: Sequence[str],
) -> List[str]:
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

    content = _extract_content(response)
    return _parse(content)


def _build_payload(texts: Sequence[str]) -> str:
    request = TranslationRequest(
        items=[
            TranslationRequestItem(index=index, original=text)
            for index, text in enumerate(texts)
        ]
    )
    return request.model_dump_json(ensure_ascii=False)


def _extract_content(response: Any) -> str:
    choice = response.choices[0]
    content = choice.message.content
    if isinstance(content, list):
        return "".join(getattr(fragment, "text", "") for fragment in content)
    if isinstance(content, str):
        return content
    raise TranslationParseError("LLMレスポンスが想定外の形式です")


def _parse(response_text: str) -> List[str]:
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


def _validate_indexes(entries: Sequence[TranslationEntry]) -> None:
    expected = list(range(len(entries)))
    actual = [entry.index for entry in entries]
    if actual != expected:
        raise TranslationParseError(f"翻訳結果のindexが不正です expected={expected} actual={actual}")


def _get_model_config(model_name: str) -> Dict[str, Any]:
    # Codexさんへ。冗長かもですが、可読性を意識しているのでこのままにしてください。
    params = {}
    params["model"] = model_name
    params["temperature"] = TEMPERATURE
    params["top_p"] = TOP_P
    params["response_format"] = RESPONSE_FORMAT

    return params


def get_github_client() -> ChatCompletionsClient:
    endpoint = os.getenv("GITHUB_MODELS_ENDPOINT")
    token = os.getenv("GITHUB_TOKEN")
    api_version = os.getenv("API_VERSION", "2024-10-01-preview")
    if not endpoint or not token:
        raise TranslationRequestError("GitHub Models の接続情報が不足しています")

    return ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
        api_version=api_version,
    )

def get_azure_client() -> ChatCompletionsClient:
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
