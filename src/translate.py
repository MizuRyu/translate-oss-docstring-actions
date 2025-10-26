from __future__ import annotations

import json
import os


from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, ValidationError
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, JsonSchemaFormat
from azure.core.credentials import AzureKeyCredential

from dotenv import load_dotenv

from util import logger
from exception import TranslationParseError, TranslationRequestError


load_dotenv()

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

DEFAULT_PROMPT = """あなたは、OSSの日本人コントリビューターであり、翻訳者です。\nユーザーが提示した、英語で書かれたdocstringやコメントを、ルールに従い、日本語に翻訳してください。\n翻訳結果は、出力フォーマットに従い、JSON形式で出力してください。\n\n### ルール\n- 固有名詞は翻訳せずに残してください。\n- 変数名やPythonの構文は翻訳しないでください。\n- docstring内の見出し（Examples など）は英語のままにしてください。\n- 以下の単語はカタカナや和訳にせず、そのまま英語で書いてください。\n  Agent, ID, Thread, Chat, Client, Class, Context, Import, Export, Key, Token, Secret, Config, Prompt, Request, Response, State, Message, Optional, None, Middleware, Executor\n\n### 出力フォーマット\n{\n  \"translations\": [\n    {\n      \"index\": <入力index>,\n      \"translation\": \"<翻訳結果>\"\n    }\n  ]\n}\n"""


@dataclass
class Settings:
    input_path: Path
    output_path: Path
    failed_output: Path
    limit: Optional[int]
    system_prompt: str
    batch_size: int
    translator_kind: str


def run(settings: Dict[str, Any]) -> None:
    cfg = Settings(
        input_path=Path(settings["input"]).resolve(),
        output_path=Path(settings["output"]).resolve(),
        failed_output=Path(settings["failed_output"]).resolve(),
        limit=settings.get("limit"),
        system_prompt=settings.get("system_prompt", DEFAULT_PROMPT),
        batch_size=settings.get("batch_size") or 5,
        translator_kind=settings.get("translator_kind", "azure"),
    )

    entries = _load_entries(cfg.input_path, cfg.limit)
    if not entries:
        logger.info("翻訳対象がありません input=%s", cfg.input_path)
        return

    request = _build_request(cfg.translator_kind)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.failed_output.parent.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    success_count = 0
    request_count = 0
    failure_items: List[Dict[str, Any]] = []

    with cfg.output_path.open("w", encoding="utf-8") as success, cfg.failed_output.open(
        "w", encoding="utf-8"
    ) as failed:
        for chunk in _chunk(entries, cfg.batch_size):
            texts = [entry["text"] for entry in chunk]
            try:
                request_count += 1
                translations = _request_translations(cfg.system_prompt, texts, request)
                logger.info("翻訳リクエスト完了 batch=%d total=%d", len(chunk), request_count)
            except TranslationRequestError as error:
                logger.warning("翻訳要求に失敗しました件数=%d error=%s", len(chunk), error)
                failure_items.extend(chunk)
                continue
            except TranslationParseError as error:
                logger.warning("翻訳結果の解析に失敗しました件数=%d error=%s", len(chunk), error)
                failure_items.extend(chunk)
                continue
            if len(translations) != len(chunk):
                logger.warning("翻訳結果件数不一致 input=%d output=%d", len(chunk), len(translations))
                failure_items.extend(chunk)
                continue
            for entry, translated in zip(chunk, translations):
                payload = {
                    "path": entry["path"],
                    "kind": entry["kind"],
                    "original": entry["text"],
                    "translated": translated,
                    "meta": entry["meta"],
                }
                success.write(json.dumps(payload, ensure_ascii=False))
                success.write("\n")
                success_count += 1
        for entry in failure_items:
            failed.write(json.dumps(entry, ensure_ascii=False))
            failed.write("\n")

    duration = perf_counter() - start
    logger.info(
        "翻訳完了 成功:%d 失敗:%d リクエスト回数:%d 処理時間:%.3f秒",
        success_count,
        len(failure_items),
        request_count,
        duration,
    )


def _load_entries(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if limit is not None and line_number >= limit:
                break
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _chunk(entries: Sequence[Dict[str, Any]], size: int) -> Iterator[List[Dict[str, Any]]]:
    if size <= 0:
        size = 5
    for start in range(0, len(entries), size):
        yield list(entries[start : start + size])


def _build_request(kind: str):
    if kind == "dummy":
        def request(_: str, texts: Sequence[str]) -> List[str]:
            return [f"{text} (mock)" for text in texts]
        return request
    if kind == "azure":
        endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT", "https://models.github.ai/inference")
        model = os.getenv("AZURE_INFERENCE_MODEL", "openai/gpt-4.1-mini")
        token = os.getenv("AZURE_INFERENCE_TOKEN") or os.getenv("GITHUB_TOKEN")
        if not token:
            raise RuntimeError("AZURE_INFERENCE_TOKEN もしくは GITHUB_TOKEN を設定してください")
        api_version = "2024-10-01-preview"
        temperature = float(os.getenv("AZURE_INFERENCE_TEMPERATURE", "0.2"))
        top_p = float(os.getenv("AZURE_INFERENCE_TOP_P", "1.0"))

        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token),
            api_version=api_version,
        )

        def request(system_prompt: str, texts: Sequence[str]) -> List[str]:
            payload = _build_payload(texts)
            try:
                response = client.complete(
                    model=model,
                    messages=[SystemMessage(system_prompt), UserMessage(payload)],
                    temperature=temperature,
                    top_p=top_p,
                    response_format=RESPONSE_FORMAT,
                )
            except Exception as error:  # pragma: no cover
                raise TranslationRequestError(str(error)) from error
            content = _extract_content(response)
            return _parse_translations(content)

        return request
    raise ValueError(f"未対応のtranslator-kindです: {kind}")


def _build_payload(texts: Sequence[str]) -> str:
    items = [{"index": index, "original": text} for index, text in enumerate(texts)]
    return json.dumps({"items": items}, ensure_ascii=False)


def _extract_content(response) -> str:
    choice = response.choices[0]
    content = choice.message.content
    if isinstance(content, list):
        return "".join(getattr(fragment, "text", "") for fragment in content)
    if isinstance(content, str):
        return content
    raise TranslationParseError("LLMレスポンスが想定外の形式です")


def _request_translations(system_prompt: str, texts: Sequence[str], request) -> List[str]:
    try:
        response_text = request(system_prompt, texts)
        if isinstance(response_text, list):
            return response_text
        return _parse_translations(response_text)
    except TranslationRequestError:
        raise
    except Exception as error:
        raise TranslationRequestError(str(error)) from error


def _parse_translations(response_text: str) -> List[str]:
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
        raise TranslationParseError(
            f"翻訳結果のindexが不正です expected={expected} actual={actual}"
        )
