"""翻訳処理のユーティリティとトークン計測。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Sequence

from azure.ai.inference.models import JsonSchemaFormat
from pydantic import BaseModel, ConfigDict, ValidationError
import tiktoken

from .exception import (
    TokenLimitExceededError,
    TranslationParseError,
    TranslationRequestError,
)
from logger import logger as root_logger

MAX_TOKENS = 8000
REQUEST_BUDGET = 7500
ENCODING_NAME = "o200k_base"


logger = root_logger.getChild("translation")


class TranslationEntryModel(BaseModel):
    index: int
    translation: str

    model_config = ConfigDict(extra="forbid")


class TranslationResponseModel(BaseModel):
    translations: List[TranslationEntryModel]

    model_config = ConfigDict(extra="forbid")


RESPONSE_JSON_FORMAT = JsonSchemaFormat(
    name="translation_response",
    schema=TranslationResponseModel.model_json_schema(),
    strict=True,
)


def count_tokens(text: str, encoding_name: str = ENCODING_NAME) -> int:
    """文字列のトークン数を計測する。"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def estimate_request_tokens(system_prompt: str, texts: Sequence[str]) -> int:
    """システムプロンプトとユーザメッセージの合計トークン数を推定する。"""
    total = count_tokens(system_prompt)
    for text in texts:
        total += count_tokens(text)
    return total


def truncate_messages(
    system_prompt: str,
    texts: Sequence[str],
    budget: int = REQUEST_BUDGET,
) -> List[str]:
    """トークン制限内で収まるテキスト群を返す。"""
    batch: List[str] = []
    for text in texts:
        candidate = batch + [text]
        tokens = estimate_request_tokens(system_prompt, candidate)
        if tokens > budget:
            if not batch:
                raise TokenLimitExceededError(
                    "単一テキストがトークン制限を超過しました",
                )
            break
        batch.append(text)
    return batch


@dataclass
class TranslationItem:
    """翻訳対象テキストとメタ情報を保持する。"""

    record: dict
    index: int


def translate_with_request(
    request_func: Callable[[str, Sequence[str]], str],
    system_prompt: str,
    texts: Sequence[str],
) -> List[str]:
    """リクエスト関数を使って翻訳結果を取得する。"""
    try:
        response = request_func(system_prompt, texts)
    except Exception as error:  # pragma: no cover - 実行時例外のラップ
        raise TranslationRequestError(str(error)) from error
    return _parse_translations(response, len(texts))


def stream_batches(
    items: Sequence[TranslationItem],
    *,
    system_prompt: str,
    budget: int = REQUEST_BUDGET,
) -> Iterator[List[TranslationItem]]:
    """トークン制限を考慮しながらバッチに分割する。"""
    buffer: List[TranslationItem] = []
    for item in items:
        candidate_texts = [it.record["text"] for it in buffer + [item]]
        tokens = estimate_request_tokens(system_prompt, candidate_texts)
        if tokens > budget:
            if not buffer:
                raise TokenLimitExceededError("単一テキストがトークン制限を超過しました")
            yield buffer
            buffer = [item]
            continue
        buffer.append(item)
    if buffer:
        yield buffer


def _parse_translations(response_text: str, expected: int) -> List[str]:
    """Pydanticモデルを使って翻訳結果を解析する。"""
    try:
        model = TranslationResponseModel.model_validate_json(response_text)
    except (ValidationError, TypeError, ValueError) as primary_error:
        try:
            raw = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise TranslationParseError(str(error)) from error
        if isinstance(raw, list):
            raw = {
                "translations": [
                    {
                        "index": index,
                        "translation": _extract_translation(entry),
                    }
                    for index, entry in enumerate(raw)
                ],
            }
        elif isinstance(raw, dict) and "translations" in raw:
            raw = {
                "translations": [
                    {
                        "index": item.get("index", idx),
                        "translation": _extract_translation(item),
                    }
                    for idx, item in enumerate(raw["translations"])
                ],
            }
        else:
            raise TranslationParseError("翻訳結果の形式が不正です") from primary_error
        try:
            model = TranslationResponseModel.model_validate(raw)
        except ValidationError as error:
            raise TranslationParseError(str(error)) from error
    sorted_entries = sorted(model.translations, key=lambda item: item.index)
    if len(sorted_entries) != expected:
        raise TranslationParseError(
            f"翻訳結果件数が一致しません expected={expected} actual={len(sorted_entries)}",
        )
    for expected_index, entry in enumerate(sorted_entries):
        if entry.index != expected_index:
            raise TranslationParseError(
                f"翻訳結果indexが不連続です expected={expected_index} actual={entry.index}",
            )
    translations = [entry.translation for entry in sorted_entries]
    logger.debug("翻訳結果を解析しました count=%d", len(translations))
    return translations


def _extract_translation(entry: object) -> str:
    if isinstance(entry, dict):
        if "translation" not in entry:
            raise TranslationParseError("翻訳結果の形式が不正です")
        return str(entry["translation"])
    return str(entry)
