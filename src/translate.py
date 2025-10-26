"""Azure AI Inference への翻訳リクエスト関数。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Sequence

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

from libcst_extractor.translation import RESPONSE_JSON_FORMAT, TranslationResponseModel
from logger import logger as root_logger


logger = root_logger.getChild("azure_translator")

load_dotenv()

ENDPOINT = os.environ.get("AZURE_INFERENCE_ENDPOINT", "https://models.github.ai/inference")
MODEL = os.environ.get("AZURE_INFERENCE_MODEL", "openai/gpt-4.1-mini")
TOKEN = os.environ.get("AZURE_INFERENCE_TOKEN") or os.environ.get("GITHUB_TOKEN")

if not TOKEN:
    raise RuntimeError("GITHUB_TOKEN もしくは AZURE_INFERENCE_TOKEN を設定してください")

CLIENT = ChatCompletionsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(TOKEN))
TEMP = float(os.environ.get("AZURE_INFERENCE_TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("AZURE_INFERENCE_TOP_P", "1"))


def request_translations(system_prompt: str, texts: Sequence[str]) -> str:
    """システムプロンプトと複数テキストをまとめて送信する。"""
    payload = _build_payload(texts)
    _dump_request(system_prompt, payload)
    messages = [SystemMessage(system_prompt), UserMessage(payload)]
    try:
        response = CLIENT.complete(
            messages=messages,
            temperature=TEMP,
            top_p=TOP_P,
            model=MODEL,
            response_format=RESPONSE_JSON_FORMAT,
        )
    except Exception as error:  # pragma: no cover - 実行時限定
        if "response_format" not in str(error):
            raise
        logger.debug("response_formatが未サポートのためフォールバックします: %s", error)
        response = CLIENT.complete(
            messages=messages,
            temperature=TEMP,
            top_p=TOP_P,
            model=MODEL,
        )
    content = _extract_content(response)
    try:
        TranslationResponseModel.model_validate_json(content)
    except Exception as error:  # pragma: no cover - デバッグ用途
        logger.debug("構造化レスポンスの検証に失敗しました: %s", error)
    _dump_response(content)
    return content


def _build_payload(texts: Sequence[str]) -> str:
    """JSON形式でまとめたユーザーメッセージを返す。"""
    items = [{"index": idx, "original": text} for idx, text in enumerate(texts)]
    return json.dumps({"items": items}, ensure_ascii=False)


def _extract_content(response) -> str:
    """LLMレスポンスから文字列を取り出す。"""
    choice = response.choices[0]
    content = choice.message.content
    if isinstance(content, list):
        return "".join(fragment.text for fragment in content if getattr(fragment, "text", ""))
    if isinstance(content, str):
        return content
    raise RuntimeError("LLMレスポンスが想定外の形式です")


def _dump_request(system_prompt: str, payload: str) -> None:
    """直近のリクエスト内容をテキストファイルに出力する。"""
    output_dir = Path(".context")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "system_prompt": system_prompt,
        "user_payload": json.loads(payload),
    }
    target = output_dir / "last_request_payload.json"
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _dump_response(raw: str) -> None:
    output_dir = Path(".context")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "last_response_payload.json"
    try:
        parsed: Dict[str, object] = json.loads(raw)
    except json.JSONDecodeError:
        target.write_text(raw, encoding="utf-8")
        return
    target.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
