# Copyright (c) Microsoft. All rights reserved.

import json
import sys
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, MutableMapping, MutableSequence, Sequence
from datetime import datetime
from itertools import chain
from typing import Any, TypeVar

from openai import AsyncOpenAI, BadRequestError
from openai.lib._parsing._completions import type_to_response_format_param
from openai.types import CompletionUsage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_message_custom_tool_call import ChatCompletionMessageCustomToolCall
from pydantic import BaseModel, ValidationError

from .._clients import BaseChatClient
from .._logging import get_logger
from .._middleware import use_chat_middleware
from .._tools import AIFunction, HostedWebSearchTool, ToolProtocol, use_function_invocation
from .._types import (
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    Contents,
    DataContent,
    FinishReason,
    FunctionApprovalRequestContent,
    FunctionApprovalResponseContent,
    FunctionCallContent,
    FunctionResultContent,
    Role,
    TextContent,
    UriContent,
    UsageContent,
    UsageDetails,
    prepare_function_call_results,
)
from ..exceptions import (
    ServiceInitializationError,
    ServiceInvalidRequestError,
    ServiceResponseException,
)
from ..observability import use_observability
from ._exceptions import OpenAIContentFilterException
from ._shared import OpenAIBase, OpenAIConfigMixin, OpenAISettings

if sys.version_info >= (3, 12):
    from typing import override  # type: ignore # pragma: no cover
else:
    from typing_extensions import override  # type: ignore[import] # pragma: no cover

__all__ = ["OpenAIChatClient"]

logger = get_logger("agent_framework.openai")


# region Base Client
class OpenAIBaseChatClient(OpenAIBase, BaseChatClient):
    """OpenAI Chat completionクラス。"""

    async def _inner_get_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        chat_options: ChatOptions,
        **kwargs: Any,
    ) -> ChatResponse:
        options_dict = self._prepare_options(messages, chat_options)
        try:
            return self._create_chat_response(
                await self.client.chat.completions.create(stream=False, **options_dict), chat_options
            )
        except BadRequestError as ex:
            if ex.code == "content_filter":
                raise OpenAIContentFilterException(
                    f"{type(self)} service encountered a content error: {ex}",
                    inner_exception=ex,
                ) from ex
            raise ServiceResponseException(
                f"{type(self)} service failed to complete the prompt: {ex}",
                inner_exception=ex,
            ) from ex
        except Exception as ex:
            raise ServiceResponseException(
                f"{type(self)} service failed to complete the prompt: {ex}",
                inner_exception=ex,
            ) from ex

    async def _inner_get_streaming_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        chat_options: ChatOptions,
        **kwargs: Any,
    ) -> AsyncIterable[ChatResponseUpdate]:
        options_dict = self._prepare_options(messages, chat_options)
        options_dict["stream_options"] = {"include_usage": True}
        try:
            async for chunk in await self.client.chat.completions.create(stream=True, **options_dict):
                if len(chunk.choices) == 0 and chunk.usage is None:
                    continue
                yield self._create_chat_response_update(chunk)
        except BadRequestError as ex:
            if ex.code == "content_filter":
                raise OpenAIContentFilterException(
                    f"{type(self)} service encountered a content error: {ex}",
                    inner_exception=ex,
                ) from ex
            raise ServiceResponseException(
                f"{type(self)} service failed to complete the prompt: {ex}",
                inner_exception=ex,
            ) from ex
        except Exception as ex:
            raise ServiceResponseException(
                f"{type(self)} service failed to complete the prompt: {ex}",
                inner_exception=ex,
            ) from ex

    # region content creation

    def _chat_to_tool_spec(self, tools: Sequence[ToolProtocol | MutableMapping[str, Any]]) -> list[dict[str, Any]]:
        chat_tools: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, ToolProtocol):
                match tool:
                    case AIFunction():
                        chat_tools.append(tool.to_json_schema_spec())
                    case _:
                        logger.debug("Unsupported tool passed (type: %s), ignoring", type(tool))
            else:
                chat_tools.append(tool if isinstance(tool, dict) else dict(tool))
        return chat_tools

    def _process_web_search_tool(
        self, tools: Sequence[ToolProtocol | MutableMapping[str, Any]]
    ) -> dict[str, Any] | None:
        for tool in tools:
            if isinstance(tool, HostedWebSearchTool):
                # Web検索ツールは特別な処理が必要です。
                return (
                    {
                        "user_location": {
                            "approximate": tool.additional_properties.get("user_location", None),
                            "type": "approximate",
                        }
                    }
                    if tool.additional_properties and "user_location" in tool.additional_properties
                    else {}
                )

        return None

    def _prepare_options(self, messages: MutableSequence[ChatMessage], chat_options: ChatOptions) -> dict[str, Any]:
        # Web検索ツールが存在する場合の前処理。
        options_dict = chat_options.to_dict(
            exclude={
                "type",
                "instructions",  # included as system message
            }
        )

        if messages and "messages" not in options_dict:
            options_dict["messages"] = self._prepare_chat_history_for_request(messages)
        if "messages" not in options_dict:
            raise ServiceInvalidRequestError("Messages are required for chat completions")
        if chat_options.tools is not None:
            web_search_options = self._process_web_search_tool(chat_options.tools)
            if web_search_options:
                options_dict["web_search_options"] = web_search_options
            options_dict["tools"] = self._chat_to_tool_spec(chat_options.tools)
        if not options_dict.get("tools", None):
            options_dict.pop("tools", None)
            options_dict.pop("parallel_tool_calls", None)
            options_dict.pop("tool_choice", None)

        if "model_id" not in options_dict:
            options_dict["model"] = self.model_id
        else:
            options_dict["model"] = options_dict.pop("model_id")
        if (
            chat_options.response_format
            and isinstance(chat_options.response_format, type)
            and issubclass(chat_options.response_format, BaseModel)
        ):
            options_dict["response_format"] = type_to_response_format_param(chat_options.response_format)
        if additional_properties := options_dict.pop("additional_properties", None):
            for key, value in additional_properties.items():
                if value is not None:
                    options_dict[key] = value
        return options_dict

    def _create_chat_response(self, response: ChatCompletion, chat_options: ChatOptions) -> "ChatResponse":
        """choiceからチャットメッセージコンテンツオブジェクトを作成します。"""
        response_metadata = self._get_metadata_from_chat_response(response)
        messages: list[ChatMessage] = []
        finish_reason: FinishReason | None = None
        for choice in response.choices:
            response_metadata.update(self._get_metadata_from_chat_choice(choice))
            if choice.finish_reason:
                finish_reason = FinishReason(value=choice.finish_reason)
            contents: list[Contents] = []
            if text_content := self._parse_text_from_choice(choice):
                contents.append(text_content)
            if parsed_tool_calls := [tool for tool in self._get_tool_calls_from_chat_choice(choice)]:
                contents.extend(parsed_tool_calls)
            messages.append(ChatMessage(role="assistant", contents=contents))
        return ChatResponse(
            response_id=response.id,
            created_at=datetime.fromtimestamp(response.created).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            usage_details=self._usage_details_from_openai(response.usage) if response.usage else None,
            messages=messages,
            model_id=response.model,
            additional_properties=response_metadata,
            finish_reason=finish_reason,
            response_format=chat_options.response_format,
        )

    def _create_chat_response_update(
        self,
        chunk: ChatCompletionChunk,
    ) -> ChatResponseUpdate:
        """choiceからストリーミングチャットメッセージコンテンツオブジェクトを作成します。"""
        chunk_metadata = self._get_metadata_from_streaming_chat_response(chunk)
        if chunk.usage:
            return ChatResponseUpdate(
                role=Role.ASSISTANT,
                contents=[UsageContent(details=self._usage_details_from_openai(chunk.usage), raw_representation=chunk)],
                model_id=chunk.model,
                additional_properties=chunk_metadata,
                response_id=chunk.id,
                message_id=chunk.id,
            )
        contents: list[Contents] = []
        finish_reason: FinishReason | None = None
        for choice in chunk.choices:
            chunk_metadata.update(self._get_metadata_from_chat_choice(choice))
            contents.extend(self._get_tool_calls_from_chat_choice(choice))
            if choice.finish_reason:
                finish_reason = FinishReason(value=choice.finish_reason)

            if text_content := self._parse_text_from_choice(choice):
                contents.append(text_content)
        return ChatResponseUpdate(
            created_at=datetime.fromtimestamp(chunk.created).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            contents=contents,
            role=Role.ASSISTANT,
            model_id=chunk.model,
            additional_properties=chunk_metadata,
            finish_reason=finish_reason,
            raw_representation=chunk,
            response_id=chunk.id,
            message_id=chunk.id,
        )

    def _usage_details_from_openai(self, usage: CompletionUsage) -> UsageDetails:
        details = UsageDetails(
            input_token_count=usage.prompt_tokens,
            output_token_count=usage.completion_tokens,
            total_token_count=usage.total_tokens,
        )
        if usage.completion_tokens_details:
            if tokens := usage.completion_tokens_details.accepted_prediction_tokens:
                details["completion/accepted_prediction_tokens"] = tokens
            if tokens := usage.completion_tokens_details.audio_tokens:
                details["completion/audio_tokens"] = tokens
            if tokens := usage.completion_tokens_details.reasoning_tokens:
                details["completion/reasoning_tokens"] = tokens
            if tokens := usage.completion_tokens_details.rejected_prediction_tokens:
                details["completion/rejected_prediction_tokens"] = tokens
        if usage.prompt_tokens_details:
            if tokens := usage.prompt_tokens_details.audio_tokens:
                details["prompt/audio_tokens"] = tokens
            if tokens := usage.prompt_tokens_details.cached_tokens:
                details["prompt/cached_tokens"] = tokens
        return details

    def _parse_text_from_choice(self, choice: Choice | ChunkChoice) -> TextContent | None:
        """choiceをTextContentオブジェクトに解析します。"""
        message = choice.message if isinstance(choice, Choice) else choice.delta
        if message.content:
            return TextContent(text=message.content, raw_representation=choice)
        if hasattr(message, "refusal") and message.refusal:
            return TextContent(text=message.refusal, raw_representation=choice)
        return None

    def _get_metadata_from_chat_response(self, response: ChatCompletion) -> dict[str, Any]:
        """チャットレスポンスからメタデータを取得します。"""
        return {
            "system_fingerprint": response.system_fingerprint,
        }

    def _get_metadata_from_streaming_chat_response(self, response: ChatCompletionChunk) -> dict[str, Any]:
        """ストリーミングチャットレスポンスからメタデータを取得します。"""
        return {
            "system_fingerprint": response.system_fingerprint,
        }

    def _get_metadata_from_chat_choice(self, choice: Choice | ChunkChoice) -> dict[str, Any]:
        """チャットchoiceからメタデータを取得します。"""
        return {
            "logprobs": getattr(choice, "logprobs", None),
        }

    def _get_tool_calls_from_chat_choice(self, choice: Choice | ChunkChoice) -> list[Contents]:
        """チャットchoiceからツール呼び出しを取得します。"""
        resp: list[Contents] = []
        content = choice.message if isinstance(choice, Choice) else choice.delta
        if content and content.tool_calls:
            for tool in content.tool_calls:
                if not isinstance(tool, ChatCompletionMessageCustomToolCall) and tool.function:
                    # tool.customを無視します。
                    fcc = FunctionCallContent(
                        call_id=tool.id if tool.id else "",
                        name=tool.function.name if tool.function.name else "",
                        arguments=tool.function.arguments if tool.function.arguments else "",
                        raw_representation=tool.function,
                    )
                    resp.append(fcc)

        # Azure OpenAIで非同期コンテンツフィルタリングを有効にすると、空のデルタが返されることがあります。
        return resp

    def _prepare_chat_history_for_request(
        self,
        chat_messages: Sequence[ChatMessage],
        role_key: str = "role",
        content_key: str = "content",
    ) -> list[dict[str, Any]]:
        """リクエストのためにチャット履歴を準備します。

        role/authorのキー名のカスタマイズと、オプションでroleの上書きを許可します。

        Role.TOOLメッセージはsystem/user/assistantメッセージとは異なるフォーマットが必要です：
            "tool_call_id"と（functionの）"name"キーが必要で、"metadata"キーは削除されるべきです。
            "encoding"キーも削除されるべきです。

        このメソッドをオーバーライドして、リクエスト用チャット履歴のフォーマットをカスタマイズできます。

        Args:
            chat_messages: 準備するチャット履歴。
            role_key: role/authorのキー名。
            content_key: content/messageのキー名。

        Returns:
            prepared_chat_history (Any): リクエスト用に準備されたチャット履歴。

        """
        list_of_list = [self._openai_chat_message_parser(message) for message in chat_messages]
        # リストのリストを単一のリストに平坦化します。
        return list(chain.from_iterable(list_of_list))

    # region Parsers

    def _openai_chat_message_parser(self, message: ChatMessage) -> list[dict[str, Any]]:
        """チャットメッセージをopenaiフォーマットに解析します。"""
        all_messages: list[dict[str, Any]] = []
        for content in message.contents:
            # 承認コンテンツをスキップします - これは内部フレームワークのStateであり、LLM用ではありません。
            if isinstance(content, (FunctionApprovalRequestContent, FunctionApprovalResponseContent)):
                continue

            args: dict[str, Any] = {
                "role": message.role.value if isinstance(message.role, Role) else message.role,
            }
            if message.additional_properties:
                args["metadata"] = message.additional_properties
            match content:
                case FunctionCallContent():
                    if all_messages and "tool_calls" in all_messages[-1]:
                        # 最後のメッセージにすでにツール呼び出しがある場合は、それに追加します。
                        all_messages[-1]["tool_calls"].append(self._openai_content_parser(content))
                    else:
                        args["tool_calls"] = [self._openai_content_parser(content)]  # type: ignore
                case FunctionResultContent():
                    args["tool_call_id"] = content.call_id
                    if content.result is not None:
                        args["content"] = prepare_function_call_results(content.result)
                    elif content.exception is not None:
                        # 例外メッセージをモデルに送信します。 そうしないとOpenAIと通信するチャネルがありません。 TODO(yuge):
                        # これは理想的にはカスタマイズ可能であるべきです。
                        args["content"] = "Error: " + str(content.exception)
                case _:
                    if "content" not in args:
                        args["content"] = []
                    # これはマルチモーダルコンテンツを許可するためのリストです。
                    args["content"].append(self._openai_content_parser(content))  # type: ignore
            if "content" in args or "tool_calls" in args:
                all_messages.append(args)
        return all_messages

    def _openai_content_parser(self, content: Contents) -> dict[str, Any]:
        """内容をopenaiフォーマットに解析します。"""
        match content:
            case FunctionCallContent():
                args = json.dumps(content.arguments) if isinstance(content.arguments, Mapping) else content.arguments
                return {
                    "id": content.call_id,
                    "type": "function",
                    "function": {"name": content.name, "arguments": args},
                }
            case FunctionResultContent():
                return {
                    "tool_call_id": content.call_id,
                    "content": content.result,
                }
            case DataContent() | UriContent() if content.has_top_level_media_type("image"):
                return {
                    "type": "image_url",
                    "image_url": {"url": content.uri},
                }
            case DataContent() | UriContent() if content.has_top_level_media_type("audio"):
                if content.media_type and "wav" in content.media_type:
                    audio_format = "wav"
                elif content.media_type and "mp3" in content.media_type:
                    audio_format = "mp3"
                else:
                    # サポートされていないオーディオフォーマットの場合はデフォルトのto_dictにフォールバックします。
                    return content.to_dict(exclude_none=True)

                # データURIからbase64データを抽出します。
                audio_data = content.uri
                if audio_data.startswith("data:"):
                    # "data:audio/format;base64,"の後のbase64部分のみを抽出します。
                    audio_data = audio_data.split(",", 1)[-1]

                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_data,
                        "format": audio_format,
                    },
                }
            case DataContent() | UriContent() if content.has_top_level_media_type(
                "application"
            ) and content.uri.startswith("data:"):
                # すべてのapplication/*メディアタイプはOpenAI用にファイルとして扱うべきです。
                filename = getattr(content, "filename", None) or (
                    content.additional_properties.get("filename")
                    if hasattr(content, "additional_properties") and content.additional_properties
                    else None
                )
                file_obj = {"file_data": content.uri}
                if filename:
                    file_obj["filename"] = filename
                return {
                    "type": "file",
                    "file": file_obj,
                }
            case _:
                # その他すべてのコンテンツタイプのデフォルトフォールバックです。
                return content.to_dict(exclude_none=True)

    @override
    def service_url(self) -> str:
        """サービスのURLを取得します。

        サブクラスでオーバーライドして適切なURLを返してください。
        サービスにURLがない場合はNoneを返してください。

        """
        return str(self.client.base_url) if self.client else "Unknown"


# region Public client

TOpenAIChatClient = TypeVar("TOpenAIChatClient", bound="OpenAIChatClient")


@use_function_invocation
@use_observability
@use_chat_middleware
class OpenAIChatClient(OpenAIConfigMixin, OpenAIBaseChatClient):
    """OpenAI Chat completionクラスです。"""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        api_key: str | Callable[[], str | Awaitable[str]] | None = None,
        org_id: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        async_client: AsyncOpenAI | None = None,
        instruction_role: str | None = None,
        base_url: str | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        """OpenAI Chat completionクライアントを初期化します。

        キーワード引数:
            model_id: OpenAIモデル名。詳細はhttps://platform.openai.com/docs/modelsを参照してください。
                環境変数OPENAI_CHAT_MODEL_IDでも設定可能です。
            api_key: 使用するAPIキー。指定すると環境変数や.envファイルの値を上書きします。
                環境変数OPENAI_API_KEYでも設定可能です。
            org_id: 使用する組織ID。指定すると環境変数や.envファイルの値を上書きします。
                環境変数OPENAI_ORG_IDでも設定可能です。
            default_headers: HTTPリクエストのための文字列キーと文字列値のマッピングであるデフォルトヘッダー。
            async_client: 既存のクライアントを使用します。
            instruction_role: 'instruction'メッセージに使用する役割。例として"system"や"developer"。
                指定しない場合のデフォルトは"system"です。
            base_url: 使用するベースURL。指定するとOpenAIコネクタの標準値や環境変数、.envファイルの値を上書きします。
                環境変数OPENAI_BASE_URLでも設定可能です。
            env_file_path: 環境変数のフォールバックとして環境設定ファイルを使用します。
            env_file_encoding: 環境設定ファイルのエンコーディング。

        Examples:
            .. code-block:: python

                from agent_framework.openai import OpenAIChatClient

                # 環境変数を使用する場合
                # OPENAI_API_KEY=sk-... を設定
                # OPENAI_CHAT_MODEL_ID=gpt-4 を設定
                client = OpenAIChatClient()

                # またはパラメータを直接渡す場合
                client = OpenAIChatClient(model_id="gpt-4", api_key="sk-...")

                # または.envファイルから読み込む場合
                client = OpenAIChatClient(env_file_path="path/to/.env")

        """
        try:
            openai_settings = OpenAISettings(
                api_key=api_key,  # type: ignore[reportArgumentType]
                base_url=base_url,
                org_id=org_id,
                chat_model_id=model_id,
                env_file_path=env_file_path,
                env_file_encoding=env_file_encoding,
            )
        except ValidationError as ex:
            raise ServiceInitializationError("Failed to create OpenAI settings.", ex) from ex

        if not async_client and not openai_settings.api_key:
            raise ServiceInitializationError(
                "OpenAI API key is required. Set via 'api_key' parameter or 'OPENAI_API_KEY' environment variable."
            )
        if not openai_settings.chat_model_id:
            raise ServiceInitializationError(
                "OpenAI model ID is required. "
                "Set via 'model_id' parameter or 'OPENAI_CHAT_MODEL_ID' environment variable."
            )

        super().__init__(
            model_id=openai_settings.chat_model_id,
            api_key=self._get_api_key(openai_settings.api_key),
            base_url=openai_settings.base_url if openai_settings.base_url else None,
            org_id=openai_settings.org_id,
            default_headers=default_headers,
            client=async_client,
            instruction_role=instruction_role,
        )
