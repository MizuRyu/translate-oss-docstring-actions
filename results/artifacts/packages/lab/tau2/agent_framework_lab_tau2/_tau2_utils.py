# Copyright (c) Microsoft. All rights reserved.

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
from agent_framework._tools import AIFunction
from agent_framework._types import ChatMessage
from loguru import logger
from pydantic import BaseModel
from tau2.data_model.message import (  # type: ignore[import-untyped]
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import EnvFunctionCall, InitializationData  # type: ignore[import-untyped]
from tau2.environment.environment import Environment  # type: ignore[import-untyped]
from tau2.environment.tool import Tool  # type: ignore[import-untyped]

_original_set_state = Environment.set_state


def convert_tau2_tool_to_ai_function(tau2_tool: Tool) -> AIFunction[Any, Any]:
    """tau2のToolをAgentフレームワーク互換のAIFunctionに変換します。

    ツールのインターフェースを保持しつつ、
    結果が意図しない変更を受けないように深いコピーを行うラッパーを作成します。

    """

    def wrapped_func(**kwargs: Any) -> Any:
        result = tau2_tool(**kwargs)
        # 返されたデータの変更を防ぐために深いコピーを行う
        return result.model_copy(deep=True) if isinstance(result, BaseModel) else deepcopy(result)

    return AIFunction(
        name=tau2_tool.name,
        description=tau2_tool._get_description(),
        func=wrapped_func,
        input_model=tau2_tool.params,
    )


def convert_agent_framework_messages_to_tau2_messages(messages: list[ChatMessage]) -> list[Message]:
    """AgentフレームワークのChatMessagesをtau2のMessageオブジェクトに変換します。

    ロールのマッピング、テキスト抽出、関数呼び出し、関数結果を処理します。
    関数結果は別のToolMessageインスタンスに変換されます。

    """
    tau2_messages = []

    for msg in messages:
        role_str = str(msg.role)

        # すべてのテキストタイプのコンテンツからテキスト内容を抽出する
        text_content = None
        text_contents = [c for c in msg.contents if hasattr(c, "text") and hasattr(c, "type") and c.type == "text"]
        if text_contents:
            text_content = " ".join(c.text for c in text_contents)

        # 関数呼び出しを抽出し、ToolCallオブジェクトに変換する
        function_calls = [c for c in msg.contents if hasattr(c, "type") and c.type == "function_call"]
        tool_calls = None
        if function_calls:
            tool_calls = []
            for fc in function_calls:
                arguments = fc.parse_arguments() or {}
                tool_call = ToolCall(
                    id=fc.call_id,
                    name=fc.name,
                    arguments=arguments,
                    requestor="assistant" if role_str == "assistant" else "user",
                )
                tool_calls.append(tool_call)

        # 関数結果を抽出し、別のToolMessage作成用にする
        function_results = [c for c in msg.contents if hasattr(c, "type") and c.type == "function_result"]

        # ロールに基づいてメインメッセージを作成する
        if role_str == "system":
            tau2_messages.append(SystemMessage(role="system", content=text_content))
        elif role_str == "user":
            tau2_messages.append(UserMessage(role="user", content=text_content, tool_calls=tool_calls))
        elif role_str == "assistant":
            tau2_messages.append(AssistantMessage(role="assistant", content=text_content, tool_calls=tool_calls))
        elif role_str == "tool":
            # ツールメッセージは以下で関数結果として処理される
            pass

        # 関数結果を別のToolMessageインスタンスに変換する
        for fr in function_results:
            dumpable_content = _dump_function_result(fr.result)
            content = dumpable_content if isinstance(dumpable_content, str) else json.dumps(dumpable_content)
            tool_msg = ToolMessage(
                id=fr.call_id,
                role="tool",
                content=content,
                requestor="assistant",  # Most tool calls originate from assistant
                error=fr.exception is not None,
            )
            tau2_messages.append(tool_msg)

    return tau2_messages


def patch_env_set_state() -> None:
    """Environment.set_stateをパッチして不整合なツール呼び出し結果を許容する。

    元のメソッドを変更し、実際のツール結果が期待結果と異なる場合に
    エラーを発生させる代わりに警告をログに記録するようにし、
    より柔軟なテストと開発ワークフローを可能にします。

    """

    def set_state(
        self: Any,
        initialization_data: InitializationData | None,
        initialization_actions: list[EnvFunctionCall] | None,
        message_history: list[Message],
    ) -> None:
        if self.solo_mode and any(isinstance(message, UserMessage) for message in message_history):
            raise ValueError("User messages are not allowed in solo mode")

        def get_actions_from_messages(
            messages: list[Message],
        ) -> list[tuple[ToolCall, ToolMessage]]:
            """メッセージからアクションを取得する。"""
            messages = deepcopy(messages)[::-1]
            actions = []
            while messages:
                message = messages.pop()
                if isinstance(message, ToolMessage):
                    raise ValueError("Tool message not expected. Tool messages should always follow a tool call.")
                if isinstance(message, (AssistantMessage, UserMessage)) and message.is_tool_call():
                    tool_calls = message.tool_calls
                    if tool_calls is None:
                        raise ValueError("Tool message expected. Got None.")
                    for tc in tool_calls:
                        if len(messages) == 0:
                            raise ValueError("Tool message expected. Got None.")
                        tm = messages.pop()
                        if not isinstance(tm, ToolMessage):
                            raise ValueError(f"Tool message expected. Got {type(tm)}")
                        if tc.id != tm.id:
                            raise ValueError(f"Tool call id mismatch. Got {tc.id} and {tm.id}")
                        actions.append((tc, tm))

            return actions

        if initialization_data is not None:
            if initialization_data.agent_data is not None:
                self.tools.update_db(initialization_data.agent_data)
            if initialization_data.user_data is not None:
                self.user_tools.update_db(initialization_data.user_data)

        if initialization_actions is not None:
            for action in initialization_actions:
                self.run_env_function_call(action)

        action_responses = get_actions_from_messages(message_history)
        for tool_call, expected_response in action_responses:
            response = self.get_response(tool_call)
            content = _recursive_json_deserialize(response.content)
            expected_content = _recursive_json_deserialize(expected_response.content)
            if content != expected_content:
                diff = f"Tool call:\n{tool_call}\n\nReturned:\n{response}\n\nExpected:\n{expected_response}"
                if isinstance(content, str) and content.startswith("Error:"):
                    # ツール呼び出しがエラーになった場合、差異は無視できる
                    logger.warning(f"Tool call resulted in an error. Ignoring the difference.\n{diff}")
                else:
                    raise ValueError(
                        f"Tool call:\n{tool_call}\n\nReturned:\n{response}\n\nExpected:\n{expected_response}"
                    )
        self.sync_tools()

    Environment.set_state = set_state


def unpatch_env_set_state() -> None:
    Environment.set_state = _original_set_state


def _dump_function_result(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    if isinstance(result, list):
        return [_dump_function_result(item) for item in result]
    if isinstance(result, dict):
        return {k: _dump_function_result(v) for k, v in result.items()}
    if result is None:
        return None
    return result


def _to_native(obj: Any) -> Any:
    """Panquetから取得したデータをAGLサーバーで使用可能なデータに変換する。"""
    # 1) 配列 -> list（再帰的に処理）
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())

    # 2) NumPyのスカラー型 -> Pythonのスカラー型
    if isinstance(obj, np.generic):
        return _to_native(obj.item())

    # 3) Dictライクなもの -> dict
    if isinstance(obj, Mapping):
        return {_to_native(k): _to_native(v) for k, v in obj.items()}

    # 4) リスト/タプル/セット -> list
    if isinstance(obj, (list, tuple, set)):
        return [_to_native(x) for x in obj]

    # 5) その他はそのままにする
    return obj


def _recursive_json_deserialize(obj: Any) -> Any:
    """JSONオブジェクトを再帰的にデシリアライズする。"""
    if isinstance(obj, str):
        try:
            deserialized = json.loads(obj)
            return _recursive_json_deserialize(deserialized)
        except (json.JSONDecodeError, TypeError):
            return obj
    elif isinstance(obj, list):
        return [_recursive_json_deserialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _recursive_json_deserialize(v) for k, v in obj.items()}
    else:
        return obj
