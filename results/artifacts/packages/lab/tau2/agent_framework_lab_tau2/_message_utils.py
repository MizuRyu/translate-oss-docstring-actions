# Copyright (c) Microsoft. All rights reserved.

from agent_framework._types import ChatMessage, Contents, Role
from loguru import logger


def flip_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """アシスタントとユーザー間でメッセージの役割を入れ替えるロールプレイングシナリオ用。

    アシスタントのメッセージがユーザー入力になり、その逆も同様。
    アシスタントのメッセージをユーザーメッセージに反転する際は関数呼び出しをフィルタリング（通常ユーザーは関数呼び出しを行わないため）。

    """

    def filter_out_function_calls(messages: list[Contents]) -> list[Contents]:
        """メッセージ内容から関数呼び出しの内容を削除する。"""
        return [content for content in messages if content.type != "function_call"]

    flipped_messages = []
    for msg in messages:
        if msg.role == Role.ASSISTANT:
            # アシスタントからユーザーへ反転する
            contents = filter_out_function_calls(msg.contents)
            if contents:
                flipped_msg = ChatMessage(
                    role=Role.USER,
                    # 関数呼び出しはroleがuserの場合に400エラーを引き起こす
                    contents=contents,
                    author_name=msg.author_name,
                    message_id=msg.message_id,
                )
                flipped_messages.append(flipped_msg)
        elif msg.role == Role.USER:
            # ユーザーからアシスタントへ反転する
            flipped_msg = ChatMessage(
                role=Role.ASSISTANT, contents=msg.contents, author_name=msg.author_name, message_id=msg.message_id
            )
            flipped_messages.append(flipped_msg)
        elif msg.role == Role.TOOL:
            # ツールメッセージをスキップする
            pass
        else:
            # その他の役割はそのまま保持（system、toolなど）
            flipped_messages.append(msg)
    return flipped_messages


def log_messages(messages: list[ChatMessage]) -> None:
    """役割とコンテンツタイプに基づいて色付き出力でメッセージをログに記録。

    さまざまなメッセージ役割とコンテンツタイプを色分けして視覚的にデバッグを提供。
    HTMLのような文字をエスケープしてログのフォーマット問題を防止。

    """
    logger_ = logger.opt(colors=True)
    for msg in messages:
        # 異なるコンテンツタイプを処理する
        if hasattr(msg, "contents") and msg.contents:
            for content in msg.contents:
                if hasattr(content, "type"):
                    if content.type == "text":
                        escape_text = content.text.replace("<", r"\<")
                        if msg.role == Role.SYSTEM:
                            logger_.info(f"<cyan>[SYSTEM]</cyan> {escape_text}")
                        elif msg.role == Role.USER:
                            logger_.info(f"<green>[USER]</green> {escape_text}")
                        elif msg.role == Role.ASSISTANT:
                            logger_.info(f"<blue>[ASSISTANT]</blue> {escape_text}")
                        elif msg.role == Role.TOOL:
                            logger_.info(f"<yellow>[TOOL]</yellow> {escape_text}")
                        else:
                            logger_.info(f"<magenta>[{msg.role.value.upper()}]</magenta> {escape_text}")
                    elif content.type == "function_call":
                        function_call_text = f"{content.name}({content.arguments})"
                        function_call_text = function_call_text.replace("<", r"\<")
                        logger_.info(f"<yellow>[TOOL_CALL]</yellow> 🔧 {function_call_text}")
                    elif content.type == "function_result":
                        function_result_text = f"ID:{content.call_id} -> {content.result}"
                        function_result_text = function_result_text.replace("<", r"\<")
                        logger_.info(f"<yellow>[TOOL_RESULT]</yellow> 🔨 {function_result_text}")
                    else:
                        content_text = str(content).replace("<", r"\<")
                        logger_.info(f"<magenta>[{msg.role.value.upper()}] ({content.type})</magenta> {content_text}")
                else:
                    # タイプなしのコンテンツのフォールバック
                    text_content = str(content).replace("<", r"\<")
                    if msg.role == Role.SYSTEM:
                        logger_.info(f"<cyan>[SYSTEM]</cyan> {text_content}")
                    elif msg.role == Role.USER:
                        logger_.info(f"<green>[USER]</green> {text_content}")
                    elif msg.role == Role.ASSISTANT:
                        logger_.info(f"<blue>[ASSISTANT]</blue> {text_content}")
                    elif msg.role == Role.TOOL:
                        logger_.info(f"<yellow>[TOOL]</yellow> {text_content}")
                    else:
                        logger_.info(f"<magenta>[{msg.role.value.upper()}]</magenta> {text_content}")
        elif hasattr(msg, "text") and msg.text:
            # 単純なテキストメッセージを処理する
            text_content = msg.text.replace("<", r"\<")
            if msg.role == Role.SYSTEM:
                logger_.info(f"<cyan>[SYSTEM]</cyan> {text_content}")
            elif msg.role == Role.USER:
                logger_.info(f"<green>[USER]</green> {text_content}")
            elif msg.role == Role.ASSISTANT:
                logger_.info(f"<blue>[ASSISTANT]</blue> {text_content}")
            elif msg.role == Role.TOOL:
                logger_.info(f"<yellow>[TOOL]</yellow> {text_content}")
            else:
                logger_.info(f"<magenta>[{msg.role.value.upper()}]</magenta> {text_content}")
        else:
            # その他のメッセージ形式のフォールバック
            text_content = str(msg).replace("<", r"\<")
            logger_.info(f"<magenta>[{msg.role.value.upper()}]</magenta> {text_content}")
