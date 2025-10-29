# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

from collections.abc import Awaitable, Callable

from agent_framework import AgentMiddleware, AgentRunContext, ChatContext, ChatMiddleware
from agent_framework._logging import get_logger
from azure.core.credentials import TokenCredential
from azure.core.credentials_async import AsyncTokenCredential

from ._client import PurviewClient
from ._models import Activity
from ._processor import ScopedContentProcessor
from ._settings import PurviewSettings

logger = get_logger("agent_framework.purview")


class PurviewPolicyMiddleware(AgentMiddleware):
    """Purviewポリシーをプロンプトとレスポンスに適用するAgent middleware。

    同期のTokenCredentialまたはAsyncTokenCredentialのいずれかを受け入れます。

    使用例:

    .. code-block:: python
    from agent_framework.microsoft import PurviewPolicyMiddleware, PurviewSettings
    from agent_framework import ChatAgent

    credential = ...  # TokenCredential または AsyncTokenCredential
    settings = PurviewSettings(app_name="My App")
    agent = ChatAgent(
        chat_client=client, instructions="...", middleware=[PurviewPolicyMiddleware(credential, settings)]
    )
    """

    def __init__(
        self,
        credential: TokenCredential | AsyncTokenCredential,
        settings: PurviewSettings,
    ) -> None:
        self._client = PurviewClient(credential, settings)
        self._processor = ScopedContentProcessor(self._client, settings)
        self._settings = settings

    async def process(
        self,
        context: AgentRunContext,
        next: Callable[[AgentRunContext], Awaitable[None]],
    ) -> None:  # type: ignore[override]
        resolved_user_id: str | None = None
        try:
            # 事前（プロンプト）チェック
            should_block_prompt, resolved_user_id = await self._processor.process_messages(
                context.messages, Activity.UPLOAD_TEXT
            )
            if should_block_prompt:
                from agent_framework import AgentRunResponse, ChatMessage, Role

                context.result = AgentRunResponse(
                    messages=[ChatMessage(role=Role.SYSTEM, text=self._settings.blocked_prompt_message)]
                )
                context.terminate = True
                return
        except Exception as ex:
            # 事前チェックでエラーがあってもログを記録して継続します。
            logger.error(f"Error in Purview policy pre-check: {ex}")

        await next(context)

        try:
            # 通常のAgentRunResponseがある場合のみ事後（レスポンス）チェックを行います。
            # レスポンス評価にはリクエストと同じuser_idを使用します。
            if context.result and not context.is_streaming:
                should_block_response, _ = await self._processor.process_messages(
                    context.result.messages,  # type: ignore[union-attr]
                    Activity.UPLOAD_TEXT,
                    user_id=resolved_user_id,
                )
                if should_block_response:
                    from agent_framework import AgentRunResponse, ChatMessage, Role

                    context.result = AgentRunResponse(
                        messages=[ChatMessage(role=Role.SYSTEM, text=self._settings.blocked_response_message)]
                    )
            else:
                # 事後チェックではストリーミングレスポンスはサポートされていません。
                logger.debug("Streaming responses are not supported for Purview policy post-checks")
        except Exception as ex:
            # 事後チェックでエラーがあってもログを記録して継続します。
            logger.error(f"Error in Purview policy post-check: {ex}")


class PurviewChatPolicyMiddleware(ChatMiddleware):
    """Purviewポリシー評価用のChat middlewareバリアント。

    これによりユーザーはPurviewの強制をチャットクライアントに直接アタッチできます。

    動作:
    * チャット前: 送信される（ユーザー＋コンテキスト）メッセージをアップロードアクティビティとして評価し、ブロックされた場合は実行を終了できます。
    * チャット後: 受信したレスポンスメッセージを評価します（現在ストリーミングはサポートされていません）。
    ブロックされたメッセージに置き換えることができます。評価中はリクエストと同じuser_idを使用し、一貫したユーザー識別を保証します。

    使用例:

    .. code-block:: python
    from agent_framework.microsoft import PurviewChatPolicyMiddleware, PurviewSettings
    from agent_framework import ChatClient

    credential = ...  # TokenCredential または AsyncTokenCredential
    settings = PurviewSettings(app_name="My App")
    client = ChatClient(..., middleware=[PurviewChatPolicyMiddleware(credential, settings)])
    """

    def __init__(
        self,
        credential: TokenCredential | AsyncTokenCredential,
        settings: PurviewSettings,
    ) -> None:
        self._client = PurviewClient(credential, settings)
        self._processor = ScopedContentProcessor(self._client, settings)
        self._settings = settings

    async def process(
        self,
        context: ChatContext,
        next: Callable[[ChatContext], Awaitable[None]],
    ) -> None:  # type: ignore[override]
        resolved_user_id: str | None = None
        try:
            should_block_prompt, resolved_user_id = await self._processor.process_messages(
                context.messages, Activity.UPLOAD_TEXT
            )
            if should_block_prompt:
                from agent_framework import ChatMessage

                context.result = [  # type: ignore[assignment]
                    ChatMessage(role="system", text=self._settings.blocked_prompt_message)
                ]
                context.terminate = True
                return
        except Exception as ex:
            logger.error(f"Error in Purview policy pre-check: {ex}")

        await next(context)

        try:
            # 非ストリーミングでメッセージ結果の形状がある場合のみ事後（レスポンス）評価を行います。
            # レスポンス評価にはリクエストと同じuser_idを使用します。
            if context.result and not context.is_streaming:
                result_obj = context.result
                messages = getattr(result_obj, "messages", None)
                if messages:
                    should_block_response, _ = await self._processor.process_messages(
                        messages, Activity.UPLOAD_TEXT, user_id=resolved_user_id
                    )
                    if should_block_response:
                        from agent_framework import ChatMessage

                        context.result = [  # type: ignore[assignment]
                            ChatMessage(role="system", text=self._settings.blocked_response_message)
                        ]
            else:
                logger.debug("Streaming responses are not supported for Purview policy post-checks")
        except Exception as ex:
            logger.error(f"Error in Purview policy post-check: {ex}")
