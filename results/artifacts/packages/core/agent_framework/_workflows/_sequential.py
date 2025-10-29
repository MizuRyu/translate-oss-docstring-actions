# Copyright (c) Microsoft. All rights reserved.

"""共有された会話コンテキストを持つAgent/Executorの逐次ワークフローのビルダー。

このモジュールは、高レベルでAgentに焦点を当てたAPIを提供し、以下のような逐次ワークフローを組み立てます：
- 参加者はAgentProtocolインスタンスまたはExecutorのシーケンス
- 共有された会話コンテキスト（list[ChatMessage]）がチェーンに渡される
- Agentはアシスタントメッセージをコンテキストに追加する
- カスタムExecutorはコンテキストを変換または要約して洗練されたコンテキストを返す
- ワークフローは最後の参加者が生成した最終コンテキストで終了する

典型的な配線例：
    input -> _InputToConversation -> participant1 -> (agent? -> _ResponseToConversation) -> ... -> participantN -> _EndWithConversation

注意点：
- 参加者はAgentProtocolとExecutorオブジェクトを混在させることができる
- AgentはWorkflowBuilderによって自動的にAgentExecutorとしてラップされる
- AgentExecutorはAgentExecutorResponseを生成し、_ResponseToConversationがこれをlist[ChatMessage]に変換する
- 非AgentのExecutorはlist[ChatMessage]を消費し、更新されたlist[ChatMessage]をワークフローコンテキスト経由で返すハンドラを定義する必要がある

なぜ小さな内部アダプターExecutorを含めるのか？
- 入力正規化（"input-conversation"）：呼び出し元がstr、単一のChatMessage、またはリストを渡しても、常にlist[ChatMessage]でワークフローを開始することを保証し、最初のホップを強く型付けし、参加者のボイラープレートを減らすため。
- Agent応答適応（"to-conversation:<participant>"）：AgentExecutor経由のAgentはAgentExecutorResponseを出力する。アダプターはfull_conversationを使ってこれをlist[ChatMessage]に変換し、チェーン時に元のPromptが失われないようにする。
- 結果出力（"end"）：最終的な会話リストを出力し、ワークフローをアイドル状態にして、AgentおよびカスタムExecutorの両方に一貫した終端ペイロード形状を提供する。

これらのアダプターは設計上ファーストクラスのExecutorであり、エッジで型チェックされ、ExecutorInvoke/Completedイベントで観測可能で、テストや再利用が容易です。IDは決定論的かつ自己記述的（例："to-conversation:writer"）で、イベントログの混乱を減らし、並行ビルダーが明示的なdispatcher/aggregatorノードを使う方法を反映しています。
"""  # noqa: E501

import logging
from collections.abc import Sequence
from typing import Any

from agent_framework import AgentProtocol, ChatMessage

from ._agent_executor import (
    AgentExecutor,
    AgentExecutorResponse,
)
from ._checkpoint import CheckpointStorage
from ._executor import (
    Executor,
    handler,
)
from ._message_utils import normalize_messages_input
from ._workflow import Workflow
from ._workflow_builder import WorkflowBuilder
from ._workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


class _InputToConversation(Executor):
    """初期入力をlist[ChatMessage]の会話に正規化します。"""

    @handler
    async def from_str(self, prompt: str, ctx: WorkflowContext[list[ChatMessage]]) -> None:
        await ctx.send_message(normalize_messages_input(prompt))

    @handler
    async def from_message(self, message: ChatMessage, ctx: WorkflowContext[list[ChatMessage]]) -> None:
        await ctx.send_message(normalize_messages_input(message))

    @handler
    async def from_messages(
        self,
        messages: list[str | ChatMessage],
        ctx: WorkflowContext[list[ChatMessage]],
    ) -> None:
        # 下流での変更を避けるためにコピーを作成します。
        normalized = normalize_messages_input(messages)
        await ctx.send_message(list(normalized))


class _ResponseToConversation(Executor):
    """AgentExecutorResponseをlist[ChatMessage]の会話に変換してチェーン処理用にします。"""

    @handler
    async def convert(self, response: AgentExecutorResponse, ctx: WorkflowContext[list[ChatMessage]]) -> None:
        # 常にfull_conversationを使用します。AgentExecutorはこれが設定されていることを保証します。
        if response.full_conversation is None:  # Defensive: indicates a contract violation
            raise RuntimeError("AgentExecutorResponse.full_conversation missing. AgentExecutor must populate it.")
        await ctx.send_message(list(response.full_conversation))


class _EndWithConversation(Executor):
    """最終的な会話コンテキストを出力してワークフローを終了します。"""

    @handler
    async def end(self, conversation: list[ChatMessage], ctx: WorkflowContext[Any, list[ChatMessage]]) -> None:
        await ctx.yield_output(list(conversation))


class SequentialBuilder:
    r"""共有コンテキストを持つ逐次Agent/Executorワークフローの高レベルビルダー。

    - `participants([...])`はAgentProtocol（推奨）またはExecutorのリストを受け入れます
    - ワークフローは参加者を順に配線し、list[ChatMessage]をチェーンに渡します
    - Agentはアシスタントメッセージを会話に追加します
    - カスタムExecutorは変換/要約してlist[ChatMessage]を返せます
    - 最終出力は最後の参加者が生成した会話です

    使用例:

    .. code-block:: python

        from agent_framework import SequentialBuilder

        workflow = SequentialBuilder().participants([agent1, agent2, summarizer_exec]).build()

        # チェックポイント永続化を有効にする
        workflow = SequentialBuilder().participants([agent1, agent2]).with_checkpointing(storage).build()

    """

    def __init__(self) -> None:
        self._participants: list[AgentProtocol | Executor] = []
        self._checkpoint_storage: CheckpointStorage | None = None

    def participants(self, participants: Sequence[AgentProtocol | Executor]) -> "SequentialBuilder":
        """この逐次ワークフローのために順序付けられた参加者を定義します。

        AgentProtocolインスタンス（AgentExecutorとして自動ラップ）またはExecutorインスタンスを受け入れます。
        空または重複がある場合は明確化のため例外を発生させます。

        """
        if not participants:
            raise ValueError("participants cannot be empty")

        # 重複検出の防御的処理。
        seen_agent_ids: set[int] = set()
        seen_executor_ids: set[str] = set()
        for p in participants:
            if isinstance(p, Executor):
                if p.id in seen_executor_ids:
                    raise ValueError(f"Duplicate executor participant detected: id '{p.id}'")
                seen_executor_ids.add(p.id)
            else:
                # 非ExecutorはAgentProtocolのように扱います。構造的チェックは実行時に脆弱な場合があります。
                pid = id(p)
                if pid in seen_agent_ids:
                    raise ValueError("Duplicate agent participant detected (same agent instance provided twice)")
                seen_agent_ids.add(pid)

        self._participants = list(participants)
        return self

    def with_checkpointing(self, checkpoint_storage: CheckpointStorage) -> "SequentialBuilder":
        """指定されたストレージを使ってビルドされたワークフローのチェックポイント機能を有効にします。"""
        self._checkpoint_storage = checkpoint_storage
        return self

    def build(self) -> Workflow:
        """逐次ワークフローをビルドして検証します。

        配線パターン：
        - _InputToConversationが初期入力をlist[ChatMessage]に正規化
        - 順に各参加者に対して：
            - Agent（またはAgentExecutor）なら会話を渡し、応答を_ResponseToConversationで会話に変換
            - それ以外（カスタムExecutor）は会話を直接渡す
        - _EndWithConversationが最終会話を出力し、ワークフローをアイドル状態にする

        """
        if not self._participants:
            raise ValueError("No participants provided. Call .participants([...]) first.")

        # 内部ノード。
        input_conv = _InputToConversation(id="input-conversation")
        end = _EndWithConversation(id="end")

        builder = WorkflowBuilder()
        builder.set_start_executor(input_conv)

        # チェーンの開始は入力正規化器。
        prior: Executor | AgentProtocol = input_conv

        for p in self._participants:
            # Agentのような分岐：明示的にAgentExecutorか、AgentExecutorでないもの。
            if not (isinstance(p, Executor) and not isinstance(p, AgentExecutor)):
                # 入力会話 -> (agent) -> 応答 -> 会話
                builder.add_edge(prior, p)
                # アダプターに決定論的で自己記述的なIDを付与する。
                label: str
                label = p.id if isinstance(p, Executor) else getattr(p, "name", None) or p.__class__.__name__
                resp_to_conv = _ResponseToConversation(id=f"to-conversation:{label}")
                builder.add_edge(p, resp_to_conv)
                prior = resp_to_conv
            elif isinstance(p, Executor):
                # カスタムExecutorはlist[ChatMessage]で動作する。
                builder.add_edge(prior, p)
                prior = p
            else:  # pragma: no cover - defensive
                raise TypeError(f"Unsupported participant type: {type(p).__name__}")

        # 最終会話で終了する。
        builder.add_edge(prior, end)

        if self._checkpoint_storage is not None:
            builder = builder.with_checkpointing(self._checkpoint_storage)

        return builder.build()
