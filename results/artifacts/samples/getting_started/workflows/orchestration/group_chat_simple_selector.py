# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging

from agent_framework import ChatAgent, GroupChatBuilder, GroupChatStateSnapshot, WorkflowOutputEvent
from agent_framework.openai import OpenAIChatClient

logging.basicConfig(level=logging.INFO)

"""
Sample: Group Chat with Simple Speaker Selector Function

What it does:
- Demonstrates the select_speakers() API for GroupChat orchestration
- Uses a pure Python function to control speaker selection based on conversation state
- Alternates between researcher and writer agents in a simple round-robin pattern
- Shows how to access conversation history, round index, and participant metadata

Key pattern:
    def select_next_speaker(state: GroupChatStateSnapshot) -> str | None:
        # state contains: task, participants, conversation, history, round_index
        # Return participant name to continue, or None to finish
        ...

Prerequisites:
- OpenAI environment variables configured for OpenAIChatClient
"""


def select_next_speaker(state: GroupChatStateSnapshot) -> str | None:
    """研究者とライターが交互に話すシンプルなスピーカーセレクター。

    この関数はコアパターンを示します：
    1. グループチャットの現在の状態を調べる
    2. 次に話すべき人を決定する
    3. 参加者名を返すか、会話を終了するためにNoneを返す

    Args:
        state: 不変のスナップショットで以下を含む：
            - task: ChatMessage - 元のユーザータスク
            - participants: dict[str, str] - 参加者名 → 説明
            - conversation: tuple[ChatMessage, ...] - 会話の全履歴
            - history: tuple[GroupChatTurn, ...] - 発言者帰属付きのターンごとの履歴
            - round_index: int - これまでの選択ラウンド数
            - pending_agent: str | None - 現在アクティブなAgent（あれば）

    Returns:
        次の話者の名前、または会話を終了するためのNone

    """
    round_idx = state["round_index"]
    history = state["history"]

    # 4ターン後に終了（研究者 → ライター → 研究者 → ライター）
    if round_idx >= 4:
        return None

    # 履歴から最後の話者を取得します。
    last_speaker = history[-1].speaker if history else None

    # シンプルな交互：研究者 → ライター → 研究者 → ライター
    if last_speaker == "Researcher":
        return "Writer"
    return "Researcher"


async def main() -> None:
    researcher = ChatAgent(
        name="Researcher",
        description="Collects relevant background information.",
        instructions="Gather concise facts that help answer the question. Be brief.",
        chat_client=OpenAIChatClient(model_id="gpt-4o-mini"),
    )

    writer = ChatAgent(
        name="Writer",
        description="Synthesizes a polished answer using the gathered notes.",
        instructions="Compose a clear, structured answer using any notes provided.",
        chat_client=OpenAIChatClient(model_id="gpt-4o-mini"),
    )

    # 参加者を指定する2つの方法： 1. リスト形式 - agent.name属性を使用：.participants([researcher, writer]) 2.
    # 辞書形式 - 明示的な名前指定：.participants(researcher=researcher, writer=writer)
    workflow = (
        GroupChatBuilder()
        .select_speakers(select_next_speaker, display_name="Orchestrator")
        .participants([researcher, writer])  # Uses agent.name for participant names
        .build()
    )

    task = "What are the key benefits of using async/await in Python?"

    print("\nStarting Group Chat with Simple Speaker Selector...\n")
    print(f"TASK: {task}\n")
    print("=" * 80)

    async for event in workflow.run_stream(task):
        if isinstance(event, WorkflowOutputEvent):
            final_message = event.data
            author = getattr(final_message, "author_name", "Unknown")
            text = getattr(final_message, "text", str(final_message))
            print(f"\n[{author}]\n{text}\n")
            print("-" * 80)

    print("\nWorkflow completed.")


if __name__ == "__main__":
    asyncio.run(main())
