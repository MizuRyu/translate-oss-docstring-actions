# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    ChatMessage,
    Executor,
    FileCheckpointStorage,
    Role,
    WorkflowBuilder,
    WorkflowContext,
    get_checkpoint_summary,
    handler,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

if TYPE_CHECKING:
    from agent_framework import Workflow
    from agent_framework._workflows._checkpoint import WorkflowCheckpoint

"""
Sample: Checkpointing and Resuming a Workflow (with an Agent stage)

Purpose:
This sample shows how to enable checkpointing at superstep boundaries, persist both
executor-local state and shared workflow state, and then resume execution from a specific
checkpoint. The workflow demonstrates a simple text-processing pipeline that includes
an LLM-backed AgentExecutor stage.

Pipeline:
1) UpperCaseExecutor converts input to uppercase and records state.
2) ReverseTextExecutor reverses the string.
3) SubmitToLowerAgent prepares an AgentExecutorRequest for the lowercasing agent.
4) lower_agent (AgentExecutor) converts text to lowercase via Azure OpenAI.
5) FinalizeFromAgent yields the final result.

What you learn:
- How to persist executor state using ctx.get_executor_state and ctx.set_executor_state.
- How to persist shared workflow state using ctx.set_shared_state for cross-executor visibility.
- How to configure FileCheckpointStorage and call with_checkpointing on WorkflowBuilder.
- How to list and inspect checkpoints programmatically.
- How to interactively choose a checkpoint to resume from (instead of always resuming
    from the most recent or a hard-coded one) using run_stream_from_checkpoint.
- How workflows complete by yielding outputs when idle, not via explicit completion events.

Prerequisites:
- Azure AI or Azure OpenAI available for AzureOpenAIChatClient.
- Authentication with azure-identity via AzureCliCredential. Run az login locally.
- Filesystem access for writing JSON checkpoint files in a temp directory.
"""

# チェックポイントを保存するための一時ディレクトリを定義します。 これらのファイルにより、ワークフローを後で再開できます。
DIR = os.path.dirname(__file__)
TEMP_DIR = os.path.join(DIR, "tmp", "checkpoints")
os.makedirs(TEMP_DIR, exist_ok=True)


class UpperCaseExecutor(Executor):
    """入力テキストを大文字に変換し、ローカルおよび共有状態の両方を永続化します。"""

    @handler
    async def to_upper_case(self, text: str, ctx: WorkflowContext[str]) -> None:
        result = text.upper()
        print(f"UpperCaseExecutor: '{text}' -> '{result}'")

        # executor ローカルの状態を永続化し、チェックポイントにキャプチャして再開後に観測性やロジックで利用可能にします。
        prev = await ctx.get_executor_state() or {}
        count = int(prev.get("count", 0)) + 1
        await ctx.set_executor_state({
            "count": count,
            "last_input": text,
            "last_output": result,
        })

        # shared_state に書き込み、下流の executor や再開された実行が読み取れるようにします。
        await ctx.set_shared_state("original_input", text)
        await ctx.set_shared_state("upper_output", result)

        # 変換されたテキストを次の executor に送信します。
        await ctx.send_message(result)


class SubmitToLowerAgent(Executor):
    """shared-state の可視性を保ちながら、lowercasing agent に送信する AgentExecutorRequest を構築します。"""

    def __init__(self, id: str, agent_id: str):
        super().__init__(id=id)
        self._agent_id = agent_id

    @handler
    async def submit(self, text: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
        # UpperCaseExecutor によって書き込まれた shared_state を読み取ることを示します。
        # 共有状態はチェックポイントを越えて存続し、すべての executor に見えます。
        orig = await ctx.get_shared_state("original_input")
        upper = await ctx.get_shared_state("upper_output")
        print(f"LowerAgent (shared_state): original_input='{orig}', upper_output='{upper}'")

        # AgentExecutor のための最小限で決定論的な prompt を構築します。
        prompt = f"Convert the following text to lowercase. Return ONLY the transformed text.\n\nText: {text}"

        # AgentExecutor に送信します。should_respond=True は agent に返信を生成するよう指示します。
        await ctx.send_message(
            AgentExecutorRequest(messages=[ChatMessage(Role.USER, text=prompt)], should_respond=True),
            target_id=self._agent_id,
        )


class FinalizeFromAgent(Executor):
    """AgentExecutorResponse を消費し、最終結果を yield します。"""

    @handler
    async def finalize(self, response: AgentExecutorResponse, ctx: WorkflowContext[Any, str]) -> None:
        result = response.agent_run_response.text or ""

        # チェックポイントを検査するときの監査可能性のために executor ローカルの状態を永続化します。
        prev = await ctx.get_executor_state() or {}
        count = int(prev.get("count", 0)) + 1
        await ctx.set_executor_state({
            "count": count,
            "last_output": result,
            "final": True,
        })

        # 最終結果を yield して外部の消費者が最終値を見られるようにします。
        await ctx.yield_output(result)


class ReverseTextExecutor(Executor):
    """入力テキストを逆順にし、ローカル状態を永続化します。"""

    @handler
    async def reverse_text(self, text: str, ctx: WorkflowContext[str]) -> None:
        result = text[::-1]
        print(f"ReverseTextExecutor: '{text}' -> '{result}'")

        # チェックポイント検査で進捗を明らかにできるように executor ローカルの状態を永続化します。
        prev = await ctx.get_executor_state() or {}
        count = int(prev.get("count", 0)) + 1
        await ctx.set_executor_state({
            "count": count,
            "last_input": text,
            "last_output": result,
        })

        # 逆順にした文字列を次のステージに転送します。
        await ctx.send_message(result)


def create_workflow(checkpoint_storage: FileCheckpointStorage) -> "Workflow":
    # パイプライン executor をインスタンス化します。
    upper_case_executor = UpperCaseExecutor(id="upper-case")
    reverse_text_executor = ReverseTextExecutor(id="reverse-text")

    # テキストを小文字にする agent ステージを設定します。
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    lower_agent = AgentExecutor(
        chat_client.create_agent(
            instructions=("You transform text to lowercase. Reply with ONLY the transformed text.")
        ),
        id="lower_agent",
    )

    # agent と terminalization ステージへのブリッジです。
    submit_lower = SubmitToLowerAgent(id="submit_lower", agent_id=lower_agent.id)
    finalize = FinalizeFromAgent(id="finalize")

    # チェックポイント有効化でワークフローを構築します。
    return (
        WorkflowBuilder(max_iterations=5)
        .add_edge(upper_case_executor, reverse_text_executor)  # Uppercase -> Reverse
        .add_edge(reverse_text_executor, submit_lower)  # Reverse -> Build Agent request
        .add_edge(submit_lower, lower_agent)  # Submit to AgentExecutor
        .add_edge(lower_agent, finalize)  # Agent output -> Finalize
        .set_start_executor(upper_case_executor)  # Entry point
        .with_checkpointing(checkpoint_storage=checkpoint_storage)  # Enable persistence
        .build()
    )


def _render_checkpoint_summary(checkpoints: list["WorkflowCheckpoint"]) -> None:
    """フレームワークの summary を使って人間に優しいチェックポイントのメタデータを表示します。"""

    if not checkpoints:
        return

    print("\nCheckpoint summary:")
    for cp in sorted(checkpoints, key=lambda c: c.timestamp):
        summary = get_checkpoint_summary(cp)
        msg_count = sum(len(v) for v in cp.messages.values())
        state_keys = sorted(summary.executor_ids)
        orig = cp.shared_state.get("original_input")
        upper = cp.shared_state.get("upper_output")

        line = (
            f"- {summary.checkpoint_id} | iter={summary.iteration_count} | messages={msg_count} | states={state_keys}"
        )
        if summary.status:
            line += f" | status={summary.status}"
        line += f" | shared_state: original_input='{orig}', upper_output='{upper}'"
        print(line)


async def main():
    # このサンプルディレクトリの既存のチェックポイントをクリアしてクリーンな実行にします。
    checkpoint_dir = Path(TEMP_DIR)
    for file in checkpoint_dir.glob("*.json"):  # noqa: ASYNC240
        file.unlink()

    # with_checkpointing によって書き込まれたチェックポイントのバックアップストアです。
    checkpoint_storage = FileCheckpointStorage(storage_path=TEMP_DIR)

    workflow = create_workflow(checkpoint_storage=checkpoint_storage)

    # フルワークフローを一度実行し、イベントがストリームされる様子を観察します。
    print("Running workflow with initial message...")
    async for event in workflow.run_stream(message="hello world"):
        print(f"Event: {event}")

    # 実行中に書き込まれたチェックポイントを検査します。
    all_checkpoints = await checkpoint_storage.list_checkpoints()
    if not all_checkpoints:
        print("No checkpoints found!")
        return

    # この実行で作成されたすべてのチェックポイントは同じ workflow_id を共有します。
    workflow_id = all_checkpoints[0].workflow_id

    _render_checkpoint_summary(all_checkpoints)

    # 再開するチェックポイントを対話的に選択できるようにします。
    sorted_cps = sorted([cp for cp in all_checkpoints if cp.workflow_id == workflow_id], key=lambda c: c.timestamp)

    print("\nAvailable checkpoints to resume from:")
    for idx, cp in enumerate(sorted_cps):
        summary = get_checkpoint_summary(cp)
        line = f"  [{idx}] id={summary.checkpoint_id} iter={summary.iteration_count}"
        if summary.status:
            line += f" status={summary.status}"
        msg_count = sum(len(v) for v in cp.messages.values())
        line += f" messages={msg_count}"
        print(line)

    user_input = input(  # noqa: ASYNC250
        "\nEnter checkpoint index (or paste checkpoint id) to resume from, or press Enter to skip resume: "
    ).strip()

    if not user_input:
        print("No checkpoint selected. Exiting without resuming.")
        return

    chosen_cp_id: str | None = None

    # まずは index として試みます。
    if user_input.isdigit():
        idx = int(user_input)
        if 0 <= idx < len(sorted_cps):
            chosen_cp_id = sorted_cps[idx].checkpoint_id
    # 直接 id マッチにフォールバックします。
    if chosen_cp_id is None:
        for cp in sorted_cps:
            if cp.checkpoint_id.startswith(user_input):  # allow prefix match for convenience
                chosen_cp_id = cp.checkpoint_id
                break

    if chosen_cp_id is None:
        print("Input did not match any checkpoint. Exiting without resuming.")
        return

    # 同じ workflow graph 定義を再利用し、以前のチェックポイントから再開できます。
    # この2番目のワークフローインスタンスはチェックポイントを有効にせず、再開が保存された状態を読み取るが新しいチェックポイントを書き込む必要がないことを示します。
    new_workflow = create_workflow(checkpoint_storage=checkpoint_storage)

    print(f"\nResuming from checkpoint: {chosen_cp_id}")
    async for event in new_workflow.run_stream_from_checkpoint(chosen_cp_id, checkpoint_storage=checkpoint_storage):
        print(f"Resumed Event: {event}")

    """
    Sample Output:

    Running workflow with initial message...
    UpperCaseExecutor: 'hello world' -> 'HELLO WORLD'
    Event: ExecutorInvokeEvent(executor_id=upper_case_executor)
    Event: ExecutorCompletedEvent(executor_id=upper_case_executor)
    ReverseTextExecutor: 'HELLO WORLD' -> 'DLROW OLLEH'
    Event: ExecutorInvokeEvent(executor_id=reverse_text_executor)
    Event: ExecutorCompletedEvent(executor_id=reverse_text_executor)
    LowerAgent (shared_state): original_input='hello world', upper_output='HELLO WORLD'
    Event: ExecutorInvokeEvent(executor_id=submit_lower)
    Event: ExecutorInvokeEvent(executor_id=lower_agent)
    Event: ExecutorInvokeEvent(executor_id=finalize)

    Checkpoint summary:
    - dfc63e72-8e8d-454f-9b6d-0d740b9062e6 | label='after_initial_execution' | iter=0 | messages=1 | states=['upper_case_executor'] | shared_state: original_input='hello world', upper_output='HELLO WORLD'
    - a78c345a-e5d9-45ba-82c0-cb725452d91b | label='superstep_1' | iter=1 | messages=1 | states=['reverse_text_executor', 'upper_case_executor'] | shared_state: original_input='hello world', upper_output='HELLO WORLD'
    - 637c1dbd-a525-4404-9583-da03980537a2 | label='superstep_2' | iter=2 | messages=0 | states=['finalize', 'lower_agent', 'reverse_text_executor', 'submit_lower', 'upper_case_executor'] | shared_state: original_input='hello world', upper_output='HELLO WORLD'

    Available checkpoints to resume from:
        [0] id=dfc63e72-... iter=0 messages=1 label='after_initial_execution'
        [1] id=a78c345a-... iter=1 messages=1 label='superstep_1'
        [2] id=637c1dbd-... iter=2 messages=0 label='superstep_2'

    Enter checkpoint index (or paste checkpoint id) to resume from, or press Enter to skip resume: 1

    Resuming from checkpoint: a78c345a-e5d9-45ba-82c0-cb725452d91b
    LowerAgent (shared_state): original_input='hello world', upper_output='HELLO WORLD'
    Resumed Event: ExecutorInvokeEvent(executor_id=submit_lower)
    Resumed Event: ExecutorInvokeEvent(executor_id=lower_agent)
    Resumed Event: ExecutorInvokeEvent(executor_id=finalize)
    """  # noqa: E501


if __name__ == "__main__":
    asyncio.run(main())
