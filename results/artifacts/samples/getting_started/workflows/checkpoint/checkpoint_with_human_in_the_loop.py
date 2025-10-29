# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    ChatMessage,
    Executor,
    FileCheckpointStorage,
    RequestInfoEvent,
    RequestInfoExecutor,
    RequestInfoMessage,
    RequestResponse,
    Role,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,
    WorkflowRunState,
    WorkflowStatusEvent,
    get_checkpoint_summary,
    handler,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

# 注意: 上記のAzureクライアントのImportは実際の依存関係です。Azure対応環境外でこのサンプルを実行する場合は、
# `agent_framework.builtin`のチャットクライアントに切り替えるか、writer executorをモックすることを検討してください。
# ここではエンドツーエンドの構成を示すために具体的なImportを残しています。

if TYPE_CHECKING:
    from agent_framework import Workflow
    from agent_framework._workflows._checkpoint import WorkflowCheckpoint

"""
Sample: Checkpoint + human-in-the-loop quickstart.

This getting-started sample keeps the moving pieces to a minimum:

1. A brief is turned into a consistent prompt for an AI copywriter.
2. The copywriter (an `AgentExecutor`) drafts release notes.
3. A reviewer gateway routes every draft through `RequestInfoExecutor` so a human
   can approve or request tweaks.
4. The workflow records checkpoints between each superstep so you can stop the
   program, restart later, and optionally pre-supply human answers on resume.

Key concepts demonstrated
-------------------------
- Minimal executor pipeline with checkpoint persistence.
- Human-in-the-loop pause/resume by pairing `RequestInfoExecutor` with
  checkpoint restoration.
- Supplying responses at restore time (`run_stream_from_checkpoint(..., responses=...)`).

Typical pause/resume flow
-------------------------
1. Run the workflow until a human approval request is emitted.
2. If the human is offline, exit the program. A checkpoint with
   ``status=awaiting human response`` now exists.
3. Later, restart the script, select that checkpoint, and provide the stored
   human decision when prompted to pre-supply responses.
   Doing so applies the answer immediately on resume, so the system does **not**
   re-emit the same `RequestInfoEvent`.
"""

# サンプルの一時的なチェックポイントファイル用ディレクトリ。デモの成果物を分離し、繰り返し実行時に他のサンプルと衝突しないようにし、スクリプト終了時のクリーンアップでディレクトリを削除できるようにします。
TEMP_DIR = Path(__file__).with_suffix("").parent / "tmp" / "checkpoints_hitl"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class BriefPreparer(Executor):
    """ユーザーの概要を正規化し、単一のAgentExecutorRequestを送信します。"""

    # ワークフローの最初のExecutor。小さく保つことで、後でチェックポイントにキャプチャされるStateを理解しやすくします。
    # 人間が提供した概要を整理し、決定論的なプロンプト構造でAgentの実行を開始する役割を担います。

    def __init__(self, id: str, agent_id: str) -> None:
        super().__init__(id=id)
        self._agent_id = agent_id

    @handler
    async def prepare(self, brief: str, ctx: WorkflowContext[AgentExecutorRequest, str]) -> None:
        # 余分な空白を折りたたんで、プロンプトが実行間で安定するようにします。
        normalized = " ".join(brief.split()).strip()
        if not normalized.endswith("."):
            normalized += "."
        # 整理された概要を共有Stateに永続化し、下流のExecutorや将来のチェックポイントが元の意図を復元できるようにします。
        await ctx.set_shared_state("brief", normalized)
        prompt = (
            "You are drafting product release notes. Summarise the brief below in two sentences. "
            "Keep it positive and end with a call to action.\n\n"
            f"BRIEF: {normalized}"
        )
        # プロンプトをwriter
        # Agentに渡します。常にワークフローContextを経由してルーティングし、ランタイムがメッセージをチェックポイント用にキャプチャできるようにします。
        await ctx.send_message(
            AgentExecutorRequest(messages=[ChatMessage(Role.USER, text=prompt)], should_respond=True),
            target_id=self._agent_id,
        )


@dataclass
class HumanApprovalRequest(RequestInfoMessage):
    """RequestInfoExecutorを通じて人間のレビュアーに送信されるメッセージ。"""

    # これらのフィールドは意図的にシンプルにしています。チェックポイントにシリアライズされるためです。
    # プリミティブ型を保つことで、新しい`pending_requests_from_checkpoint`ヘルパーが再開時にそれらを再構築できることを保証します。
    prompt: str = ""
    draft: str = ""
    iteration: int = 0


class ReviewGateway(Executor):
    """Agentのドラフトを人間にルーティングし、必要に応じて修正のために戻す。"""

    def __init__(self, id: str, reviewer_id: str, writer_id: str, finalize_id: str) -> None:
        super().__init__(id=id)
        self._reviewer_id = reviewer_id
        self._writer_id = writer_id
        self._finalize_id = finalize_id

    @handler
    async def on_agent_response(
        self,
        response: AgentExecutorResponse,
        ctx: WorkflowContext[HumanApprovalRequest, str],
    ) -> None:
        # Agentの出力をキャプチャし、レビュアーに提示し、反復を永続化します。`RequestInfoExecutor`はチェックポイント復元時にこのStateを再構築します。
        draft = response.agent_run_response.text or ""
        iteration = int((await ctx.get_executor_state() or {}).get("iteration", 0)) + 1
        await ctx.set_executor_state({"iteration": iteration, "last_draft": draft})
        # 人間の承認リクエストを発行します。これがRequestInfoExecutorを通るため、回答が対話的にまたは事前提供されたレスポンスで供給されるまでワークフローは一時停止します。
        await ctx.send_message(
            HumanApprovalRequest(
                prompt="Review the draft. Reply 'approve' or provide edit instructions.",
                draft=draft,
                iteration=iteration,
            ),
            target_id=self._reviewer_id,
        )

    @handler
    async def on_human_feedback(
        self,
        feedback: RequestResponse[HumanApprovalRequest, str],
        ctx: WorkflowContext[AgentExecutorRequest | str, str],
    ) -> None:
        # RequestResponseラッパーは、人間のデータと元のRequestメッセージの両方を提供します。チェックポイントからの再開時でも同様です。
        reply = (feedback.data or "").strip()
        state = await ctx.get_executor_state() or {}
        draft = state.get("last_draft") or (feedback.original_request.draft if feedback.original_request else "")

        if reply.lower() == "approve":
            # 人間が承認すると、ワークフローをショートサーキットし、承認済みドラフトを最終Executorに送信できます。
            await ctx.send_message(draft, target_id=self._finalize_id)
            return

        # その他のレスポンスは新たな指示を伴ってwriterに戻します。
        guidance = reply or "Tighten the copy and emphasise customer benefit."
        iteration = int(state.get("iteration", 1)) + 1
        await ctx.set_executor_state({"iteration": iteration, "last_draft": draft})
        prompt = (
            "Revise the launch note. Respond with the new copy only.\n\n"
            f"Previous draft:\n{draft}\n\n"
            f"Human guidance: {guidance}"
        )
        await ctx.send_message(
            AgentExecutorRequest(messages=[ChatMessage(Role.USER, text=prompt)], should_respond=True),
            target_id=self._writer_id,
        )


class FinaliseExecutor(Executor):
    """承認済みテキストを公開します。"""

    @handler
    async def publish(self, text: str, ctx: WorkflowContext[Any, str]) -> None:
        # 出力を保存し、診断やUIが最終コピーを取得できるようにします。
        await ctx.set_executor_state({"published_text": text})
        # 最終出力をyieldし、ワークフローを正常に完了させます。
        await ctx.yield_output(text)


def create_workflow(*, checkpoint_storage: FileCheckpointStorage | None = None) -> "Workflow":
    """初回実行と再開の両方で使用されるワークフローグラフを組み立てます。"""

    # Azureクライアントは一度作成され、AgentExecutorがホストされたモデルに呼び出しを発行できます。Agent
    # IDは実行間で安定しており、チェックポイントの決定論性を保ちます。
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    writer = AgentExecutor(
        chat_client.create_agent(
            instructions="Write concise, warm release notes that sound human and helpful.",
        ),
        id="writer",
    )
    # RequestInfoExecutorはhuman-in-the-loopの要です。すべてのドラフトがここを通るため、チェックポイントは回答待ちで一時停止できます。
    review = RequestInfoExecutor(id="request_info")
    finalise = FinaliseExecutor(id="finalise")
    gateway = ReviewGateway(
        id="review_gateway",
        reviewer_id=review.id,
        writer_id=writer.id,
        finalize_id=finalise.id,
    )
    prepare = BriefPreparer(id="prepare_brief", agent_id=writer.id)

    # ワークフローDAGを配線します。エッジはモジュールドキュメント文字列で説明された番号付きステップを反映しています。
    # `WorkflowBuilder`は宣言的なので、これらのエッジを読むことが実行順序を理解する最速の方法です。
    builder = (
        WorkflowBuilder(max_iterations=6)
        .set_start_executor(prepare)
        .add_edge(prepare, writer)
        .add_edge(writer, gateway)
        .add_edge(gateway, review)
        .add_edge(review, gateway)  # human resumes loop
        .add_edge(gateway, writer)  # revisions
        .add_edge(gateway, finalise)
    )
    # 呼び出し元がストレージを提供した場合に永続化をオプトインします。チェックポイントの有無にかかわらず、ワークフローオブジェクト自体は同一です。
    if checkpoint_storage:
        builder = builder.with_checkpointing(checkpoint_storage=checkpoint_storage)
    return builder.build()


def _render_checkpoint_summary(checkpoints: list["WorkflowCheckpoint"]) -> None:
    """新しいフレームワークのサマリーで保存されたチェックポイントを整形表示します。"""

    print("\nCheckpoint summary:")
    for summary in [get_checkpoint_summary(cp) for cp in sorted(checkpoints, key=lambda c: c.timestamp)]:
        # チェックポイントごとに1行で構成し、ユーザーが出力をスキャンして未処理の人間作業がある再開ポイントを選べるようにします。
        line = (
            f"- {summary.checkpoint_id} | iter={summary.iteration_count} "
            f"| targets={summary.targets} | states={summary.executor_ids}"
        )
        if summary.status:
            line += f" | status={summary.status}"
        if summary.draft_preview:
            line += f" | draft_preview={summary.draft_preview}"
        if summary.pending_requests:
            line += f" | pending_request_id={summary.pending_requests[0].request_id}"
        print(line)


def _print_events(events: list[Any]) -> tuple[str | None, list[tuple[str, HumanApprovalRequest]]]:
    """ワークフローイベントをコンソールにエコーし、未処理のリクエストを収集します。"""

    completed_output: str | None = None
    requests: list[tuple[str, HumanApprovalRequest]] = []

    for event in events:
        print(f"Event: {event}")
        if isinstance(event, WorkflowOutputEvent):
            completed_output = event.data
        if isinstance(event, RequestInfoEvent) and isinstance(event.data, HumanApprovalRequest):
            # 保留中の人間承認をキャプチャし、呼び出し元が現在のイベントバッチ処理後にユーザーに入力を求められるようにします。
            requests.append((event.request_id, event.data))
        elif isinstance(event, WorkflowStatusEvent) and event.state in {
            WorkflowRunState.IN_PROGRESS_PENDING_REQUESTS,
            WorkflowRunState.IDLE_WITH_PENDING_REQUESTS,
        }:
            print(f"Workflow state: {event.state.name}")

    return completed_output, requests


def _prompt_for_responses(requests: list[tuple[str, HumanApprovalRequest]]) -> dict[str, str] | None:
    """ライブのRequestInfoリクエストに対する対話型CLIプロンプト。"""

    if not requests:
        return None
    answers: dict[str, str] = {}
    for request_id, request in requests:
        # プロンプトを会話形式に保ち、テスターがワークフローAPIを覚えなくてもスクリプトを使えるようにします。
        print("\n=== Human approval needed ===")
        print(f"request_id: {request_id}")
        if request.iteration:
            print(f"Iteration: {request.iteration}")
        print(request.prompt)
        print("Draft: \n---\n" + request.draft + "\n---")
        answer = input("Type 'approve' or enter revision guidance (or 'exit' to quit): ").strip()  # noqa: ASYNC250
        if answer.lower() == "exit":
            raise SystemExit("Stopped by user.")
        answers[request_id] = answer
    return answers


def _maybe_pre_supply_responses(cp: "WorkflowCheckpoint") -> dict[str, str] | None:
    """チェックポイント再開前にレスポンス収集を提案します。"""

    pending = get_checkpoint_summary(cp).pending_requests
    if not pending:
        return None

    print(
        "This checkpoint still has pending human input. Provide the responses now so the resume step "
        "applies them immediately and does not re-emit the original RequestInfo event."
    )
    choice = input("Pre-supply responses for this checkpoint? [y/N]: ").strip().lower()  # noqa: ASYNC250
    if choice not in {"y", "yes"}:
        return None

    answers: dict[str, str] = {}
    for item in pending:
        iteration = item.iteration or 0
        print(f"\nPending draft (iteration {iteration} | request_id={item.request_id}):")
        draft_text = (item.draft or "").strip()
        if draft_text:
            # サマリーの短縮プレビューはテキストを切り詰める可能性があります。ここではレビュアーが十分な判断を下せるように完全なドラフトを表示します。
            print("Draft:\n---\n" + draft_text + "\n---")
        else:
            print("Draft: [not captured in checkpoint payload - refer to your notes/log]")
        prompt_text = (item.prompt or "Review the draft").strip()
        print(prompt_text)
        answer = input("Response ('approve' or guidance, 'exit' to abort): ").strip()  # noqa: ASYNC250
        if answer.lower() == "exit":
            raise SystemExit("Resume aborted by user.")
        answers[item.request_id] = answer
    return answers


async def _consume(stream: AsyncIterable[Any]) -> list[Any]:
    """非同期イベントストリームをリストに具現化します。"""

    return [event async for event in stream]


async def run_interactive_session(workflow: "Workflow", initial_message: str) -> str | None:
    """ワークフローを完了または人間入力待ちで一時停止するまで実行します。"""

    pending_responses: dict[str, str] | None = None
    completed_output: str | None = None
    first = True

    while completed_output is None:
        if first:
            # 初期の概要でワークフローを開始します。Agentがドラフトを生成するとRequestInfoイベントが含まれます。
            events = await _consume(workflow.run_stream(initial_message))
            first = False
        elif pending_responses:
            # ユーザーが入力した回答をワークフローにフィードバックします。
            events = await _consume(workflow.send_responses_streaming(pending_responses))
        else:
            break

        completed_output, requests = _print_events(events)
        if completed_output is None:
            pending_responses = _prompt_for_responses(requests)

    return completed_output


async def resume_from_checkpoint(
    workflow: "Workflow",
    checkpoint_id: str,
    storage: FileCheckpointStorage,
    pre_supplied: dict[str, str] | None,
) -> None:
    """保存されたチェックポイントを再開し、完了または次の一時停止まで続行します。"""

    print(f"\nResuming from checkpoint: {checkpoint_id}")
    events = await _consume(
        workflow.run_stream_from_checkpoint(
            checkpoint_id,
            checkpoint_storage=storage,
            responses=pre_supplied,
        )
    )
    completed_output, requests = _print_events(events)
    if pre_supplied and not requests and completed_output is None:
        # チェックポイントが提供された回答のみを必要とした場合、ワークフローが次のスーパーステップ（通常は別のAgentレスポンス）を待っていることをユーザーに知らせます。
        print("Pre-supplied responses applied automatically; workflow is now waiting for the next step.")

    pending = _prompt_for_responses(requests)
    while completed_output is None and pending:
        events = await _consume(workflow.send_responses_streaming(pending))
        completed_output, requests = _print_events(events)
        if completed_output is None:
            pending = _prompt_for_responses(requests)
        else:
            break

    if completed_output:
        print(f"Workflow completed with: {completed_output}")


async def main() -> None:
    """初回実行と再開の両方で使用されるエントリーポイント。"""

    for file in TEMP_DIR.glob("*.json"):
        # 各実行をクリーンスレートで開始し、ディレクトリに古いチェックポイントがあってもデモが決定論的になるようにします。
        file.unlink()

    storage = FileCheckpointStorage(storage_path=TEMP_DIR)
    workflow = create_workflow(checkpoint_storage=storage)

    brief = (
        "Introduce our limited edition smart coffee grinder. Mention the $249 price, highlight the "
        "sensor that auto-adjusts the grind, and invite customers to pre-order on the website."
    )

    print("Running workflow (human approval required)...")
    completed = await run_interactive_session(workflow, initial_message=brief)
    if completed:
        print(f"Initial run completed with final copy: {completed}")
    else:
        print("Initial run paused for human input.")

    checkpoints = await storage.list_checkpoints()
    if not checkpoints:
        print("No checkpoints recorded.")
        return

    # ユーザーにインデックスを入力する前に利用可能なものを表示します。 summary helper はこの出力を他のツールと一貫させます。
    _render_checkpoint_summary(checkpoints)

    sorted_cps = sorted(checkpoints, key=lambda c: c.timestamp)
    print("\nAvailable checkpoints:")
    for idx, cp in enumerate(sorted_cps):
        print(f"  [{idx}] id={cp.checkpoint_id} iter={cp.iteration_count}")

    # pause/resume デモでは通常、summary status が "awaiting human response"
    # と表示される最新のチェックポイントを選びます。 これは、ワークフローが再構築され、保留中の回答を収集し、休止後に続行できることを証明する保存された状態です。
    selection = input("\nResume from which checkpoint? (press Enter to skip): ").strip()  # noqa: ASYNC250
    if not selection:
        print("No resume selected. Exiting.")
        return

    try:
        idx = int(selection)
    except ValueError:
        print("Invalid input; exiting.")
        return

    if not 0 <= idx < len(sorted_cps):
        print("Index out of range; exiting.")
        return

    chosen = sorted_cps[idx]
    summary = get_checkpoint_summary(chosen)
    if summary.status == "completed":
        print("Selected checkpoint already reflects a completed workflow; nothing to resume.")
        return

    # ユーザーが望む場合は、resume 呼び出しがワークフローにプッシュして再プロンプトを避けられるように、今すぐ彼らの決定をキャプチャします。
    pre_responses = _maybe_pre_supply_responses(chosen)

    resumed_workflow = create_workflow()
    # 新しいワークフローインスタンスで再開します。チェックポイントは永続的な状態を保持し、このオブジェクトはランタイムの配線を保持します。
    await resume_from_checkpoint(resumed_workflow, chosen.checkpoint_id, storage, pre_responses)


if __name__ == "__main__":
    asyncio.run(main())
