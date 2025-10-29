# Copyright (c) Microsoft. All rights reserved.

import asyncio
from dataclasses import dataclass

from agent_framework import (
    AgentExecutor,  # Executor that runs the agent
    AgentExecutorRequest,  # Message bundle sent to an AgentExecutor
    AgentExecutorResponse,  # Result returned by an AgentExecutor
    ChatMessage,  # Chat message structure
    Executor,  # Base class for workflow executors
    RequestInfoEvent,  # Event emitted when human input is requested
    RequestInfoExecutor,  # Special executor that collects human input out of band
    RequestInfoMessage,  # Base class for request payloads sent to RequestInfoExecutor
    RequestResponse,  # Correlates a human response with the original request
    Role,  # Enum of chat roles (user, assistant, system)
    WorkflowBuilder,  # Fluent builder for assembling the graph
    WorkflowContext,  # Per run context and event bus
    WorkflowOutputEvent,  # Event emitted when workflow yields output
    WorkflowRunState,  # Enum of workflow run states
    WorkflowStatusEvent,  # Event emitted on run state changes
    handler,  # Decorator to expose an Executor method as a step
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from pydantic import BaseModel

"""
Sample: Human in the loop guessing game

An agent guesses a number, then a human guides it with higher, lower, or
correct via RequestInfoExecutor. The loop continues until the human confirms
correct, at which point the workflow completes when idle with no pending work.

Purpose:
Show how to integrate a human step in the middle of an LLM workflow using RequestInfoExecutor and correlated
RequestResponse objects.

Demonstrate:
- Alternating turns between an AgentExecutor and a human, driven by events.
- Using Pydantic response_format to enforce structured JSON output from the agent instead of regex parsing.
- Driving the loop in application code with run_stream and send_responses_streaming.

Prerequisites:
- Azure OpenAI configured for AzureOpenAIChatClient with required environment variables.
- Authentication via azure-identity. Use AzureCliCredential and run az login before executing the sample.
- Basic familiarity with WorkflowBuilder, executors, edges, events, and streaming runs.
"""

# RequestInfoExecutorの役割: RequestInfoExecutorはワークフローにネイティブなブリッジで、情報要求でグラフを一時停止し、
# 型付きペイロードを持つRequestInfoEventを発行し、
# アプリケーションが発行したrequest_idに対応するRequestResponseを受け取るまでグラフを再開しません。自身で入力を収集しません。
# アプリケーションはUIやCLIから人間の返信を収集し、request_idをキーとした辞書をsend_responses_streamingに渡す責任があります。
# このexecutorは一時停止と再開の人間によるゲーティングを標準化し、型付きリクエストペイロードを運び、相関を保持するために存在します。
# RequestInfoExecutorに送られるリクエストタイプは人間のフィードバック用です。 agentの最後の推測を含めることでUIやCLIがコンテキストを表示でき、
# ターンマネージャーが余計なState読み込みを避けるのに役立ちます。 RequestInfoMessageをサブクラス化する理由:
# RequestInfoMessageをサブクラス化すると、人間が見るリクエストの正確なスキーマを定義できます。
# これにより強力な型付け、将来互換性のある検証、明確な相関セマンティクスが得られます。 また、前回の推測などのコンテキストフィールドを添付できるため、
# UIが他の場所から余計なStateを取得せずにリッチなPromptをレンダリングできます。
@dataclass
class HumanFeedbackRequest(RequestInfoMessage):
    prompt: str = ""
    guess: int | None = None


class GuessOutput(BaseModel):
    """agentからの構造化出力。response_formatで強制され、信頼性の高い解析を実現します。"""

    guess: int


class TurnManager(Executor):
    """agentと人間のターンを調整します。

    Responsibilities:
    - 最初のagentターンを開始します。
    - 各agentの返信後にHumanFeedbackRequestで人間のフィードバックを要求します。
    - 各人間の返信後にゲームを終了するか、フィードバック付きでagentに再度プロンプトを送ります。

    """

    def __init__(self, id: str | None = None):
        super().__init__(id=id or "turn_manager")

    @handler
    async def start(self, _: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
        """ゲームを開始し、agentに初期推測を依頼します。

        Contract:
        - 入力は単純なスタータートークン（ここでは無視されます）。
        - 出力はAgentExecutorRequestで、agentに推測を生成させます。

        """
        user = ChatMessage(Role.USER, text="Start by making your first guess.")
        await ctx.send_message(AgentExecutorRequest(messages=[user], should_respond=True))

    @handler
    async def on_agent_response(
        self,
        result: AgentExecutorResponse,
        ctx: WorkflowContext[HumanFeedbackRequest],
    ) -> None:
        """agentの推測を処理し、人間の指示を要求します。

        Steps:
        1) agentのJSONをGuessOutputに解析して堅牢性を確保します。
        2) HumanFeedbackRequestをRequestInfoExecutorに送信し、明確な指示を与えます:
           - higherは人間の秘密の数がagentの推測より大きいことを意味します。
           - lowerは人間の秘密の数がagentの推測より小さいことを意味します。
           - correctは推測が正確であることを確認します。
           - exitはデモを終了します。

        """
        # 構造化モデル出力を解析します（agentが返信しなかった場合の防御的デフォルト）。
        text = result.agent_run_response.text or ""
        last_guess = GuessOutput.model_validate_json(text).guess if text else None

        # agentの推測に対してhigherとlowerを定義した正確な人間用プロンプトを作成します。
        prompt = (
            f"The agent guessed: {last_guess if last_guess is not None else text}. "
            "Type one of: higher (your number is higher than this guess), "
            "lower (your number is lower than this guess), correct, or exit."
        )
        await ctx.send_message(HumanFeedbackRequest(prompt=prompt, guess=last_guess))

    @handler
    async def on_human_feedback(
        self,
        feedback: RequestResponse[HumanFeedbackRequest, str],
        ctx: WorkflowContext[AgentExecutorRequest, str],
    ) -> None:
        """人間のフィードバックに基づいてゲームを続行または終了します。

        RequestResponseには人間の文字列返信と相関するHumanFeedbackRequestが含まれ、
        便宜上前回の推測を保持します。

        """
        reply = (feedback.data or "").strip().lower()
        # 余計な共有Stateの読み込みを避けるため、相関するrequestの推測を優先します。
        last_guess = getattr(feedback.original_request, "guess", None)

        if reply == "correct":
            await ctx.yield_output(f"Guessed correctly: {last_guess}")
            return

        # agentに再挑戦のフィードバックを提供します。 agentの出力は厳密にJSONのままにして、次のターンでの安定した解析を保証します。
        user_msg = ChatMessage(
            Role.USER,
            text=(f'Feedback: {reply}. Return ONLY a JSON object matching the schema {{"guess": <int 1..10>}}.'),
        )
        await ctx.send_message(AgentExecutorRequest(messages=[user_msg], should_respond=True))


async def main() -> None:
    # chat agentを作成し、AgentExecutorでラップします。
    # response_formatはモデルがGuessOutputと互換性のあるJSONを生成することを強制します。
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = chat_client.create_agent(
        instructions=(
            "You guess a number between 1 and 10. "
            "If the user says 'higher' or 'lower', adjust your next guess. "
            'You MUST return ONLY a JSON object exactly matching this schema: {"guess": <integer 1..10>}. '
            "No explanations or additional text."
        ),
        response_format=GuessOutput,
    )

    # シンプルなループを構築します: TurnManager <-> AgentExecutor <-> RequestInfoExecutor.
    # TurnManagerが調整し、AgentExecutorがモデルを実行し、RequestInfoExecutorが人間の返信を収集します。
    turn_manager = TurnManager(id="turn_manager")
    agent_exec = AgentExecutor(agent=agent, id="agent")

    # 命名に関する注意: この変数は歴史的な理由から現在hitlと名付けられています。この名前は曖昧または魔法的に感じられることがあります。
    # 明確にするために、自分のコードではrequest_info_executorに名前を変更することを検討してください。これはRequestInfoExecutorノードを直接表しており、人間の返信をバンド外で収集します。
    hitl = RequestInfoExecutor(id="request_info")

    top_builder = (
        WorkflowBuilder()
        .set_start_executor(turn_manager)
        .add_edge(turn_manager, agent_exec)  # Ask agent to make/adjust a guess
        .add_edge(agent_exec, turn_manager)  # Agent's response comes back to coordinator
        .add_edge(turn_manager, hitl)  # Ask human for guidance
        .add_edge(hitl, turn_manager)  # Feed human guidance back to coordinator
    )

    # ワークフローを構築します（この最小サンプルではチェックポイントはありません）。
    workflow = top_builder.build()

    # Human in the loopの実行：ワークフローの呼び出しと収集したレスポンスの提供を交互に行います。
    pending_responses: dict[str, str] | None = None
    completed = False
    workflow_output: str | None = None

    # ユーザーガイダンスの表示: ユーザーに事前に指示を出したい場合は、ループの前に短いバナーを表示してください。 例: print( "Interactive
    # mode. When prompted, type one of: higher, lower, correct, or exit. " "The agent
    # will keep guessing until you reply correct.", flush=True, )

    while not completed:
        # 最初のイテレーションはrun_stream("start")を使用します。
        # 以降のイテレーションはコンソールからのpending_responsesを使ってsend_responses_streamingを使用します。
        stream = (
            workflow.send_responses_streaming(pending_responses) if pending_responses else workflow.run_stream("start")
        )
        # このターンのイベントを収集します。これらの中には、ワークフローが人間の入力のために一時停止するときに状態IDLE_WITH_PENDING_REQUESTSを持つWorkflowStatusEventが含まれることがあります。
        # これはリクエストが発行される際にIN_PROGRESS_PENDING_REQUESTSに先行されます。
        events = [event async for event in stream]
        pending_responses = None

        # 人間のリクエスト、ワークフローの出力を収集し、完了をチェックします。
        requests: list[tuple[str, str]] = []  # (request_id, prompt)
        for event in events:
            if isinstance(event, RequestInfoEvent) and isinstance(event.data, HumanFeedbackRequest):
                # 私たちのHumanFeedbackRequestのRequestInfoEvent。
                requests.append((event.request_id, event.data.prompt))
            elif isinstance(event, WorkflowOutputEvent):
                # ワークフローの出力を生成されるたびにキャプチャします。
                workflow_output = str(event.data)
                completed = True  # このサンプルでは、1つの出力後に終了します。

        # より良い開発者体験のために実行状態の遷移を検出します。
        pending_status = any(
            isinstance(e, WorkflowStatusEvent) and e.state == WorkflowRunState.IN_PROGRESS_PENDING_REQUESTS
            for e in events
        )
        idle_with_requests = any(
            isinstance(e, WorkflowStatusEvent) and e.state == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS
            for e in events
        )
        if pending_status:
            print("State: IN_PROGRESS_PENDING_REQUESTS (requests outstanding)")
        if idle_with_requests:
            print("State: IDLE_WITH_PENDING_REQUESTS (awaiting human input)")

        # 人間のリクエストがあれば、ユーザーにプロンプトを表示し、レスポンスを準備します。
        if requests and not completed:
            responses: dict[str, str] = {}
            for req_id, prompt in requests:
                # サンプル用のシンプルなコンソールプロンプト。
                print(f"HITL> {prompt}")
                # 指示の印刷はすでに上に表示されています。以下の入力行がユーザーの入力ポイントです。
                # 必要に応じてここにさらにガイダンスを追加できますが、簡潔にしてください。
                answer = input("Enter higher/lower/correct/exit: ").lower()  # noqa: ASYNC250
                if answer == "exit":
                    print("Exiting...")
                    return
                responses[req_id] = answer
            pending_responses = responses

    # ストリーミング中にキャプチャされたワークフローの出力から最終結果を表示します。
    print(f"Workflow output: {workflow_output}")
    """
    Sample Output:

    HITL> The agent guessed: 5. Type one of: higher (your number is higher than this guess), lower (your number is lower than this guess), correct, or exit.
    Enter higher/lower/correct/exit: higher
    HITL> The agent guessed: 8. Type one of: higher (your number is higher than this guess), lower (your number is lower than this guess), correct, or exit.
    Enter higher/lower/correct/exit: higher
    HITL> The agent guessed: 10. Type one of: higher (your number is higher than this guess), lower (your number is lower than this guess), correct, or exit.
    Enter higher/lower/correct/exit: lower
    HITL> The agent guessed: 9. Type one of: higher (your number is higher than this guess), lower (your number is lower than this guess), correct, or exit.
    Enter higher/lower/correct/exit: correct
    Workflow output: Guessed correctly: 9
    """  # noqa: E501


if __name__ == "__main__":
    asyncio.run(main())
