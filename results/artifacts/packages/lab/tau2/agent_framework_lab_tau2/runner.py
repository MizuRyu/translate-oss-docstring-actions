# Copyright (c) Microsoft. All rights reserved.

import uuid
from typing import cast

from agent_framework._agents import ChatAgent
from agent_framework._types import AgentRunResponse, ChatMessage, Role
from agent_framework._workflows import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    FunctionExecutor,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
)
from agent_framework.openai import OpenAIChatClient
from loguru import logger
from tau2.data_model.simulation import SimulationRun, TerminationReason  # type: ignore[import-untyped]
from tau2.data_model.tasks import Task  # type: ignore[import-untyped]
from tau2.domains.airline.environment import get_environment  # type: ignore[import-untyped]
from tau2.evaluator.evaluator import EvaluationType, RewardInfo, evaluate_simulation  # type: ignore[import-untyped]
from tau2.user.user_simulator import (  # type: ignore[import-untyped]
    OUT_OF_SCOPE,
    STOP,
    TRANSFER,
    get_global_user_sim_guidelines,
)
from tau2.utils.utils import get_now  # type: ignore[import-untyped]

from ._message_utils import flip_messages, log_messages
from ._sliding_window import SlidingWindowChatMessageStore
from ._tau2_utils import convert_agent_framework_messages_to_tau2_messages, convert_tau2_tool_to_ai_function

__all__ = ["ASSISTANT_AGENT_ID", "ORCHESTRATOR_ID", "USER_SIMULATOR_ID", "TaskRunner"]

# tau2のLLMAgentに対応するAgentの指示
ASSISTANT_AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.
Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

# Agentのデフォルト最初のメッセージ（tau2に対応）
DEFAULT_FIRST_AGENT_MESSAGE = "Hi! How can I help you today?"

# Agent executor IDの定数
ASSISTANT_AGENT_ID = "assistant_agent"
USER_SIMULATOR_ID = "user_simulator"
ORCHESTRATOR_ID = "orchestrator"


class TaskRunner:
    """tau2ベンチマークのためにAgentフレームワークのワークフローを使ってタスク実行をオーケストレーションします。

    アシスタントAgentとユーザーシミュレータ間の会話フローを管理し、
    終了条件を処理し、tau2の指標でパフォーマンスを評価します。

    現時点では「airline」ドメインのみ対応しています。

    """

    # 状態の追跡
    step_count: int
    full_conversation: list[ChatMessage]
    termination_reason: TerminationReason | None
    full_reward_info: RewardInfo | None
    _final_user_message: list[ChatMessage] | None
    _assistant_executor: AgentExecutor | None
    _user_executor: AgentExecutor | None

    # 設定
    max_steps: int
    assistant_sampling_temperature: float
    assistant_window_size: int

    def __init__(self, max_steps: int, assistant_sampling_temperature: float = 0.0, assistant_window_size: int = 32768):
        """TaskRunnerを初期化します。

        Args:
            max_steps: 実行する最大ステップ数。
            assistant_sampling_temperature: アシスタントAgentのサンプリング温度。
            assistant_window_size: アシスタントAgentのウィンドウサイズ。

        """
        self.assistant_sampling_temperature = assistant_sampling_temperature
        self.assistant_window_size = assistant_window_size
        self.max_steps = max_steps
        self.reinit()

    def reinit(self) -> "TaskRunner":
        """新しいタスク実行のためにすべての状態をリセットする。"""
        self.step_count = 0
        self.full_conversation = []
        self.termination_reason = None
        self.full_reward_info = None
        self._final_user_message = None
        self._assistant_executor = None
        self._user_executor = None
        logger.info("TaskRunner has been re-initialized.")
        return self

    def __repr__(self) -> str:
        """TaskRunnerの文字列表現を返す。"""
        return (
            f"TaskRunner(max_steps={self.max_steps}, step_count={self.step_count}, "
            f"full_conversation_length={len(self.full_conversation)}, "
            f"termination_reason={self.termination_reason}, full_reward_info={self.full_reward_info})"
        )

    def should_not_stop(self, response: AgentExecutorResponse) -> bool:
        """レスポンスに基づいて会話を停止すべきかどうかをチェックする。"""
        # executor_idに基づいて送信者を判定する
        is_from_agent = response.executor_id == ASSISTANT_AGENT_ID
        is_from_user = response.executor_id == USER_SIMULATOR_ID

        self.step_count += 1

        logger.opt(colors=True).info(
            f"<bold>[Step {self.step_count}] Received the following response from "
            f"{'<blue>assistant</blue>' if is_from_agent else '<green>user</green>'}</bold>, "
            f"routing to {'<green>user</green>' if is_from_agent else '<blue>assistant</blue>'}:"
        )
        log_messages(response.agent_run_response.messages)

        if self.step_count >= self.max_steps:
            logger.info(f"Max steps ({self.max_steps}) reached - terminating conversation")
            self.termination_reason = TerminationReason.MAX_STEPS
            # ワークフローを終了する
            return False

        response_text = response.agent_run_response.text
        if is_from_agent and self._is_agent_stop(response_text):
            logger.info("Agent requested stop - terminating conversation")
            self.termination_reason = TerminationReason.AGENT_STOP
            return False

        if is_from_user and self._is_user_stop(response_text):
            logger.info(f"User requested stop with message: '{response_text}' - terminating conversation")
            self.termination_reason = TerminationReason.USER_STOP
            # 最終ユーザーメッセージはアシスタントのメッセージストアに現れません。 なぜならそこに到達しないためです。 評価に必要なので保存する必要があります。
            self._final_user_message = flip_messages(response.agent_run_response.messages)
            return False

        return True

    def _is_agent_stop(self, _: str) -> bool:
        """Agentが会話停止を望んでいるかチェックする。"""
        # Agentが特定の停止トークンを使う場合はそれをチェックできる
        return False  # このセットアップではAgentに明示的な停止はない

    def _is_user_stop(self, text: str) -> bool:
        """ユーザーが会話停止を望んでいるかチェックする。"""
        return STOP in text or TRANSFER in text or OUT_OF_SCOPE in text

    def assistant_agent(self, assistant_chat_client: OpenAIChatClient) -> ChatAgent:
        """アシスタントAgentを作成する。

        ユーザーはこのメソッドをオーバーライドしてカスタムのアシスタントAgentを提供可能。

        Args:
            assistant_chat_client: アシスタントAgent用のチャットクライアント。

        Returns:
            アシスタントAgent。

        """
        # tau2環境を初期化し、ツールとポリシーを抽出する これはドメイン固有のコンテキスト（この場合は航空会社のカスタマーサービス）を提供する
        env = get_environment()
        tools = env.get_tools()  # アシスタントが取れる利用可能なアクション
        policy = env.get_policy()  # アシスタントが従うべきガイドライン

        logger.info(
            f"Environment has {len(env.get_tools())} tools: {', '.join([tool.name for tool in env.get_tools()])}"
        )

        # tau2のツールをAgentフレームワークのAIFunction形式に変換する
        # これはtau2のツールシステムとAgentフレームワークの期待値のギャップを埋めるもの
        ai_functions = [convert_tau2_tool_to_ai_function(tool) for tool in tools]

        # 一般的なカスタマーサービスの振る舞いと特定のポリシーガイドラインを組み合わせる
        assistant_system_prompt = f"""<instructions>
{ASSISTANT_AGENT_INSTRUCTION}
</instructions>
<policy>
{policy}
</policy>"""

        # アシスタントAgentは以下を持つ: - すべてのドメインツールへのアクセス（予約、キャンセルなど） -
        # トークン制限内で長い会話を扱うためのスライディングウィンドウメモリ - 温度制御されたレスポンス生成
        return ChatAgent(
            chat_client=assistant_chat_client,
            instructions=assistant_system_prompt,
            tools=ai_functions,
            temperature=self.assistant_sampling_temperature,
            chat_message_store_factory=lambda: SlidingWindowChatMessageStore(
                system_message=assistant_system_prompt,
                tool_definitions=[tool.openai_schema for tool in tools],
                max_tokens=self.assistant_window_size,
            ),
        )

    def user_simulator(self, user_simuator_chat_client: OpenAIChatClient, task: Task) -> ChatAgent:
        """ユーザーシミュレータAgentを作成する。

        ユーザーはこのメソッドをオーバーライドしてカスタムのユーザーシミュレータAgentを提供可能。

        Args:
            user_simuator_chat_client: ユーザーシミュレータAgent用のチャットクライアント。
            task: 実行するタスク。

        Returns:
            ユーザーシミュレータAgent。

        """
        # ユーザーシミュレータはtau2のガイドラインに従い現実的な顧客行動を模倣する ツールは利用不可 - ユーザーは通常システムに直接アクセスしない
        user_sim_guidelines = get_global_user_sim_guidelines(use_tools=False)

        # ユーザーシミュレータのプロンプトは一般的なガイドラインとタスク固有のシナリオを組み合わせる
        user_sim_system_prompt = f"""{user_sim_guidelines}
<scenario>
{task.user_scenario.instructions}
</scenario>"""

        return ChatAgent(
            chat_client=user_simuator_chat_client,
            instructions=user_sim_system_prompt,
            temperature=0.0,
            # ユーザーシミュレータにはスライディングウィンドウはなく、会話全体のコンテキストを保持する TODO(yuge):
            # 将来的により現実的なシナリオのためにユーザーツールの追加を検討する
        )

    async def conversation_orchestrator(
        self, response: AgentExecutorResponse, ctx: WorkflowContext[AgentExecutorRequest]
    ) -> None:
        """アシスタントとユーザーシミュレータ間の会話フローをオーケストレーションする。

        これは中央のルーティングハブであり:

        1. アシスタントAgentまたはユーザーシミュレータからのレスポンスを受け取る
        2. メッセージのロールを反転させて適切な会話フローを作成する（assistant->user, user->assistant）
        3. 反転したメッセージを適切なターゲットAgentにルーティングする
        4. 終了条件が満たされるまで会話ループを維持する

        Args:
            response: アシスタントまたはユーザーシミュレータAgentからのレスポンス
            ctx: 他のexecutorにメッセージを送るためのワークフローコンテキスト

        """
        # 適切な会話フローのためにメッセージのロールを反転させる アシスタントのメッセージはユーザーメッセージになり、その逆も同様
        flipped = flip_messages(response.agent_run_response.messages)

        # 送信元を判定して正しいターゲットにルーティングする
        is_from_agent = response.executor_id == ASSISTANT_AGENT_ID

        # 反転したメッセージを反対のAgentに送信する 重要: ターゲットIDを指定しないと両方のAgentにブロードキャストされてしまう
        await ctx.send_message(
            AgentExecutorRequest(messages=flipped, should_respond=True),
            target_id=USER_SIMULATOR_ID if is_from_agent else ASSISTANT_AGENT_ID,
        )

    def build_conversation_workflow(self, assistant_agent: ChatAgent, user_simulator_agent: ChatAgent) -> Workflow:
        """会話ワークフローを構築する。

        ユーザーはこのメソッドをオーバーライドしてカスタムの会話ワークフローを提供可能。

        Args:
            assistant_agent: アシスタントAgent。
            user_simulator_agent: ユーザーシミュレータAgent。

        Returns:
            会話ワークフロー。

        """
        # STEP 1: ワークフローexecutorを作成する 各executorはAgentまたは関数をラップしてワークフローのオーケストレーションを行う
        self._assistant_executor = AgentExecutor(assistant_agent, id=ASSISTANT_AGENT_ID)
        self._user_executor = AgentExecutor(user_simulator_agent, id=USER_SIMULATOR_ID)
        orchestrator = FunctionExecutor(func=self.conversation_orchestrator, id=ORCHESTRATOR_ID)

        # STEP 2: 会話ワークフローを構築する 循環ワークフローを作成: Orchestrator -> Assistant -> Orchestrator
        # -> User -> Orchestrator...
        # Orchestratorはメッセージのロールを反転し適切なAgentにルーティングするメッセージルーターとして機能する
        return (
            WorkflowBuilder(max_iterations=10000)  # Unlimited - we control termination via should_not_stop
            .set_start_executor(orchestrator)  # Orchestrator manages the conversation flow
            .add_edge(orchestrator, self._assistant_executor)  # Route messages to assistant
            .add_edge(
                self._assistant_executor, orchestrator, condition=self.should_not_stop
            )  # Check termination after assistant
            .add_edge(orchestrator, self._user_executor)  # Route messages to user simulator
            .add_edge(self._user_executor, orchestrator, condition=self.should_not_stop)  # Check termination after user
            .build()
        )

    async def run(
        self,
        task: Task,
        assistant_chat_client: OpenAIChatClient,
        user_simuator_chat_client: OpenAIChatClient,
    ) -> list[ChatMessage]:
        """ワークフローベースのAgentオーケストレーションを使ってtau2タスクを実行する。

        このメソッドは複雑なマルチAgentシミュレーションをオーケストレーションする:

        1. tau2環境をセットアップし、ツールをAgentフレームワーク互換に変換
        2. 2つのAgentを作成: ツール付きのアシスタントとツールなしのユーザーシミュレータ
        3. Agent間のメッセージルーティングをオーケストレーションするワークフローを構築
        4. 終了条件が満たされるまで会話フローを管理
        5. 評価用に完全な会話履歴を返す

        Args:
            task: シナリオ、ポリシー、評価基準を含むtau2タスク
            assistant_chat_client: アシスタントAgent用のLLMクライアント
            user_simuator_chat_client: ユーザーシミュレータ用のLLMクライアント

        Returns:
            評価用のChatMessageリストとしての完全な会話履歴

        """
        logger.info(f"Starting workflow agent for task {task.id}: {task.description.purpose}")  # type: ignore[unused-ignore]
        logger.info(f"Assistant chat client: {assistant_chat_client}")
        logger.info(f"User simulator chat client: {user_simuator_chat_client}")

        # STEP 1: Agentを作成する
        assistant_agent = self.assistant_agent(assistant_chat_client)
        user_simulator_agent = self.user_simulator(user_simuator_chat_client, task)

        # STEP 2: 会話ワークフローを作成する
        workflow = self.build_conversation_workflow(assistant_agent, user_simulator_agent)

        # STEP 3: 標準的な挨拶で会話を初期化する tau2の期待する会話開始パターンに一致させる
        logger.info(f"Starting workflow with hardcoded greeting: '{DEFAULT_FIRST_AGENT_MESSAGE}'")

        first_message = ChatMessage(Role.ASSISTANT, text=DEFAULT_FIRST_AGENT_MESSAGE)
        initial_greeting = AgentExecutorResponse(
            executor_id=ASSISTANT_AGENT_ID,
            agent_run_response=AgentRunResponse(messages=[first_message]),
            full_conversation=[ChatMessage(Role.ASSISTANT, text=DEFAULT_FIRST_AGENT_MESSAGE)],
        )

        # STEP 4: ワークフローを実行し結果を収集する 終了条件（最大ステップ数、停止信号など）が満たされるまでワークフローを実行する
        await workflow.run(initial_greeting)

        # STEP 5: 評価に必要な会話履歴をまとめる。 3つの部分から成る: 1. 初期挨拶 2.
        # アシスタントのメッセージストア（単なる切り詰められたウィンドウではない） 3. 最終ユーザーメッセージ（あれば）
        assistant_executor = cast(AgentExecutor, self._assistant_executor)
        message_store = cast(SlidingWindowChatMessageStore, assistant_executor._agent_thread.message_store)
        full_conversation = [first_message] + await message_store.list_all_messages()
        if self._final_user_message is not None:
            full_conversation.extend(self._final_user_message)

        logger.opt(colors=True).info(
            f"<green>WORKFLOW COMPLETED WITH {len(full_conversation)} MESSAGES. "
            f"Termination reason: {self.termination_reason}.</green>"
        )
        log_messages(full_conversation)

        return full_conversation

    def evaluate(
        self, task_input: Task, conversation: list[ChatMessage], termination_reason: TerminationReason | None
    ) -> float:
        """tau2の包括的な評価システムを使ってAgentのパフォーマンスを評価する。

        Agentフレームワークの会話結果とtau2の評価パイプラインを橋渡しする。
        タスク完了度、ポリシー遵守、会話品質、ツール使用を考慮する。

        Args:
            task_input: 評価基準を含む元のtau2タスク
            conversation: ワークフロー実行からの完全な会話履歴
            termination_reason: 会話終了の理由（スコアに影響）

        Returns:
            全体のパフォーマンスを表す数値報酬スコア（0.0-1.0）

        Side Effects:
            詳細な評価結果をself.full_reward_infoに保存する

        """
        # 終了理由が欠落している場合の処理（予期しないワークフロー終了時に発生する可能性あり）
        if termination_reason is None:
            termination_reason = TerminationReason.TOO_MANY_ERRORS

        # AgentフレームワークのChatMessagesをtau2のMessage形式に変換して評価用にする
        tau2_messages = convert_agent_framework_messages_to_tau2_messages(conversation)

        # 会話とメタデータをtau2の評価システム用にパッケージ化する
        simulation = SimulationRun(
            id=str(uuid.uuid4()),  # Unique identifier for this evaluation run
            task_id=task_input.id,  # Links evaluation back to original task
            start_time=get_now(),  # Timestamp for evaluation records
            end_time=get_now(),  # Duration is 0 since this is post-hoc evaluation
            duration=0.0,
            termination_reason=termination_reason,  # Context for how conversation ended
            messages=tau2_messages,  # The actual conversation to evaluate
        )

        # 包括的な多次元評価を実行します EvaluationType.ALL: タスクの完了度、ポリシー遵守、会話の質などを評価します。
        # solo_mode=False: マルチエージェント会話（アシスタント＋ユーザーシミュレーター）を示します。
        self.full_reward_info = evaluate_simulation(
            simulation=simulation,
            task=task_input,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain="airline",
        )

        logger.info(
            f"Evaluation completed - Reward: {self.full_reward_info.reward if self.full_reward_info else None}, "
            f"Info: {self.full_reward_info}"
        )
        return self.full_reward_info.reward if self.full_reward_info else 0.0
