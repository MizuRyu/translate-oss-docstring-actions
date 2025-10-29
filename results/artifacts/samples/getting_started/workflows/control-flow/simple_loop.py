# Copyright (c) Microsoft. All rights reserved.

import asyncio
from enum import Enum

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    ChatMessage,
    Executor,
    ExecutorCompletedEvent,
    Role,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,
    handler,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

"""
Sample: Simple Loop (with an Agent Judge)

What it does:
- Guesser performs a binary search; judge is an agent that returns ABOVE/BELOW/MATCHED.
- Demonstrates feedback loops in workflows with agent steps.
- The workflow completes when the correct number is guessed.

Prerequisites:
- Azure AI/ Azure OpenAI for `AzureOpenAIChatClient` agent.
- Authentication via `azure-identity` — uses `AzureCliCredential()` (run `az login`).
"""


class NumberSignal(Enum):
    """ワークフローの数値信号を表すEnum。"""

    # ターゲットの数は推測より大きい。
    ABOVE = "above"
    # ターゲットの数は推測より小さい。
    BELOW = "below"
    # 推測がターゲットの数と一致している。
    MATCHED = "matched"
    # 推測プロセスを開始するための初期信号。
    INIT = "init"


class GuessNumberExecutor(Executor):
    """数値を推測するexecutor。"""

    def __init__(self, bound: tuple[int, int], id: str | None = None):
        """ターゲットの数でexecutorを初期化します。"""
        super().__init__(id=id or "guess_number")
        self._lower = bound[0]
        self._upper = bound[1]

    @handler
    async def guess_number(self, feedback: NumberSignal, ctx: WorkflowContext[int, str]) -> None:
        """数値を推測してタスクを実行します。"""
        if feedback == NumberSignal.INIT:
            self._guess = (self._lower + self._upper) // 2
            await ctx.send_message(self._guess)
        elif feedback == NumberSignal.MATCHED:
            # 前回の推測が正解でした。
            await ctx.yield_output(f"Guessed the number: {self._guess}")
        elif feedback == NumberSignal.ABOVE:
            # 前回の推測が低すぎました。 下限を前回の推測に更新します。 新しい範囲内の数値を生成します。
            self._lower = self._guess + 1
            self._guess = (self._lower + self._upper) // 2
            await ctx.send_message(self._guess)
        else:
            # 前回の推測が高すぎました。 上限を前回の推測に更新します。 新しい範囲内の数値を生成します。
            self._upper = self._guess - 1
            self._guess = (self._lower + self._upper) // 2
            await ctx.send_message(self._guess)


class SubmitToJudgeAgent(Executor):
    """数値の推測をjudge agentに送り、ABOVE/BELOW/MATCHEDで応答を受け取ります。"""

    def __init__(self, judge_agent_id: str, target: int, id: str | None = None):
        super().__init__(id=id or "submit_to_judge")
        self._judge_agent_id = judge_agent_id
        self._target = target

    @handler
    async def submit(self, guess: int, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
        prompt = (
            "You are a number judge. Given a target number and a guess, reply with exactly one token:"
            " 'MATCHED' if guess == target, 'ABOVE' if the target is above the guess,"
            " or 'BELOW' if the target is below.\n"
            f"Target: {self._target}\nGuess: {guess}\nResponse:"
        )
        await ctx.send_message(
            AgentExecutorRequest(messages=[ChatMessage(Role.USER, text=prompt)], should_respond=True),
            target_id=self._judge_agent_id,
        )


class ParseJudgeResponse(Executor):
    """AgentExecutorResponseをNumberSignalに解析してループに使用します。"""

    @handler
    async def parse(self, response: AgentExecutorResponse, ctx: WorkflowContext[NumberSignal]) -> None:
        text = response.agent_run_response.text.strip().upper()
        if "MATCHED" in text:
            await ctx.send_message(NumberSignal.MATCHED)
        elif "ABOVE" in text and "BELOW" not in text:
            await ctx.send_message(NumberSignal.ABOVE)
        else:
            await ctx.send_message(NumberSignal.BELOW)


async def main():
    """ワークフローを実行するメイン関数。"""
    # ステップ1: executorを作成します。
    guess_number_executor = GuessNumberExecutor((1, 100))

    # Agent judgeのセットアップ。
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    judge_agent = AgentExecutor(
        chat_client.create_agent(
            instructions=(
                "You strictly respond with one of: MATCHED, ABOVE, BELOW based on the given target and guess."
            )
        ),
        id="judge_agent",
    )
    submit_to_judge = SubmitToJudgeAgent(judge_agent_id=judge_agent.id, target=30, id="submit_judge")
    parse_judge = ParseJudgeResponse(id="parse_judge")

    # ステップ2: 定義したエッジでワークフローを構築します。 今回はワークフローにループを作成します。
    workflow = (
        WorkflowBuilder()
        .add_edge(guess_number_executor, submit_to_judge)
        .add_edge(submit_to_judge, judge_agent)
        .add_edge(judge_agent, parse_judge)
        .add_edge(parse_judge, guess_number_executor)
        .set_start_executor(guess_number_executor)
        .build()
    )

    # ステップ3: ワークフローを実行し、イベントを出力します。
    iterations = 0
    async for event in workflow.run_stream(NumberSignal.INIT):
        if isinstance(event, ExecutorCompletedEvent) and event.executor_id == guess_number_executor.id:
            iterations += 1
        elif isinstance(event, WorkflowOutputEvent):
            print(f"Final result: {event.data}")
        print(f"Event: {event}")

    # これは本質的に二分探索なので、反復回数は対数的です。 最大反復回数は[log2(範囲サイズ)]です。範囲が1から100の場合、これはlog2(100)で7です。
    # 最後のラウンドはMATCHEDイベントなので差し引きます。
    print(f"Guessed {iterations - 1} times.")


if __name__ == "__main__":
    asyncio.run(main())
