# Copyright (c) Microsoft. All rights reserved.

import asyncio
import contextlib
import json
import logging
import re
import sys
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeVar, Union, cast
from uuid import uuid4

from agent_framework import (
    AgentProtocol,
    AgentRunResponse,
    AgentRunResponseUpdate,
    ChatClientProtocol,
    ChatMessage,
    FunctionCallContent,
    FunctionResultContent,
    Role,
)

from ._base_group_chat_orchestrator import BaseGroupChatOrchestrator
from ._checkpoint import CheckpointStorage, WorkflowCheckpoint
from ._const import EXECUTOR_STATE_KEY
from ._events import WorkflowEvent
from ._executor import Executor, handler
from ._group_chat import (
    GroupChatBuilder,
    _GroupChatConfig,  # type: ignore[reportPrivateUsage]
    _GroupChatParticipantPipeline,  # type: ignore[reportPrivateUsage]
    _GroupChatRequestMessage,  # type: ignore[reportPrivateUsage]
    _GroupChatResponseMessage,  # type: ignore[reportPrivateUsage]
    group_chat_orchestrator,
)
from ._message_utils import normalize_messages_input
from ._model_utils import DictConvertible, encode_value
from ._participant_utils import GroupChatParticipantSpec, participant_description
from ._request_info_executor import RequestInfoExecutor, RequestInfoMessage, RequestResponse
from ._workflow import Workflow, WorkflowRunResult
from ._workflow_context import WorkflowContext

if sys.version_info >= (3, 11):
    from typing import Self  # pragma: no cover
else:
    from typing_extensions import Self  # pragma: no cover

logger = logging.getLogger(__name__)

# Magenticマネージャー/オーケストレーターが生成するメッセージの一貫した著者名
MAGENTIC_MANAGER_NAME = "magentic_manager"

# 汎用オーケストレーターのメッセージコールバックのための任意の種類
ORCH_MSG_KIND_USER_TASK = "user_task"
ORCH_MSG_KIND_TASK_LEDGER = "task_ledger"
# 統一されたコールバックコンシューマのための新たに公開された種類
ORCH_MSG_KIND_INSTRUCTION = "instruction"
ORCH_MSG_KIND_NOTICE = "notice"


def _message_to_payload(message: ChatMessage) -> Any:
    if hasattr(message, "to_dict") and callable(getattr(message, "to_dict", None)):
        with contextlib.suppress(Exception):
            return message.to_dict()  # type: ignore[attr-defined]
    if hasattr(message, "to_json") and callable(getattr(message, "to_json", None)):
        with contextlib.suppress(Exception):
            json_payload = message.to_json()  # type: ignore[attr-defined]
            if isinstance(json_payload, str):
                with contextlib.suppress(Exception):
                    return json.loads(json_payload)
            return json_payload
    if hasattr(message, "__dict__"):
        return encode_value(message.__dict__)
    return message


def _message_from_payload(payload: Any) -> ChatMessage:
    if isinstance(payload, ChatMessage):
        return payload
    if hasattr(ChatMessage, "from_dict") and isinstance(payload, dict):
        with contextlib.suppress(Exception):
            return ChatMessage.from_dict(payload)  # type: ignore[attr-defined,no-any-return]
    if hasattr(ChatMessage, "from_json") and isinstance(payload, str):
        with contextlib.suppress(Exception):
            return ChatMessage.from_json(payload)  # type: ignore[attr-defined,no-any-return]
    if isinstance(payload, dict):
        with contextlib.suppress(Exception):
            return ChatMessage(**payload)  # type: ignore[arg-type]
    if isinstance(payload, str):
        with contextlib.suppress(Exception):
            decoded = json.loads(payload)
            if isinstance(decoded, dict):
                return _message_from_payload(decoded)
    raise TypeError("Unable to reconstruct ChatMessage from payload")


# region Unified callback API (developer-facing)


@dataclass
class MagenticOrchestratorMessageEvent(WorkflowEvent):
    orchestrator_id: str = ""
    message: ChatMessage | None = None
    kind: str = ""

    def __post_init__(self) -> None:
        super().__init__(data=self.message)


@dataclass
class MagenticAgentDeltaEvent(WorkflowEvent):
    agent_id: str | None = None
    text: str | None = None
    function_call_id: str | None = None
    function_call_name: str | None = None
    function_call_arguments: Any | None = None
    function_result_id: str | None = None
    function_result: Any | None = None
    role: Role | None = None

    def __post_init__(self) -> None:
        super().__init__(data=self.text)


@dataclass
class MagenticAgentMessageEvent(WorkflowEvent):
    agent_id: str = ""
    message: ChatMessage | None = None

    def __post_init__(self) -> None:
        super().__init__(data=self.message)


@dataclass
class MagenticFinalResultEvent(WorkflowEvent):
    message: ChatMessage | None = None

    def __post_init__(self) -> None:
        super().__init__(data=self.message)


MagenticCallbackEvent = Union[
    MagenticOrchestratorMessageEvent,
    MagenticAgentDeltaEvent,
    MagenticAgentMessageEvent,
    MagenticFinalResultEvent,
]


class CallbackSink(Protocol):
    async def __call__(self, event: MagenticCallbackEvent) -> None: ...


# endregion Unified callback API region Magentic One Prompts

ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT = """Below I will present you a request.

Before we begin addressing the request, please answer the following pre-survey to the best of your ability.
Keep in mind that you are Ken Jennings-level with trivia, and Mensa-level with puzzles, so there should be
a deep well to draw from.

Here is the request:

{task}

Here is the pre-survey:

    1. Please list any specific facts or figures that are GIVEN in the request itself. It is possible that
       there are none.
    2. Please list any facts that may need to be looked up, and WHERE SPECIFICALLY they might be found.
       In some cases, authoritative sources are mentioned in the request itself.
    3. Please list any facts that may need to be derived (e.g., via logical deduction, simulation, or computation)
    4. Please list any facts that are recalled from memory, hunches, well-reasoned guesses, etc.

When answering this survey, keep in mind that "facts" will typically be specific names, dates, statistics, etc.
Your answer should use headings:

    1. GIVEN OR VERIFIED FACTS
    2. FACTS TO LOOK UP
    3. FACTS TO DERIVE
    4. EDUCATED GUESSES

DO NOT include any other headings or sections in your response. DO NOT list next steps or plans until asked to do so.
"""

ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT = """Fantastic. To address this request we have assembled the following team:

{team}

Based on the team composition, and known and unknown facts, please devise a short bullet-point plan for addressing the
original request. Remember, there is no requirement to involve all team members. A team member's particular expertise
may not be needed for this task.
"""

# 元の動作を反映し、元帳を単一のassistantメッセージでレンダリングするために追加されました。
ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT = """
We are working to address the following user request:

{task}


To answer this request we have assembled the following team:

{team}


Here is an initial fact sheet to consider:

{facts}


Here is the plan to follow as best as possible:

{plan}
"""

ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT = """As a reminder, we are working to solve the following task:

{task}

It is clear we are not making as much progress as we would like, but we may have learned something new.
Please rewrite the following fact sheet, updating it to include anything new we have learned that may be helpful.

Example edits can include (but are not limited to) adding new guesses, moving educated guesses to verified facts
if appropriate, etc. Updates may be made to any section of the fact sheet, and more than one section of the fact
sheet can be edited. This is an especially good time to update educated guesses, so please at least add or update
one educated guess or hunch, and explain your reasoning.

Here is the old fact sheet:

{old_facts}
"""

ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT = """Please briefly explain what went wrong on this last run
(the root cause of the failure), and then come up with a new plan that takes steps and includes hints to overcome prior
challenges and especially avoids repeating the same mistakes. As before, the new plan should be concise, expressed in
bullet-point form, and consider the following team composition:

{team}
"""

ORCHESTRATOR_PROGRESS_LEDGER_PROMPT = """
Recall we are working on the following request:

{task}

And we have assembled the following team:

{team}

To make progress on the request, please answer the following questions, including necessary reasoning:

    - Is the request fully satisfied? (True if complete, or False if the original request has yet to be
      SUCCESSFULLY and FULLY addressed)
    - Are we in a loop where we are repeating the same requests and or getting the same responses as before?
      Loops can span multiple turns, and can include repeated actions like scrolling up or down more than a
      handful of times.
    - Are we making forward progress? (True if just starting, or recent messages are adding value. False if recent
      messages show evidence of being stuck in a loop or if there is evidence of significant barriers to success
      such as the inability to read from a required file)
    - Who should speak next? (select from: {names})
    - What instruction or question would you give this team member? (Phrase as if speaking directly to them, and
      include any specific information they may need)

Please output an answer in pure JSON format according to the following schema. The JSON object must be parsable as-is.
DO NOT OUTPUT ANYTHING OTHER THAN JSON, AND DO NOT DEVIATE FROM THIS SCHEMA:

{{
    "is_request_satisfied": {{

        "reason": string,
        "answer": boolean
    }},
    "is_in_loop": {{
        "reason": string,
        "answer": boolean
    }},
    "is_progress_being_made": {{
        "reason": string,
        "answer": boolean
    }},
    "next_speaker": {{
        "reason": string,
        "answer": string (select from: {names})
    }},
    "instruction_or_question": {{
        "reason": string,
        "answer": string
    }}
}}
"""

ORCHESTRATOR_FINAL_ANSWER_PROMPT = """
We are working on the following task:
{task}

We have completed the task.

The above messages contain the conversation that took place to complete the task.

Based on the information gathered, provide the final answer to the original request.
The answer should be phrased as if you were speaking to the user.
"""


# region Messages and Types


def _new_chat_history() -> list[ChatMessage]:
    """型チェッカーを満たすために、chat historyリストのデフォルトファクトリに型を付けました。"""
    return []


def _new_participant_descriptions() -> dict[str, str]:
    """型チェッカーを満たすために、participant descriptions辞書のデフォルトファクトリに型を付けました。"""
    return {}


def _new_chat_message_list() -> list[ChatMessage]:
    """型チェッカーを満たすために、ChatMessageリストのデフォルトファクトリに型を付けました。"""
    return []


@dataclass
class _MagenticStartMessage(DictConvertible):
    """内部用: magenticワークフローを開始するためのメッセージ。"""

    messages: list[ChatMessage] = field(default_factory=_new_chat_message_list)

    def __init__(
        self,
        messages: str | ChatMessage | Sequence[str] | Sequence[ChatMessage] | None = None,
        *,
        task: ChatMessage | None = None,
    ) -> None:
        normalized = normalize_messages_input(messages)
        if task is not None:
            normalized += normalize_messages_input(task)
        if not normalized:
            raise ValueError("MagenticStartMessage requires at least one message input.")
        self.messages: list[ChatMessage] = normalized

    @property
    def task(self) -> ChatMessage:
        """タスクの最終ユーザーメッセージ。"""
        return self.messages[-1]

    @classmethod
    def from_string(cls, task_text: str) -> "_MagenticStartMessage":
        """単純な文字列からMagenticStartMessageを作成します。"""
        return cls(task_text)

    def to_dict(self) -> dict[str, Any]:
        """メッセージの辞書表現を作成します。"""
        return {
            "messages": [message.to_dict() for message in self.messages],
            "task": self.task.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_MagenticStartMessage":
        """辞書から作成します。"""
        if "messages" in data:
            raw_messages = data["messages"]
            if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
                raise TypeError("MagenticStartMessage 'messages' must be a sequence.")
            messages: list[ChatMessage] = [ChatMessage.from_dict(raw) for raw in raw_messages]  # type: ignore[arg-type]
            return cls(messages)
        if "task" in data:
            task = ChatMessage.from_dict(data["task"])
            return cls(task)
        raise KeyError("Expected 'messages' or 'task' in MagenticStartMessage payload.")


@dataclass
class _MagenticRequestMessage(_GroupChatRequestMessage):
    """内部用: magenticワークフロー内のエージェント向けリクエストメッセージタイプ。"""

    task_context: str = ""


class _MagenticResponseMessage(_GroupChatResponseMessage):
    """内部用: レスポンスメッセージタイプ。

    オーケストレーターが発行する場合、すべてのエージェントへのブロードキャストとしてマークするか、
    名前で特定のエージェントをターゲットにすることができます。

    """

    def __init__(
        self,
        body: ChatMessage,
        target_agent: str | None = None,  # deliver only to this agent if set
        broadcast: bool = False,  # deliver to all agents if True
    ) -> None:
        agent_name = body.author_name or ""
        super().__init__(
            agent_name=agent_name,
            message=body,
        )
        self.body = body
        self.target_agent = target_agent
        self.broadcast = broadcast

    def to_dict(self) -> dict[str, Any]:
        """メッセージの辞書表現を作成します。"""
        return {"body": self.body.to_dict(), "target_agent": self.target_agent, "broadcast": self.broadcast}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "_MagenticResponseMessage":
        """辞書から作成します。"""
        body = ChatMessage.from_dict(value["body"])
        target_agent = value.get("target_agent")
        broadcast = value.get("broadcast", False)
        return cls(body=body, target_agent=target_agent, broadcast=broadcast)


@dataclass
class _MagenticPlanReviewRequest(RequestInfoMessage):
    """内部用: 実行前に計画をレビューし、オプションで編集するためのHuman-in-the-loopリクエスト。"""

    # RequestInfoMessageがデフォルトフィールド(request_id)を定義しているため、
    # サブクラスのフィールドもdataclassルールを満たすためにデフォルトを持つ必要があります。
    task_text: str = ""
    facts_text: str = ""
    plan_text: str = ""
    round_index: int = 0  # これまでのレビューラウンド数


class MagenticPlanReviewDecision(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"


@dataclass
class _MagenticPlanReviewReply:
    """内部用: 計画レビューリクエストに対する人間の返信。"""

    decision: MagenticPlanReviewDecision
    edited_plan_text: str | None = None  # 指定された場合、そのまま新しい計画テキストになります。
    comments: str | None = None  # 編集されたテキストが提供されなかった場合の再計画の指針。


@dataclass
class _MagenticTaskLedger(DictConvertible):
    """内部用: Standard Magenticマネージャーのタスク元帳。"""

    facts: ChatMessage
    plan: ChatMessage

    def to_dict(self) -> dict[str, Any]:
        return {"facts": _message_to_payload(self.facts), "plan": _message_to_payload(self.plan)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_MagenticTaskLedger":
        return cls(
            facts=_message_from_payload(data.get("facts")),
            plan=_message_from_payload(data.get("plan")),
        )


@dataclass
class _MagenticProgressLedgerItem(DictConvertible):
    """内部用: 進捗元帳アイテム。"""

    reason: str
    answer: str | bool

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "answer": self.answer}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_MagenticProgressLedgerItem":
        answer_value = data.get("answer")
        if not isinstance(answer_value, (str, bool)):
            answer_value = ""  # strまたはboolでない場合は空文字列をデフォルトにします。
        return cls(reason=data.get("reason", ""), answer=answer_value)


@dataclass
class _MagenticProgressLedger(DictConvertible):
    """内部用: ワークフローの進捗を追跡するための進捗元帳。"""

    is_request_satisfied: _MagenticProgressLedgerItem
    is_in_loop: _MagenticProgressLedgerItem
    is_progress_being_made: _MagenticProgressLedgerItem
    next_speaker: _MagenticProgressLedgerItem
    instruction_or_question: _MagenticProgressLedgerItem

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_request_satisfied": self.is_request_satisfied.to_dict(),
            "is_in_loop": self.is_in_loop.to_dict(),
            "is_progress_being_made": self.is_progress_being_made.to_dict(),
            "next_speaker": self.next_speaker.to_dict(),
            "instruction_or_question": self.instruction_or_question.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_MagenticProgressLedger":
        return cls(
            is_request_satisfied=_MagenticProgressLedgerItem.from_dict(data.get("is_request_satisfied", {})),
            is_in_loop=_MagenticProgressLedgerItem.from_dict(data.get("is_in_loop", {})),
            is_progress_being_made=_MagenticProgressLedgerItem.from_dict(data.get("is_progress_being_made", {})),
            next_speaker=_MagenticProgressLedgerItem.from_dict(data.get("next_speaker", {})),
            instruction_or_question=_MagenticProgressLedgerItem.from_dict(data.get("instruction_or_question", {})),
        )


@dataclass
class MagenticContext(DictConvertible):
    """Magenticマネージャーのコンテキスト。"""

    task: ChatMessage
    chat_history: list[ChatMessage] = field(default_factory=_new_chat_history)
    participant_descriptions: dict[str, str] = field(default_factory=_new_participant_descriptions)
    round_count: int = 0
    stall_count: int = 0
    reset_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": _message_to_payload(self.task),
            "chat_history": [_message_to_payload(msg) for msg in self.chat_history],
            "participant_descriptions": dict(self.participant_descriptions),
            "round_count": self.round_count,
            "stall_count": self.stall_count,
            "reset_count": self.reset_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MagenticContext":
        chat_history_payload = data.get("chat_history", [])
        history: list[ChatMessage] = []
        for item in chat_history_payload:
            history.append(_message_from_payload(item))
        return cls(
            task=_message_from_payload(data.get("task")),
            chat_history=history,
            participant_descriptions=dict(data.get("participant_descriptions", {})),
            round_count=data.get("round_count", 0),
            stall_count=data.get("stall_count", 0),
            reset_count=data.get("reset_count", 0),
        )

    def reset(self) -> None:
        """コンテキストをリセットします。

        これによりチャット履歴がクリアされ、スタールカウントがリセットされます。
        タスク、ラウンドカウント、参加者説明はリセットされません。

        """
        self.chat_history.clear()
        self.stall_count = 0
        self.reset_count += 1


# endregion Messages and Types region Utilities


def _team_block(participants: dict[str, str]) -> str:
    """参加者説明を読みやすいブロックとしてレンダリングします。"""
    return "\n".join(f"- {name}: {desc}" for name, desc in participants.items())


def _first_assistant(messages: list[ChatMessage]) -> ChatMessage | None:
    for msg in reversed(messages):
        if msg.role == Role.ASSISTANT:
            return msg
    return None


def _extract_json(text: str) -> dict[str, Any]:
    """一時的なヘルパーメソッドの可能性があります。

    注意: このメソッドは現在必要です。なぜならChatClientProtocolがresponse.textを呼ぶと
    重複したJSONペイロードを返すためで、その原因を調査中です。

    `text`メソッドは複数のメッセージからのテキスト内容を連結して単一の文字列にしています。

    """
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fence:
        candidate = fence.group(1)
    else:
        # 最初のバランスの取れたJSONオブジェクトを見つけます。
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found.")
        depth = 0
        end = None
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            raise ValueError("Unbalanced JSON braces.")
        candidate = text[start:end]

    for attempt in (candidate, candidate.replace("True", "true").replace("False", "false").replace("None", "null")):
        with contextlib.suppress(Exception):
            val = json.loads(attempt)
            if isinstance(val, dict):
                return cast(dict[str, Any], val)

    with contextlib.suppress(Exception):
        import ast

        obj = ast.literal_eval(candidate)
        if isinstance(obj, dict):
            return cast(dict[str, Any], obj)

    raise ValueError("Unable to parse JSON from model output.")


T = TypeVar("T")


def _coerce_model(model_cls: type[T], data: dict[str, Any]) -> T:
    # 動的属性アクセスのmypyエラーを抑制するためにtype: ignoreを使用します。 最初にhasattr()でチェックしているので安全です。
    if hasattr(model_cls, "from_dict") and callable(model_cls.from_dict):  # type: ignore[attr-defined]
        return model_cls.from_dict(data)  # type: ignore[attr-defined,return-value,no-any-return]
    return model_cls(**data)  # type: ignore[arg-type,call-arg]


# endregion Utilities region Magentic Manager


class MagenticManagerBase(ABC):
    """Magentic Oneマネージャーの基底クラス。"""

    def __init__(
        self,
        *,
        max_stall_count: int = 3,
        max_reset_count: int | None = None,
        max_round_count: int | None = None,
    ) -> None:
        self.max_stall_count = max_stall_count
        self.max_reset_count = max_reset_count
        self.max_round_count = max_round_count
        # 型安全のための基本プロンプトサーフェス。具体的なマネージャーはstrフィールドでオーバーライド可能です。
        self.task_ledger_full_prompt: str = ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT

    @abstractmethod
    async def plan(self, magentic_context: MagenticContext) -> ChatMessage:
        """タスクの計画を作成します。"""
        ...

    @abstractmethod
    async def replan(self, magentic_context: MagenticContext) -> ChatMessage:
        """タスクの再計画を行います。"""
        ...

    @abstractmethod
    async def create_progress_ledger(self, magentic_context: MagenticContext) -> _MagenticProgressLedger:
        """進捗元帳を作成します。"""
        ...

    @abstractmethod
    async def prepare_final_answer(self, magentic_context: MagenticContext) -> ChatMessage:
        """最終回答を準備します。"""
        ...

    def snapshot_state(self) -> dict[str, Any]:
        """チェックポイント用にランタイム状態をシリアライズします。"""
        return {}

    def restore_state(self, state: dict[str, Any]) -> None:
        """チェックポイントデータからランタイム状態を復元します。"""
        return


class StandardMagenticManager(MagenticManagerBase):
    """ChatAgentを介して実際のLLM呼び出しを行うStandard Magenticマネージャー。

    マネージャーは元のMagentic Oneオーケストレーションを反映するプロンプトを構築します:
    - 事実収集
    - 計画作成
    - JSON形式の進捗元帳
    - リセット時の事実更新と計画更新
    - 最終回答の合成

    """

    task_ledger: _MagenticTaskLedger | None

    def snapshot_state(self) -> dict[str, Any]:
        state = super().snapshot_state()
        if self.task_ledger is not None:
            state = dict(state)
            state["task_ledger"] = self.task_ledger.to_dict()
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        super().restore_state(state)
        ledger = state.get("task_ledger")
        if ledger is not None:
            try:
                self.task_ledger = _MagenticTaskLedger.from_dict(ledger)
            except Exception:  # pragma: no cover - defensive
                logger.warning("Failed to restore manager task ledger from checkpoint state")

    def __init__(
        self,
        chat_client: ChatClientProtocol,
        task_ledger: _MagenticTaskLedger | None = None,
        *,
        instructions: str | None = None,
        task_ledger_facts_prompt: str | None = None,
        task_ledger_plan_prompt: str | None = None,
        task_ledger_full_prompt: str | None = None,
        task_ledger_facts_update_prompt: str | None = None,
        task_ledger_plan_update_prompt: str | None = None,
        progress_ledger_prompt: str | None = None,
        final_answer_prompt: str | None = None,
        max_stall_count: int = 3,
        max_reset_count: int | None = None,
        max_round_count: int | None = None,
        progress_ledger_retry_count: int | None = None,
    ) -> None:
        """Standard Magentic Managerを初期化します。

        Args:
            chat_client: LLM呼び出しに使用するチャットクライアント。
            instructions: オーケストレーターAgentへの指示。

        Keyword Args:
            task_ledger: タスク状態管理用のオプションのタスク元帳。
            task_ledger_facts_prompt: タスク元帳の事実用オプションプロンプト。
            task_ledger_plan_prompt: タスク元帳の計画用オプションプロンプト。
            task_ledger_full_prompt: タスク元帳全体用オプションプロンプト。
            task_ledger_facts_update_prompt: タスク元帳事実更新用オプションプロンプト。
            task_ledger_plan_update_prompt: タスク元帳計画更新用オプションプロンプト。
            progress_ledger_prompt: 進捗元帳用オプションプロンプト。
            final_answer_prompt: 最終回答用オプションプロンプト。
            max_stall_count: 最大スタール回数。
            max_reset_count: 最大リセット回数。
            max_round_count: 最大ラウンド数。
            progress_ledger_retry_count: 進捗元帳の最大リトライ回数。

        """
        super().__init__(
            max_stall_count=max_stall_count,
            max_reset_count=max_reset_count,
            max_round_count=max_round_count,
        )

        self.chat_client: ChatClientProtocol = chat_client
        self.instructions: str | None = instructions
        self.task_ledger: _MagenticTaskLedger | None = task_ledger

        # 必要に応じてプロンプトをオーバーライド可能です。
        self.task_ledger_facts_prompt: str = task_ledger_facts_prompt or ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT
        self.task_ledger_plan_prompt: str = task_ledger_plan_prompt or ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT
        self.task_ledger_full_prompt = task_ledger_full_prompt or ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT
        self.task_ledger_facts_update_prompt: str = (
            task_ledger_facts_update_prompt or ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT
        )
        self.task_ledger_plan_update_prompt: str = (
            task_ledger_plan_update_prompt or ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT
        )
        self.progress_ledger_prompt: str = progress_ledger_prompt or ORCHESTRATOR_PROGRESS_LEDGER_PROMPT
        self.final_answer_prompt: str = final_answer_prompt or ORCHESTRATOR_FINAL_ANSWER_PROMPT

        self.progress_ledger_retry_count: int = (
            progress_ledger_retry_count if progress_ledger_retry_count is not None else 3
        )

    async def _complete(
        self,
        messages: list[ChatMessage],
    ) -> ChatMessage:
        """基盤となるChatClientProtocolを直接呼び出し、最後のassistantメッセージを返します。

        マネージャーの指示が提供されている場合、それらはSYSTEMメッセージとして
        リクエストの先頭に挿入され、モデルを一貫してガイドします。
        中間のAgentラッパーは不要です。

        """
        # システム指示があれば先頭に追加します。
        request_messages: list[ChatMessage] = []
        if self.instructions:
            request_messages.append(ChatMessage(role=Role.SYSTEM, text=self.instructions))
        request_messages.extend(messages)

        # チャットクライアントの非ストリーミングAPIを呼び出します。
        response = await self.chat_client.get_response(request_messages)
        try:
            out_messages: list[ChatMessage] | None = list(response.messages)  # type: ignore[assignment]
        except Exception:
            out_messages = None

        if out_messages:
            last = out_messages[-1]
            return ChatMessage(
                role=last.role or Role.ASSISTANT,
                text=last.text or "",
                author_name=last.author_name or MAGENTIC_MANAGER_NAME,
            )

        # メッセージがない場合のフォールバック。
        return ChatMessage(role=Role.ASSISTANT, text="No output produced.", author_name=MAGENTIC_MANAGER_NAME)

    async def plan(self, magentic_context: MagenticContext) -> ChatMessage:
        """モデルを使って事実と計画を作成し、結合されたタスク元帳を単一のassistantメッセージとしてレンダリングします。"""
        task_text = magentic_context.task.text
        team_text = _team_block(magentic_context.participant_descriptions)

        # 事実を収集します。
        facts_user = ChatMessage(
            role=Role.USER,
            text=self.task_ledger_facts_prompt.format(task=task_text),
        )
        facts_msg = await self._complete([*magentic_context.chat_history, facts_user])

        # 計画を作成します。
        plan_user = ChatMessage(
            role=Role.USER,
            text=self.task_ledger_plan_prompt.format(team=team_text),
        )
        plan_msg = await self._complete([*magentic_context.chat_history, facts_user, facts_msg, plan_user])

        # 元帳を保存し、完全な結合ビューをレンダリングします。
        self.task_ledger = _MagenticTaskLedger(facts=facts_msg, plan=plan_msg)

        # より良いグラウンディングのために個別メッセージもchat_historyに保存します。 これにより進捗元帳モデルが詳細な推論にアクセスできます。
        magentic_context.chat_history.extend([facts_user, facts_msg, plan_user, plan_msg])

        combined = self.task_ledger_full_prompt.format(
            task=task_text,
            team=team_text,
            facts=facts_msg.text,
            plan=plan_msg.text,
        )
        return ChatMessage(role=Role.ASSISTANT, text=combined, author_name=MAGENTIC_MANAGER_NAME)

    async def replan(self, magentic_context: MagenticContext) -> ChatMessage:
        """スタールやループが検出された場合に事実と計画を更新します。"""
        if self.task_ledger is None:
            raise RuntimeError("replan() called before plan(); call plan() once before requesting a replan.")

        task_text = magentic_context.task.text
        team_text = _team_block(magentic_context.participant_descriptions)

        # 事実を更新します。
        facts_update_user = ChatMessage(
            role=Role.USER,
            text=self.task_ledger_facts_update_prompt.format(task=task_text, old_facts=self.task_ledger.facts.text),
        )
        updated_facts = await self._complete([*magentic_context.chat_history, facts_update_user])

        # 計画を更新します。
        plan_update_user = ChatMessage(
            role=Role.USER,
            text=self.task_ledger_plan_update_prompt.format(team=team_text),
        )
        updated_plan = await self._complete([
            *magentic_context.chat_history,
            facts_update_user,
            updated_facts,
            plan_update_user,
        ])

        # 保存してレンダリングします。
        self.task_ledger = _MagenticTaskLedger(facts=updated_facts, plan=updated_plan)

        # より良いグラウンディングのために個別メッセージもchat_historyに保存します。 これにより進捗元帳モデルが詳細な推論にアクセスできます。
        magentic_context.chat_history.extend([facts_update_user, updated_facts, plan_update_user, updated_plan])

        combined = self.task_ledger_full_prompt.format(
            task=task_text,
            team=team_text,
            facts=updated_facts.text,
            plan=updated_plan.text,
        )
        return ChatMessage(role=Role.ASSISTANT, text=combined, author_name=MAGENTIC_MANAGER_NAME)

    async def create_progress_ledger(self, magentic_context: MagenticContext) -> _MagenticProgressLedger:
        """これまでの会話に基づいてモデルにJSON形式の進捗元帳を生成させます。

        一時的な解析問題に対して軽量なリトライとバックオフを追加し、
        存在しない"unknown"エージェントの選択を避けます。
        参加者がいない場合は明確なエラーを発生させます。

        """
        agent_names = list(magentic_context.participant_descriptions.keys())
        if not agent_names:
            raise RuntimeError("No participants configured; cannot determine next speaker.")

        names_csv = ", ".join(agent_names)
        team_text = _team_block(magentic_context.participant_descriptions)

        prompt = self.progress_ledger_prompt.format(
            task=magentic_context.task.text,
            team=team_text,
            names=names_csv,
        )
        user_message = ChatMessage(role=Role.USER, text=prompt)

        # モデルが現在のステージを判断しやすいように完全なコンテキストを含め、小さなリトライループを設けます。
        attempts = 0
        last_error: Exception | None = None
        while attempts < self.progress_ledger_retry_count:
            raw = await self._complete([*magentic_context.chat_history, user_message])
            try:
                ledger_dict = _extract_json(raw.text)
                return _coerce_model(_MagenticProgressLedger, ledger_dict)
            except Exception as ex:
                last_error = ex
                attempts += 1
                logger.warning(
                    f"Progress ledger JSON parse failed (attempt {attempts}/{self.progress_ledger_retry_count}): {ex}"
                )
                if attempts < self.progress_ledger_retry_count:
                    # 次の試行までの短いバックオフ。
                    await asyncio.sleep(0.25 * attempts)

        raise RuntimeError(
            f"Progress ledger parse failed after {self.progress_ledger_retry_count} attempt(s): {last_error}"
        )

    async def prepare_final_answer(self, magentic_context: MagenticContext) -> ChatMessage:
        """モデルにユーザー宛の最終回答を生成させます。"""
        prompt = self.final_answer_prompt.format(task=magentic_context.task.text)
        user_message = ChatMessage(role=Role.USER, text=prompt)
        response = await self._complete([*magentic_context.chat_history, user_message])
        # 役割がassistantであることを保証します。
        return ChatMessage(
            role=Role.ASSISTANT,
            text=response.text,
            author_name=response.author_name or MAGENTIC_MANAGER_NAME,
        )


# endregion Magentic Manager region Magentic Executors


class MagenticOrchestratorExecutor(BaseGroupChatOrchestrator):
    """MagenticオーケストレーターExecutorで、すべてのオーケストレーションロジックを処理します。

    このExecutorはMagentic Oneワークフロー全体を管理します:
    - 初期計画とタスク元帳作成
    - 進捗追跡と完了検出
    - エージェントの調整とメッセージルーティング
    - リセットと再計画ロジック

    """

    # 型付き属性（__init__で初期化）
    _agent_executors: dict[str, "MagenticAgentExecutor"]
    _context: "MagenticContext | None"
    _task_ledger: "ChatMessage | None"
    _inner_loop_lock: asyncio.Lock
    _require_plan_signoff: bool
    _plan_review_round: int
    _max_plan_review_rounds: int
    _terminated: bool

    def __init__(
        self,
        manager: MagenticManagerBase,
        participants: dict[str, str],
        *,
        require_plan_signoff: bool = False,
        max_plan_review_rounds: int = 10,
        executor_id: str | None = None,
    ) -> None:
        """MagenticOrchestratorExecutorの新しいインスタンスを初期化します。

        Args:
            manager: Magenticマネージャーのインスタンス。
            participants: 参加者IDから名前への辞書。
            require_plan_signoff: 人間による計画承認が必要かどうか。
            max_plan_review_rounds: 計画レビューの最大ラウンド数。
            executor_id: オプションのExecutor ID。

        """
        super().__init__(executor_id or f"magentic_orchestrator_{uuid4().hex[:8]}")
        self._manager = manager
        self._participants = participants
        self._context = None
        self._task_ledger = None
        self._require_plan_signoff = require_plan_signoff
        self._plan_review_round = 0
        self._max_plan_review_rounds = max_plan_review_rounds
        # 内部調整用のエージェントExecutorのレジストリ（例: リセット）
        self._agent_executors = {}
        # 完了や制限後にさらなる処理を停止するための終端状態マーカー
        self._terminated = False
        # この実行でチェックポイント状態が適用されたかどうかを追跡します。
        self._state_restored = False

    def _get_author_name(self) -> str:
        """オーケストレーター生成メッセージ用のmagenticマネージャー名を取得します。"""
        return MAGENTIC_MANAGER_NAME

    def register_agent_executor(self, name: str, executor: "MagenticAgentExecutor") -> None:
        """内部制御用にエージェントExecutorを登録します（メッセージはなし）。"""
        self._agent_executors[name] = executor

    async def _emit_orchestrator_message(
        self,
        ctx: WorkflowContext[Any, ChatMessage],
        message: ChatMessage,
        kind: str,
    ) -> None:
        """オーケストレーターのメッセージをワークフローイベントストリームに発行します。

        オーケストレーターのメッセージは統一されたワークフローイベントストリームを通じて
        MagenticOrchestratorMessageEventインスタンスとして流れます。
        コンシューマはworkflow.run_stream()を通じてこれらのイベントを購読すべきです。

        Args:
            ctx: イベントをストリームに追加するためのワークフローコンテキスト
            message: 発行するオーケストレーターのメッセージ（task, plan, instruction, notice）
            kind: メッセージ分類（user_task, task_ledger, instruction, notice）

        Example:
            async for event in workflow.run_stream("task"):
                if isinstance(event, MagenticOrchestratorMessageEvent):
                    print(f"Orchestrator {event.kind}: {event.message.text}")

        """
        event = MagenticOrchestratorMessageEvent(
            orchestrator_id=self.id,
            message=message,
            kind=kind,
        )
        await ctx.add_event(event)

    def snapshot_state(self) -> dict[str, Any]:
        """チェックポイント用に現在のオーケストレーター状態をキャプチャします。

        OrchestrationStateを構造に使用しますが、既存のチェックポイントとの互換性のために
        Magenticの複雑なメタデータをトップレベルに保持します。

        Returns:
            チェックポイント永続化用の辞書

        """
        state: dict[str, Any] = {
            "plan_review_round": self._plan_review_round,
            "max_plan_review_rounds": self._max_plan_review_rounds,
            "require_plan_signoff": self._require_plan_signoff,
            "terminated": self._terminated,
        }
        if self._context is not None:
            state["magentic_context"] = self._context.to_dict()
        if self._task_ledger is not None:
            state["task_ledger"] = _message_to_payload(self._task_ledger)
        manager_state: dict[str, Any] | None = None
        with contextlib.suppress(Exception):
            manager_state = self._manager.snapshot_state()
        if manager_state:
            state["manager_state"] = manager_state
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        """チェックポイントからオーケストレーター状態を復元します。

        既存のMagenticチェックポイントとの互換性を維持しつつ
        OrchestrationState構造をサポートします。

        Args:
            state: チェックポイントデータの辞書

        """
        # 旧フォーマット（直接キー）と新フォーマット（OrchestrationStateでラップ）の両方をサポートします。
        if "metadata" in state and isinstance(state.get("metadata"), dict):
            # 新しいOrchestrationStateフォーマット - メタデータを抽出します。
            from ._orchestration_state import OrchestrationState

            orch_state = OrchestrationState.from_dict(state)
            state = orch_state.metadata

        ctx_payload = state.get("magentic_context")
        if ctx_payload is not None:
            try:
                if isinstance(ctx_payload, dict):
                    self._context = MagenticContext.from_dict(ctx_payload)  # type: ignore[arg-type]
                else:
                    self._context = None
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to restore magentic context: %s", exc)
                self._context = None
        ledger_payload = state.get("task_ledger")
        if ledger_payload is not None:
            try:
                self._task_ledger = _message_from_payload(ledger_payload)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to restore task ledger message: %s", exc)
                self._task_ledger = None

        if "plan_review_round" in state:
            try:
                self._plan_review_round = int(state["plan_review_round"])
            except Exception:  # pragma: no cover
                logger.debug("Ignoring invalid plan_review_round in checkpoint state")
        if "max_plan_review_rounds" in state:
            self._max_plan_review_rounds = state.get("max_plan_review_rounds")  # type: ignore[assignment]
        if "require_plan_signoff" in state:
            self._require_plan_signoff = bool(state.get("require_plan_signoff"))
        if "terminated" in state:
            self._terminated = bool(state.get("terminated"))

        manager_state = state.get("manager_state")
        if manager_state is not None:
            try:
                self._manager.restore_state(manager_state)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to restore manager state: %s", exc)

        self._reconcile_restored_participants()

    def _reconcile_restored_participants(self) -> None:
        """復元された参加者リストが現在のworkflow graphと一致していることを確認します。"""
        if self._context is None:
            return

        restored = self._context.participant_descriptions or {}
        expected = self._participants

        restored_names = set(restored.keys())
        expected_names = set(expected.keys())

        if restored_names != expected_names:
            missing = ", ".join(sorted(expected_names - restored_names)) or "none"
            unexpected = ", ".join(sorted(restored_names - expected_names)) or "none"
            raise RuntimeError(
                "Magentic checkpoint restore failed: participant names do not match the checkpoint. "
                "Ensure MagenticBuilder.participants keys remain stable across runs. "
                f"Missing names: {missing}; unexpected names: {unexpected}."
            )

        # prompt surfacesが再構築されたworkflow inputsを常に反映するように、説明を更新します。
        for name, description in expected.items():
            restored[name] = description

    def _snapshot_pattern_metadata(self) -> dict[str, Any]:
        """パターン固有のstateをシリアライズします。

        Magenticはbase classのフックの代わりにカスタムのsnapshot_state()を使用します。
        このメソッドはbase classの契約を満たすために存在します。

        Returns:
        空のdict（Magenticは自身のstateを管理します）

        """
        return {}

    def _restore_pattern_metadata(self, metadata: dict[str, Any]) -> None:
        """パターン固有のstateを復元します。

        Magenticはbase classのフックの代わりにカスタムのrestore_state()を使用します。
        このメソッドはbase classの契約を満たすために存在します。

        Args:
        metadata: パターン固有のstate dict（無視されます）

        """
        pass

    async def _ensure_state_restored(
        self,
        context: WorkflowContext[Any, Any],
    ) -> None:
        if self._state_restored and self._context is not None:
            return
        state = await context.get_executor_state()
        if not state:
            self._state_restored = True
            return
        if not isinstance(state, dict):
            self._state_restored = True
            return
        try:
            self.restore_state(state)
        except Exception as exc:  # pragma: no cover
            logger.warning("Magentic Orchestrator: Failed to apply checkpoint state: %s", exc, exc_info=True)
            raise
        else:
            self._state_restored = True

    @handler
    async def handle_start_message(
        self,
        message: _MagenticStartMessage,
        context: WorkflowContext[
            _MagenticResponseMessage | _MagenticRequestMessage | _MagenticPlanReviewRequest, ChatMessage
        ],
    ) -> None:
        """オーケストレーションを開始するための初期startメッセージを処理します。"""
        if getattr(self, "_terminated", False):
            return
        logger.info("Magentic Orchestrator: Received start message")

        self._context = MagenticContext(
            task=message.task,
            participant_descriptions=self._participants,
        )
        if message.messages:
            self._context.chat_history.extend(message.messages)
        self._state_restored = True
        # タスクの受領に対するオーケストレーターの非ストリーミングコールバック
        await self._emit_orchestrator_message(context, message.task, ORCH_MSG_KIND_USER_TASK)

        # 実際のモデル呼び出しを伴うマネージャーによる初期計画
        self._task_ledger = await self._manager.plan(self._context.clone(deep=True))

        # 人間の承認が必要な場合は、今すぐ尋ねて戻ります。レスポンスハンドラーが再開します。
        if self._require_plan_signoff:
            await self._send_plan_review_request(context)
            return

        # タスク台帳を会話履歴に追加します。
        self._context.chat_history.append(self._task_ledger)

        logger.debug("Task ledger created.")

        await self._emit_orchestrator_message(context, self._task_ledger, ORCH_MSG_KIND_TASK_LEDGER)

        # 内部ループを開始します。
        ctx2 = cast(
            WorkflowContext[_MagenticResponseMessage | _MagenticRequestMessage, ChatMessage],
            context,
        )
        await self._run_inner_loop(ctx2)

    @handler
    async def handle_task_text(
        self,
        task_text: str,
        context: WorkflowContext[
            _MagenticResponseMessage | _MagenticRequestMessage | _MagenticPlanReviewRequest, ChatMessage
        ],
    ) -> None:
        await self.handle_start_message(_MagenticStartMessage.from_string(task_text), context)

    @handler
    async def handle_task_message(
        self,
        task_message: ChatMessage,
        context: WorkflowContext[
            _MagenticResponseMessage | _MagenticRequestMessage | _MagenticPlanReviewRequest, ChatMessage
        ],
    ) -> None:
        await self.handle_start_message(_MagenticStartMessage(task_message), context)

    @handler
    async def handle_task_messages(
        self,
        conversation: list[ChatMessage],
        context: WorkflowContext[
            _MagenticResponseMessage | _MagenticRequestMessage | _MagenticPlanReviewRequest, ChatMessage
        ],
    ) -> None:
        await self.handle_start_message(_MagenticStartMessage(conversation), context)

    @handler
    async def handle_response_message(
        self,
        message: _MagenticResponseMessage,
        context: WorkflowContext[_MagenticResponseMessage | _MagenticRequestMessage, ChatMessage],
    ) -> None:
        """Agentからのレスポンスを処理します。"""
        if getattr(self, "_terminated", False):
            return
        await self._ensure_state_restored(context)
        if self._context is None:
            raise RuntimeError("Magentic Orchestrator: Received response but not initialized")

        logger.debug("Magentic Orchestrator: Received response from agent")

        # 必要に応じて転送メッセージを追加します。
        if message.body.role != Role.USER:
            transfer_msg = ChatMessage(
                role=Role.USER,
                text=f"Transferred to {getattr(message.body, 'author_name', 'agent')}",
            )
            self._context.chat_history.append(transfer_msg)

        # Agentのレスポンスをcontextに追加します。
        self._context.chat_history.append(message.body)

        # 内部ループを続行します。
        await self._run_inner_loop(context)

    @handler
    async def handle_plan_review_response(
        self,
        response: RequestResponse[_MagenticPlanReviewRequest, _MagenticPlanReviewReply],
        context: WorkflowContext[
            # 次に台帳をブロードキャストするか、別のレビューラウンドを求めるかもしれません。
            _MagenticResponseMessage | _MagenticRequestMessage | _MagenticPlanReviewRequest, ChatMessage
        ],
    ) -> None:
        if getattr(self, "_terminated", False):
            return
        await self._ensure_state_restored(context)
        if self._context is None:
            return

        human = response.data
        if human is None:  # type: ignore[unreachable]
            # 防御的フォールバック：空のコメントで修正として扱います。
            human = _MagenticPlanReviewReply(decision=MagenticPlanReviewDecision.REVISE, comments="")

        if human.decision == MagenticPlanReviewDecision.APPROVE:
            # 承認でレビューのループを閉じます（この実行ではこれ以上の計画レビュー要求はありません）。
            self._require_plan_signoff = False
            # ユーザーが編集済みの計画を提供した場合、それを採用します。
            if human.edited_plan_text:
                # マネージャーの内部台帳を更新し、結合メッセージを再構築します。
                mgr_ledger = getattr(self._manager, "task_ledger", None)
                if mgr_ledger is not None:
                    mgr_ledger.plan.text = human.edited_plan_text
                team_text = _team_block(self._participants)
                combined = self._manager.task_ledger_full_prompt.format(
                    task=self._context.task.text,
                    team=team_text,
                    facts=(mgr_ledger.facts.text if mgr_ledger else ""),
                    plan=human.edited_plan_text,
                )
                self._task_ledger = ChatMessage(
                    role=Role.ASSISTANT,
                    text=combined,
                    author_name=MAGENTIC_MANAGER_NAME,
                )
            # コメント付きで承認されたが編集テキストがない場合、コメントを再計画に適用して進めます（追加のレビューなし）。
            elif human.comments:
                # 基盤付けのために人間のフィードバックを記録します。
                self._context.chat_history.append(
                    ChatMessage(role=Role.USER, text=f"Human plan feedback: {human.comments}")
                )
                # コメントに基づいてマネージャーに再計画を依頼し、直ちに進めます。
                self._task_ledger = await self._manager.replan(self._context.clone(deep=True))

            # 承認済みの計画を記録します（ブロードキャストなし）。
            if self._task_ledger:
                self._context.chat_history.append(self._task_ledger)
                await self._emit_orchestrator_message(context, self._task_ledger, ORCH_MSG_KIND_TASK_LEDGER)

            # 通常の調整ループに入ります。
            ctx2 = cast(
                WorkflowContext[_MagenticResponseMessage | _MagenticRequestMessage, ChatMessage],
                context,
            )
            await self._run_inner_loop(ctx2)
            return

        # それ以外はREVISIONラウンドです。
        self._plan_review_round += 1
        if self._plan_review_round > self._max_plan_review_rounds:
            logger.warning("Magentic Orchestrator: Max plan review rounds reached. Proceeding with current plan.")
            # この実行の残り期間、これ以上の計画レビュー要求を停止します。
            self._require_plan_signoff = False
            # レビューが終了したことをユーザーに知らせるために会話に明確なメモを追加します。
            notice = ChatMessage(
                role=Role.ASSISTANT,
                text=(
                    "Plan review closed after max rounds. Proceeding with the current plan and will no longer "
                    "prompt for plan approval."
                ),
                author_name=MAGENTIC_MANAGER_NAME,
            )
            self._context.chat_history.append(notice)
            await self._emit_orchestrator_message(context, notice, ORCH_MSG_KIND_NOTICE)
            if self._task_ledger:
                self._context.chat_history.append(self._task_ledger)
                # これ以上のレビュー要求はなく、直接調整に進みます。
            ctx2 = cast(
                WorkflowContext[_MagenticResponseMessage | _MagenticRequestMessage, ChatMessage],
                context,
            )
            await self._run_inner_loop(ctx2)
            return

        # ユーザーが編集済みの計画を提供した場合、それを直接採用し、もう一度確認を求めます。
        if human.edited_plan_text:
            mgr_ledger2 = getattr(self._manager, "task_ledger", None)
            if mgr_ledger2 is not None:
                mgr_ledger2.plan.text = human.edited_plan_text
            # 次のレビュー要求でプレビューするために結合メッセージを再構築します。
            team_text = _team_block(self._participants)
            combined = self._manager.task_ledger_full_prompt.format(
                task=self._context.task.text,
                team=team_text,
                facts=(mgr_ledger2.facts.text if mgr_ledger2 else ""),
                plan=human.edited_plan_text,
            )
            self._task_ledger = ChatMessage(role=Role.ASSISTANT, text=combined, author_name=MAGENTIC_MANAGER_NAME)
            await self._send_plan_review_request(context)
            return

        # それ以外の場合、コメントをチャット履歴に渡し、マネージャーと再計画します。
        if human.comments:
            self._context.chat_history.append(
                ChatMessage(role=Role.USER, text=f"Human plan feedback: {human.comments}")
            )

        # マネージャーに再計画を依頼します。これは計画段階のみを調整し、完全なリセットではありません。
        self._task_ledger = await self._manager.replan(self._context.clone(deep=True))
        await self._send_plan_review_request(context)

    async def _run_outer_loop(
        self,
        context: WorkflowContext[_MagenticResponseMessage | _MagenticRequestMessage, ChatMessage],
    ) -> None:
        """外部オーケストレーションループを実行します - 計画フェーズ。"""
        if self._context is None:
            raise RuntimeError("Context not initialized")

        logger.info("Magentic Orchestrator: Outer loop - entering inner loop")

        # まだ存在しない場合はタスク台帳を履歴に追加します。
        if self._task_ledger and (
            not self._context.chat_history or self._context.chat_history[-1] != self._task_ledger
        ):
            self._context.chat_history.append(self._task_ledger)

        # オプションで更新されたタスク台帳をメッセージコールバックで表示します（ブロードキャストなし）。
        if self._task_ledger is not None:
            await self._emit_orchestrator_message(context, self._task_ledger, ORCH_MSG_KIND_TASK_LEDGER)

        # 内部ループを開始します。
        await self._run_inner_loop(context)

    async def _run_inner_loop(
        self,
        context: WorkflowContext[_MagenticResponseMessage | _MagenticRequestMessage, ChatMessage],
    ) -> None:
        """内部オーケストレーションループを実行します。調整フェーズ。ロックでシリアライズされます。"""
        if self._context is None or self._task_ledger is None:
            raise RuntimeError("Context or task ledger not initialized")

        await self._run_inner_loop_helper(context)

    async def _run_inner_loop_helper(
        self,
        context: WorkflowContext[_MagenticResponseMessage | _MagenticRequestMessage, ChatMessage],
    ) -> None:
        """排他アクセスで内部ループを実行します。"""
        # このメソッドの残りの部分でオプションのcontextを絞り込みます。
        ctx = self._context
        if ctx is None:
            raise RuntimeError("Context not initialized")
        # 最初に制限をチェックします。
        within_limits = await self._check_within_limits_or_complete(context)
        if not within_limits:
            return

        ctx.round_count += 1
        logger.info("Magentic Orchestrator: Inner loop - round %s", ctx.round_count)

        # マネージャーを使って進捗台帳を作成します。
        try:
            current_progress_ledger = await self._manager.create_progress_ledger(ctx.clone(deep=True))
        except Exception as ex:
            logger.warning("Magentic Orchestrator: Progress ledger creation failed, triggering reset: %s", ex)
            await self._reset_and_replan(context)
            return

        logger.debug(
            "Progress evaluation: satisfied=%s, next=%s",
            current_progress_ledger.is_request_satisfied.answer,
            current_progress_ledger.next_speaker.answer,
        )

        # タスク完了をチェックします。
        if current_progress_ledger.is_request_satisfied.answer:
            logger.info("Magentic Orchestrator: Task completed")
            await self._prepare_final_answer(context)
            return

        # 停滞やループをチェックします。
        if not current_progress_ledger.is_progress_being_made.answer or current_progress_ledger.is_in_loop.answer:
            ctx.stall_count += 1
        else:
            ctx.stall_count = max(0, ctx.stall_count - 1)

        if ctx.stall_count > self._manager.max_stall_count:
            logger.info("Magentic Orchestrator: Stalling detected. Resetting and replanning")
            await self._reset_and_replan(context)
            return

        # 次の話者と指示を決定します。
        answer_val = current_progress_ledger.next_speaker.answer
        if not isinstance(answer_val, str):
            # 台帳が非文字列を返した場合は最初の参加者にフォールバックします。
            logger.warning("Next speaker answer was not a string; selecting first participant as fallback")
            answer_val = next(iter(self._participants.keys()))
        next_speaker_value: str = answer_val
        instruction = current_progress_ledger.instruction_or_question.answer

        if next_speaker_value not in self._participants:
            logger.warning("Invalid next speaker: %s", next_speaker_value)
            await self._prepare_final_answer(context)
            return

        # 会話に指示を追加します（アシスタントのガイダンス）。
        instruction_msg = ChatMessage(
            role=Role.ASSISTANT,
            text=str(instruction),
            author_name=MAGENTIC_MANAGER_NAME,
        )
        ctx.chat_history.append(instruction_msg)
        await self._emit_orchestrator_message(context, instruction_msg, ORCH_MSG_KIND_INSTRUCTION)

        # 選択されたAgentのexecutor idを決定します。
        target_executor_id = f"agent_{next_speaker_value}"

        # 特定のAgentにレスポンスを要求します。
        logger.debug("Magentic Orchestrator: Requesting %s to respond", next_speaker_value)
        await context.send_message(
            _MagenticRequestMessage(
                agent_name=next_speaker_value,
                instruction=str(instruction),
                task_context=ctx.task.text,
            ),
            target_id=target_executor_id,
        )

    async def _reset_and_replan(
        self,
        context: WorkflowContext[_MagenticResponseMessage | _MagenticRequestMessage, ChatMessage],
    ) -> None:
        """contextをリセットして再計画します。"""
        if self._context is None:
            return

        logger.info("Magentic Orchestrator: Resetting and replanning")

        # contextをリセットします。
        self._context.reset()

        # 再計画します。
        self._task_ledger = await self._manager.replan(self._context.clone(deep=True))
        self._context.chat_history.append(self._task_ledger)
        await self._emit_orchestrator_message(context, self._task_ledger, ORCH_MSG_KIND_TASK_LEDGER)

        # 登録されたすべてのAgent executorを内部的にリセットします（ハンドラーやメッセージは関与しません）。
        for agent in self._agent_executors.values():
            with contextlib.suppress(Exception):
                agent.reset()

        # 外部ループを再起動します。
        await self._run_outer_loop(context)

    async def _prepare_final_answer(
        self,
        context: WorkflowContext[_MagenticResponseMessage | _MagenticRequestMessage, ChatMessage],
    ) -> None:
        """マネージャーを使って最終回答を準備します。"""
        if self._context is None:
            return

        logger.info("Magentic Orchestrator: Preparing final answer")
        final_answer = await self._manager.prepare_final_answer(self._context.clone(deep=True))

        # workflowの完了イベントを発行します。
        await context.yield_output(final_answer)
        await context.add_event(MagenticFinalResultEvent(message=final_answer))

    async def _check_within_limits_or_complete(
        self,
        context: WorkflowContext[_MagenticResponseMessage | _MagenticRequestMessage, ChatMessage],
    ) -> bool:
        """オーケストレーターが運用制限内にあるかチェックします。"""
        if self._context is None:
            return False
        ctx = self._context

        hit_round_limit = self._manager.max_round_count is not None and ctx.round_count >= self._manager.max_round_count
        hit_reset_limit = self._manager.max_reset_count is not None and ctx.reset_count >= self._manager.max_reset_count

        if hit_round_limit or hit_reset_limit:
            limit_type = "round" if hit_round_limit else "reset"
            logger.error("Magentic Orchestrator: Max %s count reached", limit_type)

            # 完了は一度だけ発行し、その後終了済みとしてマークします。
            if not self._terminated:
                self._terminated = True
                # 部分結果を取得します。
                partial_result = _first_assistant(ctx.chat_history)
                if partial_result is None:
                    partial_result = ChatMessage(
                        role=Role.ASSISTANT,
                        text=f"Stopped due to {limit_type} limit. No partial result available.",
                        author_name=MAGENTIC_MANAGER_NAME,
                    )

                # 部分結果をyieldし、完了を通知します。
                await context.yield_output(partial_result)
                await context.add_event(MagenticFinalResultEvent(message=partial_result))
            return False

        return True

    async def _send_plan_review_request(
        self,
        context: WorkflowContext[
            _MagenticResponseMessage | _MagenticRequestMessage | _MagenticPlanReviewRequest, ChatMessage
        ],
    ) -> None:
        """RequestInfoExecutorを介してPlanReviewRequestを発行します。"""
        # 計画承認が無効（例：レビューラウンドが尽きた）場合は何もしません。
        if not self._require_plan_signoff:
            return
        ledger = getattr(self._manager, "task_ledger", None)
        facts_text = ledger.facts.text if ledger else ""
        plan_text = ledger.plan.text if ledger else ""
        task_text = self._context.task.text if self._context else ""

        req = _MagenticPlanReviewRequest(
            task_text=task_text,
            facts_text=facts_text,
            plan_text=plan_text,
            round_index=self._plan_review_round,
        )
        await context.send_message(req)


# region Magentic Executors


class MagenticAgentExecutor(Executor):
    """Magentic agent executorはworkflowへの参加のためにAgentをラップします。

    強化されたAgentExecutorを活用し、会話注入フックを提供します：
    - タスク台帳のブロードキャスト受信
    - 特定Agentからのリクエストへの応答
    - 必要に応じたAgent stateのリセット

    """

    def __init__(
        self,
        agent: AgentProtocol | Executor,
        agent_id: str,
    ) -> None:
        super().__init__(f"agent_{agent_id}")
        self._agent = agent
        self._agent_id = agent_id
        self._chat_history: list[ChatMessage] = []
        self._state_restored = False

    def snapshot_state(self) -> dict[str, Any]:
        """チェックポイント用に現在のexecutor stateをキャプチャします。

        Returns:
        シリアライズされたチャット履歴を含むdict

        """
        from ._conversation_state import encode_chat_messages

        return {
            "chat_history": encode_chat_messages(self._chat_history),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """チェックポイントからexecutor stateを復元します。

        Args:
        state: チェックポイントデータdict

        """
        from ._conversation_state import decode_chat_messages

        history_payload = state.get("chat_history")
        if history_payload:
            try:
                self._chat_history = decode_chat_messages(history_payload)
            except Exception as exc:  # pragma: no cover
                logger.warning("Agent %s: Failed to restore chat history: %s", self._agent_id, exc)
                self._chat_history = []
        else:
            self._chat_history = []

    async def _ensure_state_restored(self, context: WorkflowContext[Any, Any]) -> None:
        if self._state_restored and self._chat_history:
            return
        state = await context.get_executor_state()
        if not state:
            self._state_restored = True
            return
        if not isinstance(state, dict):
            self._state_restored = True
            return
        try:
            self.restore_state(state)
        except Exception as exc:  # pragma: no cover
            logger.warning("Agent %s: Failed to apply checkpoint state: %s", self._agent_id, exc, exc_info=True)
            raise
        else:
            self._state_restored = True

    @handler
    async def handle_response_message(
        self, message: _MagenticResponseMessage, context: WorkflowContext[_MagenticResponseMessage]
    ) -> None:
        """レスポンスメッセージ（タスク台帳のブロードキャスト）を処理します。"""
        logger.debug("Agent %s: Received response message", self._agent_id)

        await self._ensure_state_restored(context)

        # このメッセージがこのAgent宛てかどうかをチェックします。
        if message.target_agent is not None and message.target_agent != self._agent_id and not message.broadcast:
            # メッセージは別のAgent宛てなので無視します。
            logger.debug("Agent %s: Ignoring message targeted to %s", self._agent_id, message.target_agent)
            return

        # 必要に応じて転送メッセージを追加します。
        if message.body.role != Role.USER:
            transfer_msg = ChatMessage(
                role=Role.USER,
                text=f"Transferred to {getattr(message.body, 'author_name', 'agent')}",
            )
            self._chat_history.append(transfer_msg)

        # Agentの履歴にメッセージを追加します。
        self._chat_history.append(message.body)

    def _get_persona_adoption_role(self) -> Role:
        """persona adoptionメッセージに最適なroleを決定します。

        AgentがSYSTEM roleをサポートする場合はそれを使用し、そうでなければUSERにフォールバックします。

        """
        # BaseAgent派生のAgentのみがSYSTEMメッセージを確実にサポートすると想定されます。
        from agent_framework import BaseAgent as _AF_AgentBase  # 循環を避けるためのローカルimport

        if isinstance(self._agent, _AF_AgentBase) and hasattr(self._agent, "chat_client"):
            return Role.SYSTEM
        # 他のAgentタイプやサポートが不明な場合はUSERを使用します。
        return Role.USER

    @handler
    async def handle_request_message(
        self, message: _MagenticRequestMessage, context: WorkflowContext[_MagenticResponseMessage, AgentRunResponse]
    ) -> None:
        """レスポンス要求を処理します。"""
        if message.agent_name != self._agent_id:
            return

        logger.info("Agent %s: Received request to respond", self._agent_id)

        await self._ensure_state_restored(context)

        # 適切なroleでpersona adoptionメッセージを追加します。
        persona_role = self._get_persona_adoption_role()
        persona_msg = ChatMessage(
            role=persona_role,
            text=f"Transferred to {self._agent_id}, adopt the persona immediately.",
        )
        self._chat_history.append(persona_msg)

        # オーケストレーターの指示をUSERメッセージとして追加し、Agentがpromptとして扱うようにします。
        if message.instruction:
            self._chat_history.append(ChatMessage(role=Role.USER, text=message.instruction))
        try:
            # 参加者が呼び出し可能なBaseAgentでない場合はno-opレスポンスを返します。
            from agent_framework import BaseAgent as _AF_AgentBase  # 循環を避けるためのローカルimport

            if not isinstance(self._agent, _AF_AgentBase):
                response = ChatMessage(
                    role=Role.ASSISTANT,
                    text=f"{self._agent_id} is a workflow executor and cannot be invoked directly.",
                    author_name=self._agent_id,
                )
                self._chat_history.append(response)
                await self._emit_agent_message_event(context, response)
            else:
                # Agentを呼び出します。
                response = await self._invoke_agent(context)
                self._chat_history.append(response)

            # レスポンスをオーケストレーターに送信します。
            await context.send_message(_MagenticResponseMessage(body=response))

        except Exception as e:
            logger.warning("Agent %s invoke failed: %s", self._agent_id, e)
            # フォールバックレスポンス。
            response = ChatMessage(
                role=Role.ASSISTANT,
                text=f"Agent {self._agent_id}: Error processing request - {str(e)[:100]}",
            )
            self._chat_history.append(response)
            await self._emit_agent_message_event(context, response)
            await context.send_message(_MagenticResponseMessage(body=response))

    def reset(self) -> None:
        """Agentの内部チャット履歴をリセットします（内部操作）。"""
        logger.debug("Agent %s: Resetting chat history", self._agent_id)
        self._chat_history.clear()
        self._state_restored = True

    async def _emit_agent_delta_event(
        self,
        ctx: WorkflowContext[Any, Any],
        update: AgentRunResponseUpdate,
    ) -> None:
        contents = list(getattr(update, "contents", []) or [])
        chunk = getattr(update, "text", None)
        if not chunk:
            chunk = "".join(getattr(item, "text", "") for item in contents if hasattr(item, "text"))
        if chunk:
            await ctx.add_event(
                MagenticAgentDeltaEvent(
                    agent_id=self._agent_id,
                    text=chunk or None,
                    role=getattr(update, "role", None),
                )
            )
        for item in contents:
            if isinstance(item, FunctionCallContent):
                await ctx.add_event(
                    MagenticAgentDeltaEvent(
                        agent_id=self._agent_id,
                        function_call_id=getattr(item, "call_id", None),
                        function_call_name=getattr(item, "name", None),
                        function_call_arguments=getattr(item, "arguments", None),
                        role=getattr(update, "role", None),
                    )
                )
            elif isinstance(item, FunctionResultContent):
                await ctx.add_event(
                    MagenticAgentDeltaEvent(
                        agent_id=self._agent_id,
                        function_result_id=getattr(item, "call_id", None),
                        function_result=getattr(item, "result", None),
                        role=getattr(update, "role", None),
                    )
                )

    async def _emit_agent_message_event(
        self,
        ctx: WorkflowContext[Any, Any],
        message: ChatMessage,
    ) -> None:
        await ctx.add_event(MagenticAgentMessageEvent(agent_id=self._agent_id, message=message))

    async def _invoke_agent(
        self,
        ctx: WorkflowContext[_MagenticResponseMessage, AgentRunResponse],
    ) -> ChatMessage:
        """ラップされたAgentを呼び出し、レスポンスを返します。"""
        logger.debug(f"Agent {self._agent_id}: Running with {len(self._chat_history)} messages")

        updates: list[AgentRunResponseUpdate] = []
        # この呼び出し時、ラップされた参加者はBaseAgentであることが保証されています。
        agent = cast("AgentProtocol", self._agent)
        async for update in agent.run_stream(messages=self._chat_history):  # type: ignore[attr-defined]
            updates.append(update)
            await self._emit_agent_delta_event(ctx, update)

        run_result: AgentRunResponse = AgentRunResponse.from_agent_run_response_updates(updates)

        messages: list[ChatMessage] | None = None
        with contextlib.suppress(Exception):
            messages = list(run_result.messages)  # type: ignore[assignment]
        if messages and len(messages) > 0:
            last: ChatMessage = messages[-1]
            author = last.author_name or self._agent_id
            role: Role = last.role if last.role else Role.ASSISTANT
            text = last.text or str(last)
            msg = ChatMessage(role=role, text=text, author_name=author)
            await self._emit_agent_message_event(ctx, msg)
            return msg

        msg = ChatMessage(
            role=Role.ASSISTANT,
            text=f"Agent {self._agent_id}: No output produced",
            author_name=self._agent_id,
        )
        await self._emit_agent_message_event(ctx, msg)
        return msg


# endregion Magentic Executors region Magentic Workflow Builder


class MagenticBuilder:
    """Magentic OneのマルチAgentオーケストレーションworkflowを作成するためのフルーエントビルダー。

    Magentic One workflowはLLM搭載のマネージャーを使い、複数のAgentを動的なタスク計画、進捗追跡、適応的再計画で調整します。マネージャーは計画を作成し、Agentを選択し、進捗を監視し、再計画や完了の判断を行います。

    ビルダーは参加者、マネージャー、オプションの計画レビュー、チェックポイント、イベントコールバックの設定のためのフルーエントAPIを提供します。

    Usage:

    .. code-block:: python

    from agent_framework import MagenticBuilder, StandardMagenticManager
    from azure.ai.projects.aio import AIProjectClient

    # LLMクライアント付きマネージャーを作成
    project_client = AIProjectClient.from_connection_string(...)
    chat_client = project_client.inference.get_chat_completions_client()

    # Agent付きMagentic workflowを構築
    workflow = (
        MagenticBuilder()
        .participants(researcher=research_agent, writer=writing_agent, coder=coding_agent)
        .with_standard_manager(chat_client=chat_client, max_round_count=20, max_stall_count=3)
        .with_plan_review(enable=True)
        .with_checkpointing(checkpoint_storage)
        .build()
    )

    # workflowを実行
    async for message in workflow.run("Research and write article about AI agents"):
        print(message.text)

    カスタムマネージャーの場合:

    .. code-block:: python

    # カスタムマネージャーのサブクラスを作成
    class MyCustomManager(MagenticManagerBase):
        async def plan(self, context: MagenticContext) -> ChatMessage:
            # カスタム計画ロジック
            ...


    manager = MyCustomManager()
    workflow = MagenticBuilder().participants(agent1=agent1, agent2=agent2).with_standard_manager(manager).build()

    See Also:
    - :class:`MagenticManagerBase`: カスタムマネージャーの基底クラス
    - :class:`StandardMagenticManager`: デフォルトのLLM搭載マネージャー
    - :class:`MagenticContext`: マネージャーメソッドに渡されるContextオブジェクト
    - :class:`MagenticEvent`: workflowイベントの基底クラス

    """

    def __init__(self) -> None:
        self._participants: dict[str, AgentProtocol | Executor] = {}
        self._manager: MagenticManagerBase | None = None
        self._enable_plan_review: bool = False
        self._checkpoint_storage: CheckpointStorage | None = None

    def participants(self, **participants: AgentProtocol | Executor) -> Self:
        """Add participant agents or executors to the Magentic workflow.

        Participants are the agents that will execute tasks under the manager's direction.
        Each participant should have distinct capabilities that complement the team. The
        manager will select which participant to invoke based on the current plan and
        progress state.

        Args:
            **participants: Named agents or executors to add to the workflow. Names should
                be descriptive of the agent's role (e.g., researcher=research_agent).
                Accepts BaseAgent instances or custom Executor implementations.

        Returns:
            Self for method chaining

        Usage:

        .. code-block:: python

            workflow = (
                MagenticBuilder()
                .participants(
                    researcher=research_agent, writer=writing_agent, coder=coding_agent, reviewer=review_agent
                )
                .with_standard_manager(chat_client=client)
                .build()
            )

        Notes:
            - Participant names become part of the manager's context for selection
            - Agent descriptions (if available) are extracted and provided to the manager
            - Can be called multiple times to add participants incrementally
        """
        self._participants.update(participants)
        return self

    def with_plan_review(self, enable: bool = True) -> "MagenticBuilder":
        """Enable or disable human-in-the-loop plan review before task execution.

        When enabled, the workflow will pause after the manager generates the initial
        plan and emit a _MagenticPlanReviewRequest event. A human reviewer can then
        approve, request revisions, or reject the plan. The workflow continues only
        after approval.

        This is useful for:
        - High-stakes tasks requiring human oversight
        - Validating the manager's understanding of requirements
        - Catching hallucinations or unrealistic plans early
        - Educational scenarios where learners review AI planning

        Args:
            enable: Whether to require plan review (default True)

        Returns:
            Self for method chaining

        Usage:

        .. code-block:: python

            workflow = (
                MagenticBuilder()
                .participants(agent1=agent1)
                .with_standard_manager(chat_client=client)
                .with_plan_review(enable=True)
                .build()
            )

            # During execution, handle plan review
            async for event in workflow.run_stream("task"):
                if isinstance(event, _MagenticPlanReviewRequest):
                    # Review plan and respond
                    reply = _MagenticPlanReviewReply(decision=MagenticPlanReviewDecision.APPROVE)
                    await workflow.send(reply)

        See Also:
            - :class:`_MagenticPlanReviewRequest`: Event emitted for review
            - :class:`_MagenticPlanReviewReply`: Response to send back
            - :class:`MagenticPlanReviewDecision`: Approve/Revise/Reject options
        """
        self._enable_plan_review = enable
        return self

    def with_checkpointing(self, checkpoint_storage: CheckpointStorage) -> "MagenticBuilder":
        """Enable workflow state persistence using the provided checkpoint storage.

        Checkpointing allows workflows to be paused, resumed across process restarts,
        or recovered after failures. The entire workflow state including conversation
        history, task ledgers, and progress is persisted at key points.

        Args:
            checkpoint_storage: Storage backend for checkpoints (e.g., InMemoryCheckpointStorage,
                FileCheckpointStorage, or custom implementations)

        Returns:
            Self for method chaining

        Usage:

        .. code-block:: python

            from agent_framework import InMemoryCheckpointStorage

            storage = InMemoryCheckpointStorage()
            workflow = (
                MagenticBuilder()
                .participants(agent1=agent1)
                .with_standard_manager(chat_client=client)
                .with_checkpointing(storage)
                .build()
            )

            # First run
            thread_id = "task-123"
            async for msg in workflow.run("task", thread_id=thread_id):
                print(msg.text)

            # Resume from checkpoint
            async for msg in workflow.run("continue", thread_id=thread_id):
                print(msg.text)

        Notes:
            - Checkpoints are created after each significant state transition
            - Thread ID must be consistent across runs to resume properly
            - Storage implementations may have different persistence guarantees
        """
        self._checkpoint_storage = checkpoint_storage
        return self

    def with_standard_manager(
        self,
        manager: MagenticManagerBase | None = None,
        *,
        # Constructor args for StandardMagenticManager when manager is not provided
        chat_client: ChatClientProtocol | None = None,
        task_ledger: _MagenticTaskLedger | None = None,
        instructions: str | None = None,
        # Prompt overrides
        task_ledger_facts_prompt: str | None = None,
        task_ledger_plan_prompt: str | None = None,
        task_ledger_full_prompt: str | None = None,
        task_ledger_facts_update_prompt: str | None = None,
        task_ledger_plan_update_prompt: str | None = None,
        progress_ledger_prompt: str | None = None,
        final_answer_prompt: str | None = None,
        # Limits
        max_stall_count: int = 3,
        max_reset_count: int | None = None,
        max_round_count: int | None = None,
    ) -> Self:
        """Configure the workflow manager for task planning and agent coordination.

        The manager is responsible for creating plans, selecting agents, tracking progress,
        and deciding when to replan or complete. This method supports two usage patterns:

        1. **Provide existing manager**: Pass a pre-configured manager instance (custom
           or standard) for full control over behavior
        2. **Auto-create standard manager**: Pass chat_client and options to automatically
           create a StandardMagenticManager with specified configuration

        Args:
            manager: Pre-configured manager instance (StandardMagenticManager or custom
                MagenticManagerBase subclass). If provided, all other arguments are ignored.
            chat_client: LLM chat client for generating plans and decisions. Required if
                manager is not provided.
            task_ledger: Optional custom task ledger implementation for specialized
                prompting or structured output requirements
            instructions: System instructions prepended to all manager prompts to guide
                behavior and set expectations
            task_ledger_facts_prompt: Custom prompt template for extracting facts from
                task description
            task_ledger_plan_prompt: Custom prompt template for generating initial plan
            task_ledger_full_prompt: Custom prompt template for complete task ledger
                (facts + plan combined)
            task_ledger_facts_update_prompt: Custom prompt template for updating facts
                based on agent progress
            task_ledger_plan_update_prompt: Custom prompt template for replanning when
                needed
            progress_ledger_prompt: Custom prompt template for assessing progress and
                determining next actions
            final_answer_prompt: Custom prompt template for synthesizing final response
                when task is complete
            max_stall_count: Maximum consecutive rounds without progress before triggering
                replan (default 3). Set to 0 to disable stall detection.
            max_reset_count: Maximum number of complete resets allowed before failing.
                None means unlimited resets.
            max_round_count: Maximum total coordination rounds before stopping with
                partial result. None means unlimited rounds.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If manager is None and chat_client is also None

        Usage with auto-created manager:

        .. code-block:: python

            from azure.ai.projects.aio import AIProjectClient

            project_client = AIProjectClient.from_connection_string(...)
            chat_client = project_client.inference.get_chat_completions_client()

            workflow = (
                MagenticBuilder()
                .participants(agent1=agent1, agent2=agent2)
                .with_standard_manager(
                    chat_client=chat_client,
                    max_round_count=20,
                    max_stall_count=3,
                    instructions="Be concise and focus on accuracy",
                )
                .build()
            )

        Usage with custom manager:

        .. code-block:: python

            class MyManager(MagenticManagerBase):
                async def plan(self, context: MagenticContext) -> ChatMessage:
                    # Custom planning logic
                    return ChatMessage(role=Role.ASSISTANT, text="...")


            manager = MyManager()
            workflow = MagenticBuilder().participants(agent1=agent1).with_standard_manager(manager).build()

        Usage with prompt customization:

        .. code-block:: python

            workflow = (
                MagenticBuilder()
                .participants(coder=coder_agent, reviewer=reviewer_agent)
                .with_standard_manager(
                    chat_client=chat_client,
                    task_ledger_plan_prompt="Create a detailed step-by-step plan...",
                    progress_ledger_prompt="Assess progress and decide next action...",
                    max_stall_count=2,
                )
                .build()
            )

        Notes:
            - StandardMagenticManager uses structured LLM calls for all decisions
            - Custom managers can implement alternative selection strategies
            - Prompt templates support Jinja2-style variable substitution
            - Stall detection helps prevent infinite loops in stuck scenarios
        """
        if manager is not None:
            self._manager = manager
            return self

        if chat_client is None:
            raise ValueError(
                "chat_client is required when manager is not provided: with_standard_manager(chat_client=...)"
            )

        self._manager = StandardMagenticManager(
            chat_client=chat_client,
            task_ledger=task_ledger,
            instructions=instructions,
            task_ledger_facts_prompt=task_ledger_facts_prompt,
            task_ledger_plan_prompt=task_ledger_plan_prompt,
            task_ledger_full_prompt=task_ledger_full_prompt,
            task_ledger_facts_update_prompt=task_ledger_facts_update_prompt,
            task_ledger_plan_update_prompt=task_ledger_plan_update_prompt,
            progress_ledger_prompt=progress_ledger_prompt,
            final_answer_prompt=final_answer_prompt,
            max_stall_count=max_stall_count,
            max_reset_count=max_reset_count,
            max_round_count=max_round_count,
        )
        return self

    def build(self) -> Workflow:
        """Build a Magentic workflow with the orchestrator and all agent executors."""
        if not self._participants:
            raise ValueError("No participants added to Magentic workflow")

        if self._manager is None:
            raise ValueError("No manager configured. Call with_standard_manager(...) before build().")

        logger.info("Building Magentic workflow with %d participants", len(self._participants))

        # Create participant descriptions
        participant_descriptions: dict[str, str] = {}
        for name, participant in self._participants.items():
            fallback = f"Executor {name}" if isinstance(participant, Executor) else f"Agent {name}"
            participant_descriptions[name] = participant_description(participant, fallback)

        # Type narrowing: we already checked self._manager is not None above
        manager: MagenticManagerBase = self._manager  # type: ignore[assignment]

        def _orchestrator_factory(wiring: _GroupChatConfig) -> Executor:
            return MagenticOrchestratorExecutor(
                manager=manager,
                participants=participant_descriptions,
                require_plan_signoff=self._enable_plan_review,
                executor_id="magentic_orchestrator",
            )

        def _participant_factory(
            spec: GroupChatParticipantSpec,
            wiring: _GroupChatConfig,
        ) -> _GroupChatParticipantPipeline:
            agent_executor = MagenticAgentExecutor(
                spec.participant,
                spec.name,
            )
            orchestrator = wiring.orchestrator
            if isinstance(orchestrator, MagenticOrchestratorExecutor):
                orchestrator.register_agent_executor(spec.name, agent_executor)
            return (agent_executor,)

        # Magentic provides its own orchestrator via custom factory, so no manager is needed
        group_builder = GroupChatBuilder(
            _orchestrator_factory=group_chat_orchestrator(_orchestrator_factory),
            _participant_factory=_participant_factory,
        ).participants(self._participants)

        if self._checkpoint_storage is not None:
            group_builder = group_builder.with_checkpointing(self._checkpoint_storage)

        if self._enable_plan_review:
            group_builder = group_builder.with_request_handler(
                lambda _wiring: RequestInfoExecutor(id="magentic_plan_review"),
                condition=lambda msg: isinstance(msg, _MagenticPlanReviewRequest),
            )

        return group_builder.build()

    def start_with_string(self, task: str) -> "MagenticWorkflow":
        """Build a Magentic workflow and return a wrapper with convenience methods for string tasks.

        Args:
            task: The task description as a string.

        Returns:
            A MagenticWorkflow wrapper that provides convenience methods for starting with strings.
        """
        return MagenticWorkflow(self.build(), task)

    def start_with_message(self, task: ChatMessage) -> "MagenticWorkflow":
        """Build a Magentic workflow and return a wrapper with convenience methods for ChatMessage tasks.

        Args:
            task: The task as a ChatMessage.

        Returns:
            A MagenticWorkflow wrapper that provides convenience methods.
        """
        return MagenticWorkflow(self.build(), task.text)

    def start_with(self, task: str | ChatMessage) -> "MagenticWorkflow":
        """Build a Magentic workflow and return a wrapper with convenience methods.

        Args:
            task: The task description as a string or ChatMessage.

        Returns:
            A MagenticWorkflow wrapper that provides convenience methods.
        """
        if isinstance(task, str):
            return self.start_with_string(task)
        return self.start_with_message(task)


# endregion Magentic Workflow Builder


# region Magentic Workflow


class MagenticWorkflow:
    """A wrapper around the base Workflow that provides convenience methods for Magentic workflows."""

    def __init__(self, workflow: Workflow, task_text: str | None = None):
        self._workflow = workflow
        self._task_text = task_text

    @property
    def workflow(self) -> Workflow:
        """Access the underlying workflow."""
        return self._workflow

    async def run_streaming_with_string(self, task_text: str) -> AsyncIterable[WorkflowEvent]:
        """Run the workflow with a task string.

        Args:
            task_text: The task description as a string.

        Yields:
            WorkflowEvent: The events generated during the workflow execution.
        """
        start_message = _MagenticStartMessage.from_string(task_text)
        async for event in self._workflow.run_stream(start_message):
            yield event

    async def run_streaming_with_message(self, task_message: ChatMessage) -> AsyncIterable[WorkflowEvent]:
        """Run the workflow with a ChatMessage.

        Args:
            task_message: The task as a ChatMessage.

        Yields:
            WorkflowEvent: The events generated during the workflow execution.
        """
        start_message = _MagenticStartMessage(task_message)
        async for event in self._workflow.run_stream(start_message):
            yield event

    async def run_stream(self, message: Any | None = None) -> AsyncIterable[WorkflowEvent]:
        """Run the workflow with either a message object or the preset task string.

        Args:
            message: The message to send. If None and task_text was provided during construction,
                    uses the preset task string.

        Yields:
            WorkflowEvent: The events generated during the workflow execution.
        """
        if message is None:
            if self._task_text is None:
                raise ValueError("No message provided and no preset task text available")
            message = _MagenticStartMessage.from_string(self._task_text)
        elif isinstance(message, str):
            message = _MagenticStartMessage.from_string(message)
        elif isinstance(message, (ChatMessage, list)):
            message = _MagenticStartMessage(message)  # type: ignore[arg-type]

        async for event in self._workflow.run_stream(message):
            yield event

    async def _validate_checkpoint_participants(
        self,
        checkpoint_id: str,
        checkpoint_storage: CheckpointStorage | None = None,
    ) -> None:
        """Ensure participant roster matches the checkpoint before attempting restoration."""
        orchestrator = next(
            (
                executor
                for executor in self._workflow.executors.values()
                if isinstance(executor, MagenticOrchestratorExecutor)
            ),
            None,
        )
        if orchestrator is None:
            return

        expected = getattr(orchestrator, "_participants", None)
        if not expected:
            return

        checkpoint: WorkflowCheckpoint | None = None
        if checkpoint_storage is not None:
            try:
                checkpoint = await checkpoint_storage.load_checkpoint(checkpoint_id)
            except Exception:  # pragma: no cover - best effort
                checkpoint = None

        if checkpoint is None:
            runner_context = getattr(self._workflow, "_runner_context", None)
            has_checkpointing = getattr(runner_context, "has_checkpointing", None)
            load_checkpoint = getattr(runner_context, "load_checkpoint", None)
            try:
                if callable(has_checkpointing) and has_checkpointing() and callable(load_checkpoint):
                    loaded_checkpoint = await load_checkpoint(checkpoint_id)  # type: ignore[misc]
                    if loaded_checkpoint is not None:
                        checkpoint = cast(WorkflowCheckpoint, loaded_checkpoint)
            except Exception:  # pragma: no cover - best effort
                checkpoint = None

        if checkpoint is None:
            return

        # At this point, checkpoint is guaranteed to be WorkflowCheckpoint
        executor_states: dict[str, Any] = checkpoint.shared_state.get(EXECUTOR_STATE_KEY, {})
        orchestrator_id = getattr(orchestrator, "id", "")
        orchestrator_state = executor_states.get(orchestrator_id)
        if orchestrator_state is None:
            orchestrator_state = executor_states.get("magentic_orchestrator")

        if not isinstance(orchestrator_state, dict):
            return

        context_payload = orchestrator_state.get("magentic_context")
        if not isinstance(context_payload, dict):
            return

        context_dict = cast(dict[str, Any], context_payload)
        restored_participants = context_dict.get("participant_descriptions")
        if not isinstance(restored_participants, dict):
            return

        participants_dict = cast(dict[str, str], restored_participants)
        restored_names: set[str] = set(participants_dict.keys())
        expected_names = set(expected.keys())

        if restored_names == expected_names:
            return

        missing = ", ".join(sorted(expected_names - restored_names)) or "none"
        unexpected = ", ".join(sorted(restored_names - expected_names)) or "none"
        raise RuntimeError(
            "Magentic checkpoint restore failed: participant names do not match the checkpoint. "
            "Ensure MagenticBuilder.participants keys remain stable across runs. "
            f"Missing names: {missing}; unexpected names: {unexpected}."
        )

    async def run_stream_from_checkpoint(
        self,
        checkpoint_id: str,
        checkpoint_storage: CheckpointStorage | None = None,
        responses: dict[str, Any] | None = None,
    ) -> AsyncIterable[WorkflowEvent]:
        """Resume orchestration from a checkpoint and stream resulting events."""
        await self._validate_checkpoint_participants(checkpoint_id, checkpoint_storage)
        async for event in self._workflow.run_stream_from_checkpoint(checkpoint_id, checkpoint_storage, responses):
            yield event

    async def run_with_string(self, task_text: str) -> WorkflowRunResult:
        """タスク文字列でワークフローを実行し、すべてのイベントを返します。

        Args:
            task_text: タスクの説明を文字列で指定します。

        Returns:
            WorkflowRunResult: ワークフロー実行中に生成されたすべてのイベント。

        """
        events: list[WorkflowEvent] = []
        async for event in self.run_streaming_with_string(task_text):
            events.append(event)
        return WorkflowRunResult(events)

    async def run_with_message(self, task_message: ChatMessage) -> WorkflowRunResult:
        """ChatMessageでワークフローを実行し、すべてのイベントを返します。

        Args:
            task_message: ChatMessageとしてのタスク。

        Returns:
            WorkflowRunResult: ワークフロー実行中に生成されたすべてのイベント。

        """
        events: list[WorkflowEvent] = []
        async for event in self.run_streaming_with_message(task_message):
            events.append(event)
        return WorkflowRunResult(events)

    async def run(self, message: Any | None = None) -> WorkflowRunResult:
        """ワークフローを実行し、すべてのイベントを返します。

        Args:
            message: 送信するメッセージ。Noneの場合、かつ構築時にtask_textが提供されていれば、
                    事前設定されたタスク文字列を使用します。

        Returns:
            WorkflowRunResult: ワークフロー実行中に生成されたすべてのイベント。

        """
        events: list[WorkflowEvent] = []
        async for event in self.run_stream(message):
            events.append(event)
        return WorkflowRunResult(events)

    async def run_from_checkpoint(
        self,
        checkpoint_id: str,
        checkpoint_storage: CheckpointStorage | None = None,
        responses: dict[str, Any] | None = None,
    ) -> WorkflowRunResult:
        """チェックポイントからオーケストレーションを再開し、結果として得られるすべてのイベントを収集します。"""
        events: list[WorkflowEvent] = []
        async for event in self.run_stream_from_checkpoint(checkpoint_id, checkpoint_storage, responses):
            events.append(event)
        return WorkflowRunResult(events)

    async def send_responses_streaming(self, responses: dict[str, Any]) -> AsyncIterable[WorkflowEvent]:
        """保留中のリクエストにレスポンスを転送し、結果のイベントをストリームします。

        これは基盤となるWorkflowの実装に委譲されます。

        """
        async for event in self._workflow.send_responses_streaming(responses):
            yield event

    async def send_responses(self, responses: dict[str, Any]) -> WorkflowRunResult:
        """保留中のリクエストにレスポンスを転送し、結果のすべてのイベントを返します。

        これは基盤となるWorkflowの実装に委譲されます。

        """
        return await self._workflow.send_responses(responses)

    def __getattr__(self, name: str) -> Any:
        """不明な属性は基盤となるworkflowに委譲します。"""
        return getattr(self._workflow, name)


# endregion Magentic Workflow カスタムプランレビューハンドラーを実装するユーザーが必要とする型の公開エイリアス
MagenticPlanReviewRequest = _MagenticPlanReviewRequest
MagenticPlanReviewReply = _MagenticPlanReviewReply
