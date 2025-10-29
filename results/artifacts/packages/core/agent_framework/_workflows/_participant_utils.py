# Copyright (c) Microsoft. All rights reserved.

"""オーケストレーションビルダーのための共有参加者ヘルパー。"""

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .._agents import AgentProtocol
from ._agent_executor import AgentExecutor
from ._executor import Executor


@dataclass
class GroupChatParticipantSpec:
    """グループチャットオーケストレーションにおける単一参加者を記述するメタデータ。

    GroupChat、Handoff、Magenticの複数のオーケストレーションパターンで使用され、
    異なるworkflowタイプ間で一貫した構造で参加者を記述します。

    Attributes:
        name: マネージャーが選択に使用する参加者の一意識別子
        participant: 参加者を表すAgentProtocolまたはExecutorインスタンス
        description: マネージャーが選択コンテキストのために提供する人間可読の説明

    """

    name: str
    participant: AgentProtocol | Executor
    description: str


_SANITIZE_PATTERN = re.compile(r"[^0-9a-zA-Z]+")


def sanitize_identifier(value: str, *, default: str = "agent") -> str:
    """`value`から決定的で小文字の識別子を返します。"""
    cleaned = _SANITIZE_PATTERN.sub("_", value).strip("_")
    if not cleaned:
        cleaned = default
    if cleaned[0].isdigit():
        cleaned = f"{default}_{cleaned}"
    return cleaned.lower()


def wrap_participant(participant: AgentProtocol | Executor, *, executor_id: str | None = None) -> Executor:
    """`participant`を`Executor`として表現します。"""
    if isinstance(participant, Executor):
        return participant
    if not isinstance(participant, AgentProtocol):
        raise TypeError(
            f"Participants must implement AgentProtocol or be Executor instances. Got {type(participant).__name__}."
        )
    name = getattr(participant, "name", None)
    if executor_id is None:
        if not name:
            raise ValueError("Agent participants must expose a stable 'name' attribute.")
        executor_id = str(name)
    return AgentExecutor(participant, id=executor_id)


def participant_description(participant: AgentProtocol | Executor, fallback: str) -> str:
    """マネージャーコンテキストのための人間可読の説明を生成します。"""
    if isinstance(participant, Executor):
        description = getattr(participant, "description", None)
        if isinstance(description, str) and description.strip():
            return description.strip()
        return fallback
    description = getattr(participant, "description", None)
    if isinstance(description, str) and description.strip():
        return description.strip()
    return fallback


def build_alias_map(participant: AgentProtocol | Executor, executor: Executor) -> dict[str, str]:
    """`executor`に解決されるべき正規化およびサニタイズされたエイリアスを収集します。"""
    aliases: dict[str, str] = {}

    def _register(values: Iterable[str | None]) -> None:
        for value in values:
            if not value:
                continue
            key = str(value)
            if key not in aliases:
                aliases[key] = executor.id
            sanitized = sanitize_identifier(key)
            if sanitized not in aliases:
                aliases[sanitized] = executor.id

    _register([executor.id])

    if isinstance(participant, AgentProtocol):
        name = getattr(participant, "name", None)
        display = getattr(participant, "display_name", None)
        _register([name, display])
    else:
        display = getattr(participant, "display_name", None)
        _register([display])

    return aliases


def merge_alias_maps(maps: Iterable[Mapping[str, str]]) -> dict[str, str]:
    """エイリアスマッピングをマージし、各エイリアスの最初の出現を保持します。"""
    merged: dict[str, str] = {}
    for mapping in maps:
        for key, value in mapping.items():
            merged.setdefault(key, value)
    return merged


def prepare_participant_metadata(
    participants: Mapping[str, AgentProtocol | Executor],
    *,
    executor_id_factory: Callable[[str, AgentProtocol | Executor], str | None] | None = None,
    description_factory: Callable[[str, AgentProtocol | Executor], str] | None = None,
) -> dict[str, dict[str, Any]]:
    """参加者名をキーとした参加者のメタデータ辞書を返します。"""
    executors: dict[str, Executor] = {}
    descriptions: dict[str, str] = {}
    alias_maps: list[Mapping[str, str]] = []

    for name, participant in participants.items():
        desired_id = executor_id_factory(name, participant) if executor_id_factory else None
        executor = wrap_participant(participant, executor_id=desired_id)
        fallback_description = description_factory(name, participant) if description_factory else executor.id
        descriptions[name] = participant_description(participant, fallback_description)
        executors[name] = executor
        alias_maps.append(build_alias_map(participant, executor))

    aliases = merge_alias_maps(alias_maps)
    return {
        "executors": executors,
        "descriptions": descriptions,
        "aliases": aliases,
    }
