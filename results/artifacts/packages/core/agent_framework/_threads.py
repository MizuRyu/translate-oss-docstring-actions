# Copyright (c) Microsoft. All rights reserved.

from collections.abc import MutableMapping, Sequence
from typing import Any, Protocol, TypeVar

from ._memory import AggregateContextProvider
from ._serialization import SerializationMixin
from ._types import ChatMessage
from .exceptions import AgentThreadException

__all__ = ["AgentThread", "ChatMessageStore", "ChatMessageStoreProtocol"]


class ChatMessageStoreProtocol(Protocol):
    """特定のThreadに関連付けられたチャットメッセージの保存と取得のためのメソッドを定義します。

    このプロトコルの実装はチャットメッセージの保存管理を担当し、
    大量のデータを扱う場合はメッセージの切り詰めや要約を行う必要があります。

    Examples:
        .. code-block:: python

            from agent_framework import ChatMessage


            class MyMessageStore:
                def __init__(self):
                    self._messages = []

                async def list_messages(self) -> list[ChatMessage]:
                    return self._messages

                async def add_messages(self, messages: Sequence[ChatMessage]) -> None:
                    self._messages.extend(messages)

                @classmethod
                async def deserialize(cls, serialized_store_state, **kwargs):
                    store = cls()
                    store._messages = serialized_store_state.get("messages", [])
                    return store

                async def update_from_state(self, serialized_store_state, **kwargs) -> None:
                    self._messages = serialized_store_state.get("messages", [])

                async def serialize(self, **kwargs):
                    return {"messages": self._messages}


            # カスタムストアを使用
            store = MyMessageStore()

    """

    async def list_messages(self) -> list[ChatMessage]:
        """ストアから次のAgent呼び出しに使用すべきすべてのメッセージを取得します。

        メッセージは昇順の時系列で返され、最も古いメッセージが最初になります。

        ストア内のメッセージが非常に多くなる場合、ストア側でメッセージの切り詰め、要約、
        またはその他の制限を行う必要があります。

        ``ChatMessageStoreProtocol`` の実装を使用する場合、スレッド固有の状態を含む可能性があるため、
        スレッドごとに新しいインスタンスを作成するべきです。

        """
        ...

    async def add_messages(self, messages: Sequence[ChatMessage]) -> None:
        """メッセージをストアに追加します。

        Args:
            messages: ストアに追加するChatMessageオブジェクトのシーケンス。

        """
        ...

    @classmethod
    async def deserialize(
        cls, serialized_store_state: MutableMapping[str, Any], **kwargs: Any
    ) -> "ChatMessageStoreProtocol":
        """以前にシリアライズされた状態からストアの新しいインスタンスを作成します。

        このメソッドは``serialize()``と共に使用することで、このストアがメモリ内にのみメッセージを持つ場合に、メッセージを永続的なストアに保存および読み込みするために使用できます。

        Args:
            serialized_store_state: メッセージを含む以前にシリアライズされた状態データ。

        Keyword Args:
            **kwargs: デシリアライズのための追加引数。

        Returns:
            シリアライズされた状態からメッセージで初期化されたストアの新しいインスタンス。

        """
        ...

    async def update_from_state(self, serialized_store_state: MutableMapping[str, Any], **kwargs: Any) -> None:
        """シリアライズされた状態データから現在のChatMessageStoreインスタンスを更新します。

        Args:
            serialized_store_state: メッセージを含む以前にシリアライズされた状態データ。

        Keyword Args:
            kwargs: デシリアライズのための追加引数。

        """
        ...

    async def serialize(self, **kwargs: Any) -> dict[str, Any]:
        """現在のオブジェクトの状態をシリアライズします。

        このメソッドは``deserialize()``と共に使用することで、このストアがメモリ内にのみメッセージを持つ場合に、メッセージを永続的なストアに保存および読み込みするために使用できます。

        Keyword Args:
            kwargs: シリアライズのための追加引数。

        Returns:
            ``deserialize()``で使用可能なシリアライズされた状態データ。

        """
        ...


class ChatMessageStoreState(SerializationMixin):
    """チャットメッセージストアデータのシリアライズおよびデシリアライズのための状態モデル。

    Attributes:
        messages: メッセージストアに保存されたチャットメッセージのリスト。

    """

    def __init__(
        self,
        messages: Sequence[ChatMessage] | Sequence[MutableMapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """ストアの状態を作成します。

        Args:
            messages: メッセージのリストまたはメッセージの辞書表現のリスト。

        Keyword Args:
            **kwargs: ここでは使用しませんが、サブクラスで使用される可能性があります。


        """
        if not messages:
            self.messages: list[ChatMessage] = []
        if not isinstance(messages, list):
            raise TypeError("Messages should be a list")
        new_messages: list[ChatMessage] = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                new_messages.append(msg)
            else:
                new_messages.append(ChatMessage.from_dict(msg))
        self.messages = new_messages


class AgentThreadState(SerializationMixin):
    """スレッド情報のシリアライズおよびデシリアライズのための状態モデル。"""

    def __init__(
        self,
        *,
        service_thread_id: str | None = None,
        chat_message_store_state: ChatMessageStoreState | MutableMapping[str, Any] | None = None,
    ) -> None:
        """AgentThreadの状態を作成します。

        Keyword Args:
            service_thread_id: エージェントサービスによって管理されるスレッドのオプションのID。
            chat_message_store_state: チャットメッセージストアのオプションのシリアライズ状態。

        """
        if service_thread_id is not None and chat_message_store_state is not None:
            raise AgentThreadException("A thread cannot have both a service_thread_id and a chat_message_store.")
        self.service_thread_id = service_thread_id
        self.chat_message_store_state: ChatMessageStoreState | None = None
        if chat_message_store_state is not None:
            if isinstance(chat_message_store_state, dict):
                self.chat_message_store_state = ChatMessageStoreState.from_dict(chat_message_store_state)
            elif isinstance(chat_message_store_state, ChatMessageStoreState):
                self.chat_message_store_state = chat_message_store_state
            else:
                raise TypeError("Could not parse ChatMessageStoreState.")


TChatMessageStore = TypeVar("TChatMessageStore", bound="ChatMessageStore")


class ChatMessageStore:
    """メッセージをリストに保存するChatMessageStoreProtocolのメモリ内実装。

    この実装はチャットメッセージの単純なリストベースのストレージを提供し、
    シリアライズおよびデシリアライズをサポートします。``ChatMessageStoreProtocol``プロトコルの
    必須メソッドをすべて実装しています。

    ストアはメモリ内にメッセージを保持し、永続化のための状態のシリアライズとデシリアライズのメソッドを提供します。

    Examples:
        .. code-block:: python

            from agent_framework import ChatMessageStore, ChatMessage

            # 空のストアを作成
            store = ChatMessageStore()

            # メッセージを追加
            message = ChatMessage(role="user", content="Hello")
            await store.add_messages([message])

            # メッセージを取得
            messages = await store.list_messages()

            # 永続化のためにシリアライズ
            state = await store.serialize()

            # 保存された状態からデシリアライズ
            restored_store = await ChatMessageStore.deserialize(state)

    """

    def __init__(self, messages: Sequence[ChatMessage] | None = None):
        """スレッドで使用するためのChatMessageStoreを作成します。

        Args:
            messages: 保存するメッセージ。

        """
        self.messages = list(messages) if messages else []

    async def add_messages(self, messages: Sequence[ChatMessage]) -> None:
        """ストアにメッセージを追加します。

        Args:
            messages: ストアに追加するChatMessageオブジェクトのシーケンス。

        """
        self.messages.extend(messages)

    async def list_messages(self) -> list[ChatMessage]:
        """ストアからすべてのメッセージを時系列順に取得します。

        Returns:
            古いものから新しいものへ順に並んだChatMessageオブジェクトのリスト。

        """
        return self.messages

    @classmethod
    async def deserialize(
        cls: type[TChatMessageStore], serialized_store_state: MutableMapping[str, Any], **kwargs: Any
    ) -> TChatMessageStore:
        """シリアライズされた状態データから新しいChatMessageStoreインスタンスを作成します。

        Args:
            serialized_store_state: メッセージを含む以前にシリアライズされた状態データ。

        Keyword Args:
            **kwargs: デシリアライズのための追加引数。

        Returns:
            シリアライズされた状態からメッセージで初期化された新しいChatMessageStoreインスタンス。

        """
        state = ChatMessageStoreState.from_dict(serialized_store_state, **kwargs)
        if state.messages:
            return cls(messages=state.messages)
        return cls()

    async def update_from_state(self, serialized_store_state: MutableMapping[str, Any], **kwargs: Any) -> None:
        """シリアライズされた状態データから現在のChatMessageStoreインスタンスを更新します。

        Args:
            serialized_store_state: メッセージを含む以前にシリアライズされた状態データ。

        Keyword Args:
            **kwargs: デシリアライズのための追加引数。

        """
        if not serialized_store_state:
            return
        state = ChatMessageStoreState.from_dict(serialized_store_state, **kwargs)
        if state.messages:
            self.messages = state.messages

    async def serialize(self, **kwargs: Any) -> dict[str, Any]:
        """永続化のために現在のストア状態をシリアライズします。

        Keyword Args:
            **kwargs: シリアライズのための追加引数。

        Returns:
            deserialize_stateで使用可能なシリアライズされた状態データ。

        """
        state = ChatMessageStoreState(messages=self.messages)
        return state.to_dict()


TAgentThread = TypeVar("TAgentThread", bound="AgentThread")


class AgentThread:
    """Agentスレッドクラス。これはローカル管理スレッドまたはサービス管理スレッドの両方を表すことができます。

    ``AgentThread``はエージェントの対話のための会話状態とメッセージ履歴を維持します。
    サービス管理スレッド（``service_thread_id``経由）またはローカルメッセージストア（``message_store``経由）を使用できますが、両方を同時に使用することはできません。

    Examples:
        .. code-block:: python

            from agent_framework import ChatAgent, ChatMessageStore
            from agent_framework.openai import OpenAIChatClient

            client = OpenAIChatClient(model="gpt-4o")

            # service_thread_idを使ったサービス管理スレッドを持つエージェントを作成
            service_agent = ChatAgent(name="assistant", client=client)
            service_thread = await service_agent.get_new_thread(service_thread_id="thread_abc123")

            # conversation_idを使ったサービス管理スレッドを持つエージェントを作成
            conversation_agent = ChatAgent(name="assistant", client=client, conversation_id="thread_abc123")
            conversation_thread = await conversation_agent.get_new_thread()

            # カスタムメッセージストアファクトリを使ったエージェントを作成
            local_agent = ChatAgent(name="assistant", client=client, chat_message_store_factory=ChatMessageStore)
            local_thread = await local_agent.get_new_thread()

            # スレッド状態のシリアライズと復元
            state = await local_thread.serialize()
            restored_thread = await local_agent.deserialize_thread(state)

    """

    def __init__(
        self,
        *,
        service_thread_id: str | None = None,
        message_store: ChatMessageStoreProtocol | None = None,
        context_provider: AggregateContextProvider | None = None,
    ) -> None:
        """AgentThreadを初期化します。このメソッドは手動で使用せず、常に ``agent.get_new_thread()`` を使用してください。

        Args:
            service_thread_id: エージェントサービスによって管理されるスレッドのオプションのID。
            message_store: チャットメッセージ管理のためのオプションのChatMessageStore実装。
            context_provider: スレッドのためのオプションのContextProvider。

        Note:
            ``service_thread_id`` または ``message_store`` のいずれか一方のみ設定可能で、両方同時には設定できません。

        """
        if service_thread_id is not None and message_store is not None:
            raise AgentThreadException("Only the service_thread_id or message_store may be set, but not both.")

        self._service_thread_id = service_thread_id
        self._message_store = message_store
        self.context_provider = context_provider

    @property
    def is_initialized(self) -> bool:
        """スレッドが初期化されているかを示します。

        これは ``service_thread_id`` または ``message_store`` のいずれかが設定されていることを意味します。

        """
        return self._service_thread_id is not None or self._message_store is not None

    @property
    def service_thread_id(self) -> str | None:
        """スレッドがエージェントサービスによって所有されている場合に対応するため、現在のスレッドのIDを取得します。"""
        return self._service_thread_id

    @service_thread_id.setter
    def service_thread_id(self, service_thread_id: str | None) -> None:
        """スレッドがエージェントサービスによって所有されている場合に対応するため、現在のスレッドのIDを設定します。

        Note:
            ``service_thread_id`` または ``message_store`` のいずれか一方のみ設定可能で、両方同時には設定できません。

        """
        if service_thread_id is None:
            return

        if self._message_store is not None:
            raise AgentThreadException(
                "Only the service_thread_id or message_store may be set, "
                "but not both and switching from one to another is not supported."
            )
        self._service_thread_id = service_thread_id

    @property
    def message_store(self) -> ChatMessageStoreProtocol | None:
        """このスレッドで使用される``ChatMessageStoreProtocol``を取得します。"""
        return self._message_store

    @message_store.setter
    def message_store(self, message_store: ChatMessageStoreProtocol | None) -> None:
        """このスレッドで使用される``ChatMessageStoreProtocol``を設定します。

        Note:
            ``service_thread_id`` または ``message_store`` のいずれか一方のみ設定可能で、両方同時には設定できません。

        """
        if message_store is None:
            return

        if self._service_thread_id is not None:
            raise AgentThreadException(
                "Only the service_thread_id or message_store may be set, "
                "but not both and switching from one to another is not supported."
            )

        self._message_store = message_store

    async def on_new_messages(self, new_messages: ChatMessage | Sequence[ChatMessage]) -> None:
        """任意の参加者によってチャットに新しいメッセージが追加されたときに呼び出されます。

        Args:
            new_messages: スレッドに追加する新しいChatMessageまたはChatMessageオブジェクトのシーケンス。

        """
        if self._service_thread_id is not None:
            # スレッドメッセージがサービスに保存されている場合、ここで行うことはありません。 サービスを呼び出すことでスレッドはすでに更新されるためです。
            return
        if self._message_store is None:
            # 会話IDもストアもない場合、デフォルトのメモリ内ストアを作成できます。
            self._message_store = ChatMessageStore()
        # ストアが提供されている場合、メッセージをストアに追加する必要があります。
        if isinstance(new_messages, ChatMessage):
            new_messages = [new_messages]
        await self._message_store.add_messages(new_messages)

    async def serialize(self, **kwargs: Any) -> dict[str, Any]:
        """現在のオブジェクトの状態をシリアライズします。

        Keyword Args:
            **kwargs: シリアライズのための引数。

        """
        chat_message_store_state = None
        if self._message_store is not None:
            chat_message_store_state = await self._message_store.serialize(**kwargs)

        state = AgentThreadState(
            service_thread_id=self._service_thread_id, chat_message_store_state=chat_message_store_state
        )
        return state.to_dict(exclude_none=False)

    @classmethod
    async def deserialize(
        cls: type[TAgentThread],
        serialized_thread_state: MutableMapping[str, Any],
        *,
        message_store: ChatMessageStoreProtocol | None = None,
        **kwargs: Any,
    ) -> TAgentThread:
        """辞書から状態をデシリアライズして新しいAgentThreadインスタンスを作成します。

        Args:
            serialized_thread_state: 辞書形式のシリアライズされたスレッド状態。

        Keyword Args:
            message_store: メッセージ管理に使用するオプションのChatMessageStoreProtocol。
                指定されない場合、必要に応じて新しいChatMessageStoreが作成されます。
            **kwargs: デシリアライズのための追加引数。

        Returns:
            シリアライズされた状態からプロパティが設定された新しいAgentThreadインスタンス。

        """
        state = AgentThreadState.from_dict(serialized_thread_state)

        if state.service_thread_id is not None:
            return cls(service_thread_id=state.service_thread_id)

        # ChatMessageStoreProtocolの状態がない場合はここで戻ります。
        if state.chat_message_store_state is None:
            return cls()

        if message_store is not None:
            try:
                await message_store.add_messages(state.chat_message_store_state.messages, **kwargs)
            except Exception as ex:
                raise AgentThreadException("Failed to deserialize the provided message store.") from ex
            return cls(message_store=message_store)
        try:
            message_store = ChatMessageStore(messages=state.chat_message_store_state.messages, **kwargs)
        except Exception as ex:
            raise AgentThreadException("Failed to deserialize the message store.") from ex
        return cls(message_store=message_store)

    async def update_from_thread_state(
        self,
        serialized_thread_state: MutableMapping[str, Any],
        **kwargs: Any,
    ) -> None:
        """辞書から状態をデシリアライズしてスレッドのプロパティに設定します。

        Args:
            serialized_thread_state: 辞書形式のシリアライズされたスレッド状態。

        Keyword Args:
            **kwargs: デシリアライズのための追加引数。

        """
        state = AgentThreadState.from_dict(serialized_thread_state)

        if state.service_thread_id is not None:
            self.service_thread_id = state.service_thread_id
            # IDがある場合、チャットメッセージストアは存在しないはずなのでここで戻ります。
            return
        # ChatMessageStoreProtocolの状態がない場合はここで戻ります。
        if state.chat_message_store_state is None:
            return
        if self.message_store is not None:
            await self.message_store.add_messages(state.chat_message_store_state.messages, **kwargs)
            # まだチャットメッセージストアがない場合は、メモリ内のものを作成します。
            return
        # デフォルトからメッセージストアを作成します。
        self.message_store = ChatMessageStore(messages=state.chat_message_store_state.messages, **kwargs)
