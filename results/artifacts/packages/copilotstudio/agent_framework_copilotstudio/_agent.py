# Copyright (c) Microsoft. All rights reserved.

from collections.abc import AsyncIterable
from typing import Any, ClassVar

from agent_framework import (
    AgentMiddlewares,
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    AggregateContextProvider,
    BaseAgent,
    ChatMessage,
    ContextProvider,
    Role,
    TextContent,
)
from agent_framework._pydantic import AFBaseSettings
from agent_framework.exceptions import ServiceException, ServiceInitializationError
from microsoft_agents.copilotstudio.client import AgentType, ConnectionSettings, CopilotClient, PowerPlatformCloud
from pydantic import ValidationError

from ._acquire_token import acquire_token


class CopilotStudioSettings(AFBaseSettings):
    """Copilot Studio のモデル設定。

    設定はまず 'COPILOTSTUDIOAGENT__' プレフィックスの環境変数から読み込まれる。
    環境変数が見つからない場合は、エンコーディング 'utf-8' の .env ファイルから読み込まれる。
    .env ファイルにも設定が見つからない場合は設定は無視されるが、
    バリデーションで設定が欠落していることが通知される。

    Keyword Args:
        environmentid: Copilot Studio App を含む環境の環境ID。
            環境変数 COPILOTSTUDIOAGENT__ENVIRONMENTID で設定可能。
        schemaname: 使用する Copilot のエージェント識別子またはスキーマ名。
            環境変数 COPILOTSTUDIOAGENT__SCHEMANAME で設定可能。
        agentappid: ログインに使用する App Registration のアプリID。
            環境変数 COPILOTSTUDIOAGENT__AGENTAPPID で設定可能。
        tenantid: ログインに使用する App Registration のテナントID。
            環境変数 COPILOTSTUDIOAGENT__TENANTID で設定可能。
        env_file_path: 指定すると、そのパスの .env ファイルから設定を読み込む。
        env_file_encoding: .env ファイルのエンコーディング。デフォルトは 'utf-8'。

    Examples:
        .. code-block:: python

            from agent_framework_copilotstudio import CopilotStudioSettings

            # 環境変数を使う場合
            # COPILOTSTUDIOAGENT__ENVIRONMENTID=env-123 を設定
            # COPILOTSTUDIOAGENT__SCHEMANAME=my-agent を設定
            settings = CopilotStudioSettings()

            # またはパラメータを直接渡す場合
            settings = CopilotStudioSettings(environmentid="env-123", schemaname="my-agent")

            # または .env ファイルから読み込む場合
            settings = CopilotStudioSettings(env_file_path="path/to/.env")
    """

    env_prefix: ClassVar[str] = "COPILOTSTUDIOAGENT__"

    environmentid: str | None = None
    schemaname: str | None = None
    agentappid: str | None = None
    tenantid: str | None = None


class CopilotStudioAgent(BaseAgent):
    """Copilot Studio の Agent。"""

    def __init__(
        self,
        client: CopilotClient | None = None,
        settings: ConnectionSettings | None = None,
        *,
        id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        context_providers: ContextProvider | list[ContextProvider] | AggregateContextProvider | None = None,
        middleware: AgentMiddlewares | list[AgentMiddlewares] | None = None,
        environment_id: str | None = None,
        agent_identifier: str | None = None,
        client_id: str | None = None,
        tenant_id: str | None = None,
        token: str | None = None,
        cloud: PowerPlatformCloud | None = None,
        agent_type: AgentType | None = None,
        custom_power_platform_cloud: str | None = None,
        username: str | None = None,
        token_cache: Any | None = None,
        scopes: list[str] | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        """Copilot Studio Agent を初期化する。

        Args:
            client: オプションの事前設定済み CopilotClient インスタンス。指定しない場合は他のパラメータから新規作成される。
            settings: オプションの事前設定済み ConnectionSettings。指定しない場合は他のパラメータから作成される。

        Keyword Args:
            id: CopilotAgent の id
            name: CopilotAgent の名前
            description: CopilotAgent の説明
            context_providers: Copilot Agent が使用する Context Providers。
            middleware: エージェントが使用する Middleware。
            environment_id: Copilot Studio アプリを含む Power Platform 環境の環境ID。
                環境変数 COPILOTSTUDIOAGENT__ENVIRONMENTID でも設定可能。
            agent_identifier: 使用する Copilot のエージェント識別子またはスキーマ名。
                環境変数 COPILOTSTUDIOAGENT__SCHEMANAME でも設定可能。
            client_id: 認証に使用する App Registration のアプリID。
                環境変数 COPILOTSTUDIOAGENT__AGENTAPPID でも設定可能。
            tenant_id: 認証に使用する App Registration のテナントID。
                環境変数 COPILOTSTUDIOAGENT__TENANTID でも設定可能。
            token: オプションの事前取得済み認証トークン。指定しない場合は MSAL を使って取得を試みる。
            cloud: 使用する Power Platform のクラウド（Public、GCC など）。
            agent_type: Copilot Studio agent のタイプ（Copilot、Agent など）。
            custom_power_platform_cloud: カスタム環境を使う場合のカスタム Power Platform クラウド URL。
            username: トークン取得用のオプションのユーザー名。
            token_cache: 認証トークンを保存するオプションのトークンキャッシュ。
            scopes: オプションの認証スコープリスト。指定しない場合は Power Platform API スコープがデフォルト。
            env_file_path: 設定読み込み用の .env ファイルのパス。
            env_file_encoding: .env ファイルのエンコーディング。デフォルトは 'utf-8'。

        Raises:
            ServiceInitializationError: 必須の設定が欠落または無効な場合。
        """
        super().__init__(
            id=id,
            name=name,
            description=description,
            context_providers=context_providers,
            middleware=middleware,
        )
        if not client:
            try:
                copilot_studio_settings = CopilotStudioSettings(
                    environmentid=environment_id,
                    schemaname=agent_identifier,
                    agentappid=client_id,
                    tenantid=tenant_id,
                    env_file_path=env_file_path,
                    env_file_encoding=env_file_encoding,
                )
            except ValidationError as ex:
                raise ServiceInitializationError("Failed to create Copilot Studio settings.", ex) from ex

            if not settings:
                if not copilot_studio_settings.environmentid:
                    raise ServiceInitializationError(
                        "Copilot Studio environment ID is required. Set via 'environment_id' parameter "
                        "or 'COPILOTSTUDIOAGENT__ENVIRONMENTID' environment variable."
                    )
                if not copilot_studio_settings.schemaname:
                    raise ServiceInitializationError(
                        "Copilot Studio agent identifier/schema name is required. Set via 'agent_identifier' parameter "
                        "or 'COPILOTSTUDIOAGENT__SCHEMANAME' environment variable."
                    )

                settings = ConnectionSettings(
                    environment_id=copilot_studio_settings.environmentid,
                    agent_identifier=copilot_studio_settings.schemaname,
                    cloud=cloud,
                    copilot_agent_type=agent_type,
                    custom_power_platform_cloud=custom_power_platform_cloud,
                )

            if not token:
                if not copilot_studio_settings.agentappid:
                    raise ServiceInitializationError(
                        "Copilot Studio client ID is required. Set via 'client_id' parameter "
                        "or 'COPILOTSTUDIOAGENT__AGENTAPPID' environment variable."
                    )

                if not copilot_studio_settings.tenantid:
                    raise ServiceInitializationError(
                        "Copilot Studio tenant ID is required. Set via 'tenant_id' parameter "
                        "or 'COPILOTSTUDIOAGENT__TENANTID' environment variable."
                    )

                token = acquire_token(
                    client_id=copilot_studio_settings.agentappid,
                    tenant_id=copilot_studio_settings.tenantid,
                    username=username,
                    token_cache=token_cache,
                    scopes=scopes,
                )

            client = CopilotClient(settings=settings, token=token)

        self.client = client
        self.cloud = cloud
        self.agent_type = agent_type
        self.custom_power_platform_cloud = custom_power_platform_cloud
        self.username = username
        self.token_cache = token_cache
        self.scopes = scopes

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """Agentからのレスポンスを取得します。

        このメソッドはAgentの実行の最終結果を単一のAgentRunResponseオブジェクトとして返します。
        呼び出し元は最終結果が利用可能になるまでブロックされます。

        注意: ストリーミングレスポンスの場合は、run_streamメソッドを使用してください。
        これは中間ステップと最終結果をAgentRunResponseUpdateオブジェクトのストリームとして返します。
        最終結果のみをストリーミングすることは、最終結果の利用可能なタイミングが不明であり、
        その時点まで呼び出し元をブロックすることがストリーミングシナリオでは望ましくないため、実現不可能です。

        Args:
            messages: Agentに送信するメッセージ。

        Keyword Args:
            thread: メッセージに関連付けられた会話スレッド。
            kwargs: 追加のキーワード引数。

        Returns:
            Agentのレスポンスアイテム。

        """
        if not thread:
            thread = self.get_new_thread()
        thread.service_thread_id = await self._start_new_conversation()

        input_messages = self._normalize_messages(messages)

        question = "\n".join([message.text for message in input_messages])

        activities = self.client.ask_question(question, thread.service_thread_id)
        response_messages: list[ChatMessage] = []
        response_id: str | None = None

        response_messages = [message async for message in self._process_activities(activities, streaming=False)]
        response_id = response_messages[0].message_id if response_messages else None

        return AgentRunResponse(messages=response_messages, response_id=response_id)

    async def run_stream(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """Agentをストリームとして実行します。

        このメソッドはAgentの実行の中間ステップと最終結果を
        AgentRunResponseUpdateオブジェクトのストリームとして呼び出し元に返します。

        注意: AgentRunResponseUpdateオブジェクトはメッセージのチャンクを含みます。

        Args:
            messages: Agentに送信するメッセージ。

        Keyword Args:
            thread: メッセージに関連付けられた会話スレッド。
            kwargs: 追加のキーワード引数。

        Yields:
            Agentのレスポンスアイテム。

        """
        if not thread:
            thread = self.get_new_thread()
        thread.service_thread_id = await self._start_new_conversation()

        input_messages = self._normalize_messages(messages)

        question = "\n".join([message.text for message in input_messages])

        activities = self.client.ask_question(question, thread.service_thread_id)

        async for message in self._process_activities(activities, streaming=True):
            yield AgentRunResponseUpdate(
                role=message.role,
                contents=message.contents,
                author_name=message.author_name,
                raw_representation=message.raw_representation,
                response_id=message.message_id,
                message_id=message.message_id,
            )

    async def _start_new_conversation(self) -> str:
        """Copilot Studio agentとの新しい会話を開始します。

        Returns:
            新しい会話の会話ID。

        Raises:
            ServiceException: 会話を開始できなかった場合。

        """
        conversation_id: str | None = None

        async for activity in self.client.start_conversation(emit_start_conversation_event=True):
            if activity and activity.conversation and activity.conversation.id:
                conversation_id = activity.conversation.id

        if not conversation_id:
            raise ServiceException("Failed to start a new conversation.")

        return conversation_id

    async def _process_activities(self, activities: AsyncIterable[Any], streaming: bool) -> AsyncIterable[ChatMessage]:
        """Copilot Studio agentからのアクティビティを処理します。

        Args:
            activities: agentからのアクティビティのストリーム。
            streaming: ストリーミング（typingアクティビティ）か非ストリーミング（メッセージアクティビティ）
                のレスポンスを処理するかどうか。

        Yields:
            アクティビティから作成されたChatMessageオブジェクト。

        """
        async for activity in activities:
            if activity.text and (
                (activity.type == "message" and not streaming) or (activity.type == "typing" and streaming)
            ):
                yield ChatMessage(
                    role=Role.ASSISTANT,
                    contents=[TextContent(activity.text)],
                    author_name=activity.from_property.name if activity.from_property else None,
                    message_id=activity.id,
                    raw_representation=activity,
                )
