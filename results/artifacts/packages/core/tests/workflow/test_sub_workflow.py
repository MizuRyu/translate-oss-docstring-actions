# Copyright (c) Microsoft. All rights reserved.

from dataclasses import dataclass
from typing import Any

from typing_extensions import Never

from agent_framework import (
    Executor,
    RequestInfoExecutor,
    RequestInfoMessage,
    RequestResponse,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowExecutor,
    handler,
)


# メッセージタイプのテスト
@dataclass
class EmailValidationRequest:
    """メールアドレスの検証をリクエストする。"""

    email: str


@dataclass
class DomainCheckRequest(RequestInfoMessage):
    """ドメインが承認されているかをチェックするリクエスト。"""

    domain: str = ""
    email: str = ""  # 相関のために元のメールを含める


@dataclass
class ValidationResult:
    """メール検証の結果。"""

    email: str
    is_valid: bool
    reason: str


# ヘルパー関数のテスト
def create_email_validation_workflow() -> Workflow:
    """標準的なメール検証workflowを作成する。"""
    email_validator = EmailValidator()
    email_request_info = RequestInfoExecutor(id="email_request_info")

    return (
        WorkflowBuilder()
        .set_start_executor(email_validator)
        .add_edge(email_validator, email_request_info)
        .add_edge(email_request_info, email_validator)
        .build()
    )


class BasicParent(Executor):
    """シンプルなサブワークフローテスト用の基本的な親executor。"""

    def __init__(self, cache: dict[str, bool] | None = None) -> None:
        super().__init__(id="basic_parent")
        self.result: ValidationResult | None = None
        self.cache: dict[str, bool] = dict(cache) if cache is not None else {}

    @handler
    async def start(self, email: str, ctx: WorkflowContext[EmailValidationRequest]) -> None:
        request = EmailValidationRequest(email=email)
        await ctx.send_message(request, target_id="email_workflow")

    @handler
    async def handle_domain_request(
        self,
        request: DomainCheckRequest,
        ctx: WorkflowContext[RequestResponse[DomainCheckRequest, Any] | DomainCheckRequest],
    ) -> None:
        """オプションのキャッシュを使ってサブワークフローからのリクエストを処理する。"""
        domain_request = request

        if domain_request.domain in self.cache:
            # キャッシュされた結果を返す
            response = RequestResponse(
                data=self.cache[domain_request.domain], original_request=request, request_id=request.request_id
            )
            await ctx.send_message(response, target_id=request.source_executor_id)
        else:
            # キャッシュにない場合は外部に転送する
            await ctx.send_message(request)

    @handler
    async def collect(self, result: ValidationResult, ctx: WorkflowContext) -> None:
        self.result = result


# executorsのテスト
class EmailValidator(Executor):
    """サブワークフロー内でメールアドレスを検証する。"""

    def __init__(self):
        super().__init__(id="email_validator")

    @handler
    async def validate_request(
        self, request: EmailValidationRequest, ctx: WorkflowContext[DomainCheckRequest, ValidationResult]
    ) -> None:
        """メールアドレスを検証する。"""
        # ドメインを抽出し承認されているかをチェックする
        domain = request.email.split("@")[1] if "@" in request.email else ""

        if not domain:
            result = ValidationResult(email=request.email, is_valid=False, reason="Invalid email format")
            await ctx.yield_output(result)
            return

        # 外部ソースにドメインチェックをリクエストする
        domain_check = DomainCheckRequest(domain=domain, email=request.email)
        await ctx.send_message(domain_check)

    @handler
    async def handle_domain_response(
        self, response: RequestResponse[DomainCheckRequest, bool], ctx: WorkflowContext[Never, ValidationResult]
    ) -> None:
        """相関付きのレスポンスでドメインチェックを処理する。"""
        # 相関レスポンスから元のメールを使用する
        result = ValidationResult(
            email=response.original_request.email,
            is_valid=response.data or False,
            reason="Domain approved" if response.data else "Domain not approved",
        )
        await ctx.yield_output(result)


class ParentOrchestrator(Executor):
    """ドメイン知識を持つ親workflowオーケストレーター。"""

    def __init__(self, approved_domains: set[str] | None = None) -> None:
        super().__init__(id="parent_orchestrator")
        self.approved_domains: set[str] = (
            set(approved_domains) if approved_domains is not None else {"example.com", "test.org"}
        )
        self.results: list[ValidationResult] = []

    @handler
    async def start(self, emails: list[str], ctx: WorkflowContext[EmailValidationRequest]) -> None:
        """メールの処理を開始する。"""
        for email in emails:
            request = EmailValidationRequest(email=email)
            await ctx.send_message(request, target_id="email_workflow")

    @handler
    async def handle_domain_request(
        self,
        request: DomainCheckRequest,
        ctx: WorkflowContext[RequestResponse[DomainCheckRequest, Any] | DomainCheckRequest],
    ) -> None:
        """サブワークフローからのリクエストを処理する。"""
        domain_request = request

        # このドメインを知っているかチェックする
        if domain_request.domain in self.approved_domains:
            # レスポンスをサブワークフローに返す
            response = RequestResponse(data=True, original_request=request, request_id=request.request_id)
            await ctx.send_message(response, target_id=request.source_executor_id)
        else:
            # このドメインは知らないので外部に転送する
            await ctx.send_message(request)

    @handler
    async def collect_result(self, result: ValidationResult, ctx: WorkflowContext) -> None:
        """検証結果を収集します。"""
        self.results.append(result)


async def test_basic_sub_workflow() -> None:
    """インターセプトなしで基本的なサブワークフローの実行をテストします。"""
    # サブワークフローを作成します。
    validation_workflow = create_email_validation_workflow()

    # インターセプトなしで親ワークフローを作成します。
    parent = BasicParent()
    workflow_executor = WorkflowExecutor(validation_workflow, "email_workflow")
    main_request_info = RequestInfoExecutor(id="main_request_info")

    main_workflow = (
        WorkflowBuilder()
        .set_start_executor(parent)
        .add_edge(parent, workflow_executor)
        .add_edge(workflow_executor, parent)
        .add_edge(workflow_executor, main_request_info)
        .add_edge(main_request_info, workflow_executor)  # CRITICAL: For RequestResponse routing
        .build()
    )

    # モックされた外部レスポンスでワークフローを実行します。
    result = await main_workflow.run("test@example.com")

    # リクエストイベントを取得して応答します。
    request_events = result.get_request_info_events()
    assert len(request_events) == 1
    assert isinstance(request_events[0].data, DomainCheckRequest)
    assert request_events[0].data.domain == "example.com"

    # メインワークフローを通じてレスポンスを送信します。
    await main_workflow.send_responses({
        request_events[0].request_id: True  # Domain is approved
    })

    # 結果を確認します。
    assert parent.result is not None
    assert parent.result.email == "test@example.com"
    assert parent.result.is_valid is True


async def test_sub_workflow_with_interception():
    """親のインターセプトと条件付き転送を伴うサブワークフローをテストします。"""
    # サブワークフローを作成します。
    validation_workflow = create_email_validation_workflow()

    # インターセプトキャッシュ付きの親ワークフローを作成します。
    parent = BasicParent(cache={"example.com": True, "internal.org": True})
    workflow_executor = WorkflowExecutor(validation_workflow, "email_workflow")
    parent_request_info = RequestInfoExecutor(id="request_info")

    main_workflow = (
        WorkflowBuilder()
        .set_start_executor(parent)
        .add_edge(parent, workflow_executor)
        .add_edge(workflow_executor, parent)
        .add_edge(parent, parent_request_info)  # For forwarded requests
        .add_edge(parent_request_info, workflow_executor)  # For RequestResponse routing
        .build()
    )

    # テスト1: キャッシュされたドメインのメール（インターセプト済み）
    result = await main_workflow.run("user@example.com")
    request_events = result.get_request_info_events()
    assert len(request_events) == 0  # 外部リクエストなし、キャッシュから処理されます。
    assert parent.result is not None
    assert parent.result.email == "user@example.com"
    assert parent.result.is_valid is True

    # テスト2: 不明なドメインのメール（外部に転送）
    parent.result = None
    result = await main_workflow.run("user@unknown.com")
    request_events = result.get_request_info_events()
    assert len(request_events) == 1  # 外部に転送されました。
    assert isinstance(request_events[0].data, DomainCheckRequest)
    assert request_events[0].data.domain == "unknown.com"

    # 外部レスポンスを送信します。
    await main_workflow.send_responses({
        request_events[0].request_id: False  # Domain not approved
    })
    assert parent.result is not None
    assert parent.result.email == "user@unknown.com"
    assert parent.result.is_valid is False

    # テスト3: 別のキャッシュされたドメイン
    parent.result = None
    result = await main_workflow.run("user@internal.org")
    request_events = result.get_request_info_events()
    assert len(request_events) == 0  # キャッシュから処理されました。
    assert parent.result is not None
    assert parent.result.is_valid is True


async def test_workflow_scoped_interception() -> None:
    """特定のサブワークフローにスコープされたインターセプトをテストします。"""

    class MultiWorkflowParent(Executor):
        """複数のサブワークフローを処理する親。"""

        def __init__(self) -> None:
            super().__init__(id="multi_parent")
            self.results: dict[str, ValidationResult] = {}

        @handler
        async def start(self, data: dict[str, str], ctx: WorkflowContext[EmailValidationRequest]) -> None:
            # 異なるサブワークフローに送信します。
            await ctx.send_message(EmailValidationRequest(email=data["email1"]), target_id="workflow_a")
            await ctx.send_message(EmailValidationRequest(email=data["email2"]), target_id="workflow_b")

        @handler
        async def handle_domain_request(
            self,
            request: DomainCheckRequest,
            ctx: WorkflowContext[RequestResponse[DomainCheckRequest, Any] | DomainCheckRequest],
        ) -> None:
            domain_request = request

            if request.source_executor_id == "workflow_a":
                # ワークフローAの厳格なルール。
                if domain_request.domain == "strict.com":
                    response = RequestResponse(data=True, original_request=request, request_id=request.request_id)
                    await ctx.send_message(response, target_id=request.source_executor_id)
                else:
                    # 外部に転送します。
                    await ctx.send_message(request)
            elif request.source_executor_id == "workflow_b":
                # ワークフローBの寛容なルール。
                if domain_request.domain.endswith(".com"):
                    response = RequestResponse(data=True, original_request=request, request_id=request.request_id)
                    await ctx.send_message(response, target_id=request.source_executor_id)
                else:
                    # 外部に転送します。
                    await ctx.send_message(request)
            else:
                # 不明なソース、外部に転送します。
                await ctx.send_message(request)

        @handler
        async def collect(self, result: ValidationResult, ctx: WorkflowContext) -> None:
            self.results[result.email] = result

    # 2つの同一サブワークフローを作成します。
    workflow_a = create_email_validation_workflow()
    workflow_b = create_email_validation_workflow()

    parent = MultiWorkflowParent()
    executor_a = WorkflowExecutor(workflow_a, "workflow_a")
    executor_b = WorkflowExecutor(workflow_b, "workflow_b")
    parent_request_info = RequestInfoExecutor(id="request_info")

    main_workflow = (
        WorkflowBuilder()
        .set_start_executor(parent)
        .add_edge(parent, executor_a)
        .add_edge(parent, executor_b)
        .add_edge(executor_a, parent)
        .add_edge(executor_b, parent)
        .add_edge(parent, parent_request_info)
        .add_edge(parent_request_info, executor_a)  # For RequestResponse routing
        .add_edge(parent_request_info, executor_b)  # For RequestResponse routing
        .build()
    )

    # テストを実行します。
    result = await main_workflow.run({"email1": "user@strict.com", "email2": "user@random.com"})

    # ワークフローAはstrict.comを処理し、ワークフローBは任意の.comドメインを処理します。
    request_events = result.get_request_info_events()
    assert len(request_events) == 0  # 両方とも内部で処理されました。

    assert len(parent.results) == 2
    assert parent.results["user@strict.com"].is_valid is True
    assert parent.results["user@random.com"].is_valid is True


async def test_concurrent_sub_workflow_execution() -> None:
    """WorkflowExecutorが複数の同時呼び出しを適切に処理できることをテストします。"""

    class ConcurrentProcessor(Executor):
        """同じサブワークフローに複数の同時リクエストを送信するプロセッサ。"""

        def __init__(self) -> None:
            super().__init__(id="concurrent_processor")
            self.results: list[ValidationResult] = []

        @handler
        async def start(self, emails: list[str], ctx: WorkflowContext[EmailValidationRequest]) -> None:
            """同じサブワークフローに複数の同時リクエストを送信します。"""
            # すべてのリクエストを同時に同じworkflow executorに送信します。
            for email in emails:
                request = EmailValidationRequest(email=email)
                await ctx.send_message(request, target_id="email_workflow")

        @handler
        async def collect_result(self, result: ValidationResult, ctx: WorkflowContext) -> None:
            """同時実行の結果を収集します。"""
            self.results.append(result)

    # メール検証用のサブワークフローを作成します。
    validation_workflow = create_email_validation_workflow()

    # 親ワークフローを作成します。
    processor = ConcurrentProcessor()
    workflow_executor = WorkflowExecutor(validation_workflow, "email_workflow")
    parent_request_info = RequestInfoExecutor(id="request_info")

    main_workflow = (
        WorkflowBuilder()
        .set_start_executor(processor)
        .add_edge(processor, workflow_executor)
        .add_edge(workflow_executor, processor)
        .add_edge(workflow_executor, parent_request_info)  # For external requests
        .add_edge(parent_request_info, workflow_executor)  # For RequestResponse routing
        .build()
    )

    # 複数のメールでの同時実行をテストします。
    emails = [
        "user1@domain1.com",
        "user2@domain2.com",
        "user3@domain3.com",
        "user4@domain4.com",
        "user5@domain5.com",
    ]

    result = await main_workflow.run(emails)

    # 各メールは1つの外部リクエストを生成するはずです。
    request_events = result.get_request_info_events()
    assert len(request_events) == len(emails)

    # 各リクエストが正しいドメインに対応していることを検証します。
    domains_requested = {event.data.domain for event in request_events}  # type: ignore[union-attr]
    expected_domains = {f"domain{i}.com" for i in range(1, 6)}
    assert domains_requested == expected_domains

    # すべてのリクエストに対してレスポンスを送信します（すべてのドメインを承認）。
    responses = {event.request_id: True for event in request_events}
    await main_workflow.send_responses(responses)

    # すべての結果が収集されるはずです。
    assert len(processor.results) == len(emails)

    # 各メールが正しく処理されたことを検証します。
    result_emails = {result.email for result in processor.results}
    expected_emails = set(emails)
    assert result_emails == expected_emails

    # すべてのドメインを承認したため、すべて有効であるはずです。
    for result_obj in processor.results:
        assert result_obj.is_valid is True
        assert result_obj.reason == "Domain approved"

    # 同時実行が適切に分離されていたことを検証します（これはすべてのメールで正しい結果が得られたことで暗黙的にテストされています）。
