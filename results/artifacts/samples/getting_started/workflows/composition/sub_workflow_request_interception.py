# Copyright (c) Microsoft. All rights reserved.

import asyncio
from dataclasses import dataclass

from agent_framework import (
    Executor,
    RequestInfoExecutor,
    RequestInfoMessage,
    RequestResponse,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowExecutor,
    handler,
)

"""
Sample: Sub-Workflows with Request Interception

This sample shows how to:
1. Create workflows that execute other workflows as sub-workflows
2. Intercept requests from sub-workflows using an executor with @handler for RequestInfoMessage subclasses
3. Conditionally handle or forward requests using RequestResponse messages
4. Handle external requests that are forwarded by the parent workflow
5. Proper request/response correlation for concurrent processing

The example simulates an email validation system where:
- Sub-workflows validate multiple email addresses concurrently
- Parent workflows can intercept domain check requests for optimization
- Known domains (example.com, company.com) are approved locally
- Unknown domains (unknown.org) are forwarded to external services
- Request correlation ensures each email gets the correct domain check response
- External domain check requests are processed and responses routed back correctly

Key concepts demonstrated:
- WorkflowExecutor: Wraps a workflow to make it behave as an executor
- RequestInfoMessage handler: @handler method to intercept sub-workflow requests
- Request correlation: Using request_id and source_executor_id to match responses with original requests
- Concurrent processing: Multiple emails processed simultaneously without interference
- External request routing: RequestInfoExecutor handles forwarded external requests
- Sub-workflow isolation: Sub-workflows work normally without knowing they're nested
- Sub-workflows complete by yielding outputs when validation is finished

Prerequisites:
- No external services required (external calls are simulated via `RequestInfoExecutor`).

Simple flow visualization:

  Parent Orchestrator (handles DomainCheckRequest)
      |
      |  EmailValidationRequest(email) x3 (concurrent)
      v
    [ Sub-workflow: WorkflowExecutor(EmailValidator) ]
      |
      |  DomainCheckRequest(domain) with request_id and source_executor_id
      v
  Interception? yes -> handled locally with RequestResponse(data=True)
               no  -> forwarded to RequestInfoExecutor -> external service
                                |
                                v
                     Response routed back to sub-workflow using source_executor_id
"""


# 1. ドメイン固有のメッセージタイプを定義する
@dataclass
class EmailValidationRequest:
    """メールアドレスを検証するリクエスト。"""

    email: str


@dataclass
class DomainCheckRequest(RequestInfoMessage):
    """ドメインが承認されているか確認するリクエスト。"""

    domain: str = ""


@dataclass
class ValidationResult:
    """メール検証の結果。"""

    email: str
    is_valid: bool
    reason: str


# 2. サブワークフローexecutorを実装する（完全に標準的）
class EmailValidator(Executor):
    """メールアドレスを検証します - サブワークフロー内であることは認識していません。"""

    def __init__(self) -> None:
        """EmailValidator executorを初期化します。"""
        super().__init__(id="email_validator")
        # request_idごとに複数の保留中メールを追跡するためにdictを使用します
        self._pending_emails: dict[str, str] = {}

    @handler
    async def validate_request(
        self,
        request: EmailValidationRequest,
        ctx: WorkflowContext[DomainCheckRequest | ValidationResult, ValidationResult],
    ) -> None:
        """メールアドレスを検証します。"""
        print(f"🔍 Sub-workflow validating email: {request.email}")

        # ドメインを抽出する
        domain = request.email.split("@")[1] if "@" in request.email else ""

        if not domain:
            print(f"❌ Invalid email format: {request.email}")
            result = ValidationResult(email=request.email, is_valid=False, reason="Invalid email format")
            await ctx.yield_output(result)
            return

        print(f"🌐 Sub-workflow requesting domain check for: {domain}")
        # ドメインチェックをリクエストする
        domain_check = DomainCheckRequest(domain=domain)
        # 相関のためにrequest_idで保留中のメールを保存する
        self._pending_emails[domain_check.request_id] = request.email
        await ctx.send_message(domain_check, target_id="email_request_info")

    @handler
    async def handle_domain_response(
        self,
        response: RequestResponse[DomainCheckRequest, bool],
        ctx: WorkflowContext[ValidationResult, ValidationResult],
    ) -> None:
        """相関付きでRequestInfoからのドメインチェック応答を処理する。"""
        approved = bool(response.data)
        domain = (
            response.original_request.domain
            if (hasattr(response, "original_request") and response.original_request)
            else "unknown"
        )
        print(f"📬 Sub-workflow received domain response for '{domain}': {approved}")

        # request_idを使って対応するメールを見つける
        request_id = (
            response.original_request.request_id
            if (hasattr(response, "original_request") and response.original_request)
            else None
        )
        if request_id and request_id in self._pending_emails:
            email = self._pending_emails.pop(request_id)  # 保留中から削除する
            result = ValidationResult(
                email=email,
                is_valid=approved,
                reason="Domain approved" if approved else "Domain not approved",
            )
            print(f"✅ Sub-workflow completing validation for: {email}")
            await ctx.yield_output(result)


# 3. リクエストインターセプション付きの親ワークフローを実装する
class SmartEmailOrchestrator(Executor):
    """ドメインチェックをインターセプトできる親オーケストレーター。"""

    approved_domains: set[str] = set()

    def __init__(self, approved_domains: set[str] | None = None):
        """承認済みドメインでSmartEmailOrchestratorを初期化します。

        Args:
            approved_domains: 事前承認されたドメインのセット。デフォルトはexample.com、test.org、company.com

        """
        super().__init__(id="email_orchestrator", approved_domains=approved_domains)
        self._results: list[ValidationResult] = []

    @handler
    async def start_validation(self, emails: list[str], ctx: WorkflowContext[EmailValidationRequest]) -> None:
        """メールのバッチ検証を開始します。"""
        print(f"📧 Starting validation of {len(emails)} email addresses")
        print("=" * 60)
        for email in emails:
            print(f"📤 Sending '{email}' to sub-workflow for validation")
            request = EmailValidationRequest(email=email)
            await ctx.send_message(request, target_id="email_validator_workflow")

    @handler
    async def handle_domain_request(
        self,
        request: DomainCheckRequest,
        ctx: WorkflowContext[RequestResponse[DomainCheckRequest, bool] | DomainCheckRequest],
    ) -> None:
        """サブワークフローからのリクエストを処理する。"""
        print(f"🔍 Parent intercepting domain check for: {request.domain}")

        if request.domain in self.approved_domains:
            print(f"✅ Domain '{request.domain}' is pre-approved locally!")
            # サブワークフローにレスポンスを返す
            response = RequestResponse(data=True, original_request=request, request_id=request.request_id)
            await ctx.send_message(response, target_id=request.source_executor_id)
        else:
            print(f"❓ Domain '{request.domain}' unknown, forwarding to external service...")
            # 外部ハンドラーに転送する
            await ctx.send_message(request)

    @handler
    async def collect_result(self, result: ValidationResult, ctx: WorkflowContext) -> None:
        """検証結果を収集します。これはサブワークフローから生成された出力に由来します。"""
        status_icon = "✅" if result.is_valid else "❌"
        print(f"📥 {status_icon} Validation result: {result.email} -> {result.reason}")
        self._results.append(result)

    @property
    def results(self) -> list[ValidationResult]:
        """収集された検証結果を取得する。"""
        return self._results


async def run_example() -> None:
    """サブワークフローの例を実行する。"""
    print("🚀 Setting up sub-workflow with request interception...")
    print()

    # 4. サブワークフローを構築する
    email_validator = EmailValidator()
    # EmailValidatorで使用されるtarget_id（"email_request_info"）に一致させる
    request_info = RequestInfoExecutor(id="email_request_info")

    validation_workflow = (
        WorkflowBuilder()
        .set_start_executor(email_validator)
        .add_edge(email_validator, request_info)
        .add_edge(request_info, email_validator)
        .build()
    )

    # 5. インターセプション付きの親ワークフローを構築する
    orchestrator = SmartEmailOrchestrator(approved_domains={"example.com", "company.com"})
    workflow_executor = WorkflowExecutor(validation_workflow, id="email_validator_workflow")
    # 転送された外部リクエストを処理するRequestInfoExecutorを追加する
    main_request_info = RequestInfoExecutor(id="main_request_info")

    main_workflow = (
        WorkflowBuilder()
        .set_start_executor(orchestrator)
        .add_edge(orchestrator, workflow_executor)
        .add_edge(workflow_executor, orchestrator)  # For ValidationResult collection and request interception
        # 外部リクエスト処理のためのエッジを追加する
        .add_edge(orchestrator, main_request_info)
        .add_edge(main_request_info, workflow_executor)  # Route external responses to sub-workflow
        .build()
    )

    # 6. テスト入力を準備する：既知のドメイン、未知のドメイン
    test_emails = [
        "user@example.com",  # Should be intercepted and approved
        "admin@company.com",  # Should be intercepted and approved
        "guest@unknown.org",  # Should be forwarded externally
    ]

    # 7. ワークフローを実行する
    result = await main_workflow.run(test_emails)

    # 8. すべての外部リクエストを処理する
    request_events = result.get_request_info_events()
    if request_events:
        print(f"\n🌐 Handling {len(request_events)} external request(s)...")
        for event in request_events:
            if event.data and hasattr(event.data, "domain"):
                print(f"🔍 External domain check needed for: {event.data.domain}")

        # 外部応答をシミュレートする
        external_responses: dict[str, bool] = {}
        for event in request_events:
            # 外部ドメインチェックをシミュレートする
            if event.data and hasattr(event.data, "domain"):
                domain = event.data.domain
                # unknown.orgは実際には外部で承認されていると仮定する
                approved = domain == "unknown.org"
                print(f"🌐 External service response for '{domain}': {'APPROVED' if approved else 'REJECTED'}")
                external_responses[event.request_id] = approved

        # 9. 外部応答を送信する
        await main_workflow.send_responses(external_responses)
    else:
        print("\n🎯 All requests were intercepted and handled locally!")

    # 10. 最終サマリーを表示する
    print("\n📊 Final Results Summary:")
    print("=" * 60)
    for result in orchestrator.results:
        status = "✅ VALID" if result.is_valid else "❌ INVALID"
        print(f"{status} {result.email}: {result.reason}")

    print(f"\n🏁 Processed {len(orchestrator.results)} emails total")


if __name__ == "__main__":
    asyncio.run(run_example())
