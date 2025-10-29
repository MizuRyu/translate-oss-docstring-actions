# Copyright (c) Microsoft. All rights reserved.

import asyncio
from dataclasses import dataclass
from typing import Any

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
from typing_extensions import Never

"""
Sample: Sub-workflow with parallel request handling by specialized interceptors

This sample demonstrates how different parent executors can handle different types of requests
from the same sub-workflow using regular @handler methods for RequestInfoMessage subclasses.

Prerequisites:
- No external services required (external handling simulated via `RequestInfoExecutor`).

Key architectural principles:
1. Specialized interceptors: Each parent executor handles only specific request types
2. Type-based routing: ResourceCache handles ResourceRequest, PolicyEngine handles PolicyCheckRequest
3. Automatic type filtering: Each interceptor only receives requests with matching types
4. Fallback forwarding: Unhandled requests are forwarded to external services

The example simulates a resource allocation system where:
- Sub-workflow makes mixed requests for resources (CPU, memory) and policy checks
- ResourceCache executor intercepts ResourceRequest messages, serves from cache or forwards
- PolicyEngine executor intercepts PolicyCheckRequest messages, applies rules or forwards
- Each interceptor uses typed @handler methods for automatic filtering

Flow visualization:

  Coordinator
      |
      |  Mixed list[resource + policy requests]
      v
    [ Sub-workflow: WorkflowExecutor(ResourceRequester) ]
      |
      |  Emits different RequestInfoMessage types:
      |     - ResourceRequest
      |     - PolicyCheckRequest
      v
  Parent workflow routes to specialized handlers:
      |                                    |
      | ResourceCache.handle_resource_request | PolicyEngine.handle_policy_request
      | (@handler ResourceRequest)          | (@handler PolicyCheckRequest)
      v                                    v
  Cache hit/miss decision              Policy allow/deny decision
      |                                    |
      | RequestResponse OR forward        | RequestResponse OR forward
      v                                    v
  Back to sub-workflow  <----------> External RequestInfoExecutor
                                           |
                                           v
                                    External responses route back
"""


# 1. ドメイン固有の request/response タイプを定義します。
@dataclass
class ResourceRequest(RequestInfoMessage):
    """計算リソースのリクエスト。"""

    resource_type: str = "cpu"  # cpu、memory、disk など。
    amount: int = 1
    priority: str = "normal"  # low、normal、high。


@dataclass
class PolicyCheckRequest(RequestInfoMessage):
    """リソース割り当てポリシーをチェックするリクエスト。"""

    resource_type: str = ""
    amount: int = 0
    policy_type: str = "quota"  # quota、compliance、security。


@dataclass
class ResourceResponse:
    """割り当てられたリソースのレスポンス。"""

    resource_type: str
    allocated: int
    source: str  # どのシステムがリソースを提供したか。


@dataclass
class PolicyResponse:
    """ポリシーチェックのレスポンス。"""

    approved: bool
    reason: str


@dataclass
class RequestFinished:
    pass


# 2. サブワークフロー executor を実装します - リソースとポリシーのリクエストを行います。
class ResourceRequester(Executor):
    """リソースをリクエストし、ポリシーをチェックするシンプルな executor。"""

    def __init__(self):
        super().__init__(id="resource_requester")
        self._request_count = 0

    @handler
    async def request_resources(
        self,
        requests: list[dict[str, Any]],
        ctx: WorkflowContext[ResourceRequest | PolicyCheckRequest],
    ) -> None:
        """リソースリクエストのリストを処理します。"""
        print(f"🏭 Sub-workflow processing {len(requests)} requests")
        self._request_count += len(requests)

        for req_data in requests:
            req_type = req_data.get("request_type", "resource")

            request: ResourceRequest | PolicyCheckRequest
            if req_type == "resource":
                print(f"  📦 Requesting resource: {req_data.get('type', 'cpu')} x{req_data.get('amount', 1)}")
                request = ResourceRequest(
                    resource_type=req_data.get("type", "cpu"),
                    amount=req_data.get("amount", 1),
                    priority=req_data.get("priority", "normal"),
                )
                # ターゲットID ではなく親ワークフローに送信してインターセプトします。
                await ctx.send_message(request)
            elif req_type == "policy":
                print(
                    f"  🛡️  Checking policy: {req_data.get('type', 'cpu')} x{req_data.get('amount', 1)} "
                    f"({req_data.get('policy_type', 'quota')})"
                )
                request = PolicyCheckRequest(
                    resource_type=req_data.get("type", "cpu"),
                    amount=req_data.get("amount", 1),
                    policy_type=req_data.get("policy_type", "quota"),
                )
                # ターゲットID ではなく親ワークフローに送信してインターセプトします。
                await ctx.send_message(request)

    @handler
    async def handle_resource_response(
        self,
        response: RequestResponse[ResourceRequest, ResourceResponse],
        ctx: WorkflowContext[Never, RequestFinished],
    ) -> None:
        """リソース割り当てのレスポンスを処理します。"""
        if response.data:
            source_icon = "🏪" if response.data.source == "cache" else "🌐"
            print(
                f"📦 {source_icon} Sub-workflow received: {response.data.allocated} {response.data.resource_type} "
                f"from {response.data.source}"
            )
            if self._collect_results():
                # 完了結果を親ワークフローに yield します。
                await ctx.yield_output(RequestFinished())

    @handler
    async def handle_policy_response(
        self,
        response: RequestResponse[PolicyCheckRequest, PolicyResponse],
        ctx: WorkflowContext[Never, RequestFinished],
    ) -> None:
        """ポリシーチェックのレスポンスを処理します。"""
        if response.data:
            status_icon = "✅" if response.data.approved else "❌"
            print(
                f"🛡️  {status_icon} Sub-workflow received policy response: "
                f"{response.data.approved} - {response.data.reason}"
            )
            if self._collect_results():
                # 完了結果を親ワークフローに yield します。
                await ctx.yield_output(RequestFinished())

    def _collect_results(self) -> bool:
        """結果を収集し、サマリーを作成します。"""
        self._request_count -= 1
        print(f"📊 Sub-workflow completed request ({self._request_count} remaining)")
        return self._request_count == 0


# 3. Resource Cache を実装します - ResourceRequest 用の型付きハンドラーを使用します。
class ResourceCache(Executor):
    """型付きルーティングを使ってキャッシュから RESOURCE リクエストを処理するインターセプター。"""

    # Pydantic の割り当て制限を回避するためにクラス属性を使用します。
    cache: dict[str, int] = {"cpu": 10, "memory": 50, "disk": 100}
    results: list[ResourceResponse] = []

    def __init__(self):
        super().__init__(id="resource_cache")
        # インスタンス初期化のみ；状態は上記のようにクラス属性に保持します。

    @handler
    async def handle_resource_request(
        self, request: ResourceRequest, ctx: WorkflowContext[RequestResponse[ResourceRequest, Any] | ResourceRequest]
    ) -> None:
        """サブワークフローからの RESOURCE リクエストを処理し、まずキャッシュをチェックします。"""
        resource_request = request
        print(f"🏪 CACHE interceptor checking: {resource_request.amount} {resource_request.resource_type}")

        available = self.cache.get(resource_request.resource_type, 0)

        if available >= resource_request.amount:
            # キャッシュから満たすことができます。
            self.cache[resource_request.resource_type] -= resource_request.amount
            response_data = ResourceResponse(
                resource_type=resource_request.resource_type, allocated=resource_request.amount, source="cache"
            )
            print(f"  ✅ Cache satisfied: {resource_request.amount} {resource_request.resource_type}")
            self.results.append(response_data)

            # レスポンスをサブワークフローに返送します。
            response = RequestResponse(data=response_data, original_request=request, request_id=request.request_id)
            await ctx.send_message(response, target_id=request.source_executor_id)
        else:
            # キャッシュミス - 外部に転送します。
            print(f"  ❌ Cache miss: need {resource_request.amount}, have {available} {resource_request.resource_type}")
            await ctx.send_message(request)

    @handler
    async def collect_result(
        self, response: RequestResponse[ResourceRequest, ResourceResponse], ctx: WorkflowContext
    ) -> None:
        """転送された外部リクエストから結果を収集します。"""
        if response.data and response.data.source != "cache":  # Don't double-count our own results
            self.results.append(response.data)
            print(
                f"🏪 🌐 Cache received external response: {response.data.allocated} {response.data.resource_type} "
                f"from {response.data.source}"
            )


# 4. Policy Engine を実装します - PolicyCheckRequest 用の型付きハンドラーを使用します。
class PolicyEngine(Executor):
    """型付きルーティングを使って POLICY リクエストを処理するインターセプター。"""

    # シンプルなサンプル状態のためにクラス属性を使用します。
    quota: dict[str, int] = {
        "cpu": 5,  # Only allow up to 5 CPU units
        "memory": 20,  # Only allow up to 20 memory units
        "disk": 1000,  # Liberal disk policy
    }
    results: list[PolicyResponse] = []

    def __init__(self):
        super().__init__(id="policy_engine")
        # インスタンス初期化のみ；状態は上記のようにクラス属性に保持します。

    @handler
    async def handle_policy_request(
        self,
        request: PolicyCheckRequest,
        ctx: WorkflowContext[RequestResponse[PolicyCheckRequest, Any] | PolicyCheckRequest],
    ) -> None:
        """サブワークフローからの POLICY リクエストを処理し、ルールを適用します。"""
        policy_request = request
        print(
            f"🛡️  POLICY interceptor checking: {policy_request.amount} {policy_request.resource_type}, policy={policy_request.policy_type}"
        )

        quota_limit = self.quota.get(policy_request.resource_type, 0)

        if policy_request.policy_type == "quota":
            if policy_request.amount <= quota_limit:
                response_data = PolicyResponse(approved=True, reason=f"Within quota ({quota_limit})")
                print(f"  ✅ Policy approved: {policy_request.amount} <= {quota_limit}")
                self.results.append(response_data)

                # レスポンスをサブワークフローに返送します。
                response = RequestResponse(data=response_data, original_request=request, request_id=request.request_id)
                await ctx.send_message(response, target_id=request.source_executor_id)
                return

            # クォータ超過 - レビューのため外部に転送します。
            print(f"  ❌ Policy exceeds quota: {policy_request.amount} > {quota_limit}, forwarding to external")
            await ctx.send_message(request)
            return

        # 不明なポリシータイプ - 外部に転送
        print(f"  ❓ Unknown policy type: {policy_request.policy_type}, forwarding")
        await ctx.send_message(request)

    @handler
    async def collect_policy_result(
        self, response: RequestResponse[PolicyCheckRequest, PolicyResponse], ctx: WorkflowContext
    ) -> None:
        """転送された外部リクエストからポリシー結果を収集します。"""
        if response.data:
            self.results.append(response.data)
            print(f"🛡️  🌐 Policy received external response: {response.data.approved} - {response.data.reason}")


class Coordinator(Executor):
    def __init__(self):
        super().__init__(id="coordinator")

    @handler
    async def start(self, requests: list[dict[str, Any]], ctx: WorkflowContext[list[dict[str, Any]]]) -> None:
        """リソース割り当てプロセスを開始します。"""
        await ctx.send_message(requests, target_id="resource_workflow")

    @handler
    async def handle_completion(self, completion: RequestFinished, ctx: WorkflowContext) -> None:
        """サブワークフローの完了を処理します。

        これはサブワークフローから生成された出力に由来します。

        """
        print("🎯 Main workflow received completion.")


async def main() -> None:
    """並列リクエストインターセプトパターンを示します。"""
    print("🚀 Starting Sub-Workflow Parallel Request Interception Demo...")
    print("=" * 60)

    # 5. サブワークフローを作成する
    resource_requester = ResourceRequester()
    sub_request_info = RequestInfoExecutor(id="sub_request_info")

    sub_workflow = (
        WorkflowBuilder()
        .set_start_executor(resource_requester)
        .add_edge(resource_requester, sub_request_info)
        .add_edge(sub_request_info, resource_requester)
        .build()
    )

    # 6. 適切なインターセプターパターンで親ワークフローを作成する
    cache = ResourceCache()  # ResourceRequestをインターセプトします
    policy = PolicyEngine()  # PolicyCheckRequest（異なるタイプ！）をインターセプトします
    workflow_executor = WorkflowExecutor(sub_workflow, id="resource_workflow")
    main_request_info = RequestInfoExecutor(id="main_request_info")

    # プロセスを開始するシンプルなコーディネーターを作成する
    coordinator = Coordinator()

    # TYPED ROUTING: 各executorは特定の型のRequestInfoMessageメッセージを処理します
    main_workflow = (
        WorkflowBuilder()
        .set_start_executor(coordinator)
        .add_edge(coordinator, workflow_executor)  # Start sub-workflow
        .add_edge(workflow_executor, coordinator)  # Sub-workflow completion back to coordinator
        .add_edge(workflow_executor, cache)  # WorkflowExecutor sends ResourceRequest to cache
        .add_edge(workflow_executor, policy)  # WorkflowExecutor sends PolicyCheckRequest to policy
        .add_edge(cache, workflow_executor)  # Cache sends RequestResponse back
        .add_edge(policy, workflow_executor)  # Policy sends RequestResponse back
        .add_edge(cache, main_request_info)  # Cache forwards ResourceRequest to external
        .add_edge(policy, main_request_info)  # Policy forwards PolicyCheckRequest to external
        .add_edge(main_request_info, workflow_executor)  # External responses back to sub-workflow
        .build()
    )

    # 7. さまざまなリクエスト（リソースとポリシーの混合）でテストする
    test_requests = [
        {"request_type": "resource", "type": "cpu", "amount": 2, "priority": "normal"},  # Cache hit
        {"request_type": "policy", "type": "cpu", "amount": 3, "policy_type": "quota"},  # Policy hit
        {"request_type": "resource", "type": "memory", "amount": 15, "priority": "normal"},  # Cache hit
        {"request_type": "policy", "type": "memory", "amount": 100, "policy_type": "quota"},  # Policy miss -> external
        {"request_type": "resource", "type": "gpu", "amount": 1, "priority": "high"},  # Cache miss -> external
        {"request_type": "policy", "type": "disk", "amount": 500, "policy_type": "quota"},  # Policy hit
        {"request_type": "policy", "type": "cpu", "amount": 1, "policy_type": "security"},  # Unknown policy -> external
    ]

    print(f"🧪 Testing with {len(test_requests)} mixed requests:")
    for i, req in enumerate(test_requests, 1):
        req_icon = "📦" if req["request_type"] == "resource" else "🛡️"
        print(
            f"  {i}. {req_icon} {req['type']} x{req['amount']} "
            f"({req.get('priority', req.get('policy_type', 'default'))})"
        )
    print("=" * 70)

    # 8. ワークフローを実行する
    print("🎬 Running workflow...")
    events = await main_workflow.run(test_requests)

    # 9. インターセプトできなかった外部リクエストを処理する
    request_events = events.get_request_info_events()
    if request_events:
        print(f"\n🌐 Handling {len(request_events)} external request(s)...")

        external_responses: dict[str, Any] = {}
        for event in request_events:
            if isinstance(event.data, ResourceRequest):
                # ResourceRequestを処理し、ResourceResponseを作成する
                resource_response = ResourceResponse(
                    resource_type=event.data.resource_type, allocated=event.data.amount, source="external_provider"
                )
                external_responses[event.request_id] = resource_response
                print(f"  🏭 External provider: {resource_response.allocated} {resource_response.resource_type}")
            elif isinstance(event.data, PolicyCheckRequest):
                # PolicyCheckRequestを処理し、PolicyResponseを作成する
                policy_response = PolicyResponse(approved=True, reason="External policy service approved")
                external_responses[event.request_id] = policy_response
                print(f"  🔒 External policy: {'✅ APPROVED' if policy_response.approved else '❌ DENIED'}")

        await main_workflow.send_responses(external_responses)
    else:
        print("\n🎯 All requests were intercepted internally!")

    # 10. 結果と分析を表示する
    print("\n" + "=" * 70)
    print("📊 RESULTS ANALYSIS")
    print("=" * 70)

    print(f"\n🏪 Cache Results ({len(cache.results)} handled):")
    for result in cache.results:
        print(f"  ✅ {result.allocated} {result.resource_type} from {result.source}")

    print(f"\n🛡️  Policy Results ({len(policy.results)} handled):")
    for result in policy.results:
        status_icon = "✅" if result.approved else "❌"
        print(f"  {status_icon} Approved: {result.approved} - {result.reason}")

    print("\n💾 Final Cache State:")
    for resource, amount in cache.cache.items():
        print(f"  📦 {resource}: {amount} remaining")

    print("\n📈 Summary:")
    print(f"  🎯 Total requests: {len(test_requests)}")
    print(f"  🏪 Resource requests handled: {len(cache.results)}")
    print(f"  🛡️  Policy requests handled: {len(policy.results)}")
    print(f"  🌐 External requests: {len(request_events) if request_events else 0}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
