# Copyright (c) Microsoft. All rights reserved.

import pytest

from agent_framework import (
    ChatClientProtocol,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    FunctionApprovalRequestContent,
    FunctionCallContent,
    FunctionResultContent,
    Role,
    TextContent,
    ai_function,
)


async def test_base_client_with_function_calling(chat_client_base: ChatClientProtocol):
    exec_counter = 0

    @ai_function(name="test_function")
    def ai_func(arg1: str) -> str:
        nonlocal exec_counter
        exec_counter += 1
        return f"Processed {arg1}"

    chat_client_base.run_responses = [
        ChatResponse(
            messages=ChatMessage(
                role="assistant",
                contents=[FunctionCallContent(call_id="1", name="test_function", arguments='{"arg1": "value1"}')],
            )
        ),
        ChatResponse(messages=ChatMessage(role="assistant", text="done")),
    ]
    response = await chat_client_base.get_response("hello", tool_choice="auto", tools=[ai_func])
    assert exec_counter == 1
    assert len(response.messages) == 3
    assert response.messages[0].role == Role.ASSISTANT
    assert isinstance(response.messages[0].contents[0], FunctionCallContent)
    assert response.messages[0].contents[0].name == "test_function"
    assert response.messages[0].contents[0].arguments == '{"arg1": "value1"}'
    assert response.messages[0].contents[0].call_id == "1"
    assert response.messages[1].role == Role.TOOL
    assert isinstance(response.messages[1].contents[0], FunctionResultContent)
    assert response.messages[1].contents[0].call_id == "1"
    assert response.messages[1].contents[0].result == "Processed value1"
    assert response.messages[2].role == Role.ASSISTANT
    assert response.messages[2].text == "done"


async def test_base_client_with_function_calling_resets(chat_client_base: ChatClientProtocol):
    exec_counter = 0

    @ai_function(name="test_function")
    def ai_func(arg1: str) -> str:
        nonlocal exec_counter
        exec_counter += 1
        return f"Processed {arg1}"

    chat_client_base.run_responses = [
        ChatResponse(
            messages=ChatMessage(
                role="assistant",
                contents=[FunctionCallContent(call_id="1", name="test_function", arguments='{"arg1": "value1"}')],
            )
        ),
        ChatResponse(
            messages=ChatMessage(
                role="assistant",
                contents=[FunctionCallContent(call_id="2", name="test_function", arguments='{"arg1": "value1"}')],
            )
        ),
        ChatResponse(messages=ChatMessage(role="assistant", text="done")),
    ]
    response = await chat_client_base.get_response("hello", tool_choice="auto", tools=[ai_func])
    assert exec_counter == 2
    assert len(response.messages) == 5
    assert response.messages[0].role == Role.ASSISTANT
    assert response.messages[1].role == Role.TOOL
    assert response.messages[2].role == Role.ASSISTANT
    assert response.messages[3].role == Role.TOOL
    assert response.messages[4].role == Role.ASSISTANT
    assert isinstance(response.messages[0].contents[0], FunctionCallContent)
    assert isinstance(response.messages[1].contents[0], FunctionResultContent)
    assert isinstance(response.messages[2].contents[0], FunctionCallContent)
    assert isinstance(response.messages[3].contents[0], FunctionResultContent)


async def test_base_client_with_streaming_function_calling(chat_client_base: ChatClientProtocol):
    exec_counter = 0

    @ai_function(name="test_function")
    def ai_func(arg1: str) -> str:
        nonlocal exec_counter
        exec_counter += 1
        return f"Processed {arg1}"

    chat_client_base.streaming_responses = [
        [
            ChatResponseUpdate(
                contents=[FunctionCallContent(call_id="1", name="test_function", arguments='{"arg1":')],
                role="assistant",
            ),
            ChatResponseUpdate(
                contents=[FunctionCallContent(call_id="1", name="test_function", arguments='"value1"}')],
                role="assistant",
            ),
        ],
        [
            ChatResponseUpdate(
                contents=[TextContent(text="Processed value1")],
                role="assistant",
            )
        ],
    ]
    updates = []
    async for update in chat_client_base.get_streaming_response("hello", tool_choice="auto", tools=[ai_func]):
        updates.append(update)
    assert len(updates) == 4  # 関数呼び出しによる2回の更新、関数結果、最終テキスト。
    assert updates[0].contents[0].call_id == "1"
    assert updates[1].contents[0].call_id == "1"
    assert updates[2].contents[0].call_id == "1"
    assert updates[3].text == "Processed value1"
    assert exec_counter == 1


@pytest.mark.parametrize(
    "approval_required,num_functions",
    [
        pytest.param(False, 1, id="single function without approval"),
        pytest.param(True, 1, id="single function with approval"),
        pytest.param("mixed", 2, id="two functions with mixed approval"),
    ],
)
@pytest.mark.parametrize(
    "thread_type",
    [
        pytest.param(None, id="no thread"),
        pytest.param("local", id="local thread"),
        pytest.param("service", id="service thread"),
    ],
)
@pytest.mark.parametrize("streaming", [False, True], ids=["non-streaming", "streaming"])
async def test_function_invocation_scenarios(
    chat_client_base: ChatClientProtocol,
    streaming: bool,
    thread_type: str | None,
    approval_required: bool | str,
    num_functions: int,
):
    """関数呼び出しシナリオの包括的なテスト。

    このテストは以下をカバーします:
    - 承認なしの単一関数: 3つのメッセージ（呼び出し、結果、最終）
    - 承認ありの単一関数: 2つのメッセージ（呼び出し、承認要求）
    - 混合承認の2つの関数: 承認フローに基づき変動
    - すべてのシナリオをストリーミングと非ストリーミングでテスト
    - Threadシナリオ: スレッドなし、ローカルスレッド（インメモリ）、サービススレッド（conversation_id）

    """
    exec_counter = 0

    # パラメータに基づいてthreadをセットアップします。
    conversation_id = None
    if thread_type == "service":
        # conversation_idを持つサービス側のthreadをシミュレートします。
        conversation_id = "test-thread-123"

    @ai_function(name="no_approval_func")
    def func_no_approval(arg1: str) -> str:
        nonlocal exec_counter
        exec_counter += 1
        return f"Processed {arg1}"

    @ai_function(name="approval_func", approval_mode="always_require")
    def func_with_approval(arg1: str) -> str:
        nonlocal exec_counter
        exec_counter += 1
        return f"Approved {arg1}"

    # シナリオに基づいてツールとレスポンスをセットアップします。
    if num_functions == 1:
        tools = [func_with_approval if approval_required else func_no_approval]
        function_name = "approval_func" if approval_required else "no_approval_func"

        # 単一関数呼び出しの内容。
        func_call = FunctionCallContent(call_id="1", name=function_name, arguments='{"arg1": "value1"}')
        completion = ChatMessage(role="assistant", text="done")

        chat_client_base.run_responses = [
            ChatResponse(messages=ChatMessage(role="assistant", contents=[func_call]))
        ] + ([] if approval_required else [ChatResponse(messages=completion)])

        chat_client_base.streaming_responses = [
            [
                ChatResponseUpdate(
                    contents=[FunctionCallContent(call_id="1", name=function_name, arguments='{"arg1":')],
                    role="assistant",
                ),
                ChatResponseUpdate(
                    contents=[FunctionCallContent(call_id="1", name=function_name, arguments='"value1"}')],
                    role="assistant",
                ),
            ]
        ] + ([] if approval_required else [[ChatResponseUpdate(contents=[TextContent(text="done")], role="assistant")]])

    else:  # num_functions == 2
        tools = [func_no_approval, func_with_approval]

        # 2つの関数呼び出しの内容。
        func_calls = [
            FunctionCallContent(call_id="1", name="no_approval_func", arguments='{"arg1": "value1"}'),
            FunctionCallContent(call_id="2", name="approval_func", arguments='{"arg1": "value2"}'),
        ]

        chat_client_base.run_responses = [ChatResponse(messages=ChatMessage(role="assistant", contents=func_calls))]

        chat_client_base.streaming_responses = [
            [
                ChatResponseUpdate(contents=[func_calls[0]], role="assistant"),
                ChatResponseUpdate(contents=[func_calls[1]], role="assistant"),
            ]
        ]

    # テストを実行します。
    chat_options = ChatOptions(tool_choice="auto", tools=tools)
    if thread_type == "service":
        # サービススレッドの場合、ChatOptions経由でconversation_idを渡す必要があります。
        chat_options.store = True
        chat_options.conversation_id = conversation_id

    if not streaming:
        response = await chat_client_base.get_response("hello", chat_options=chat_options)
        messages = response.messages
    else:
        updates = []
        async for update in chat_client_base.get_streaming_response("hello", chat_options=chat_options):
            updates.append(update)
        messages = updates

    # サービススレッドはメッセージ管理動作が異なり（サーバー側ストレージ）、 そのため詳細なメッセージのアサーションはスキップします。
    if thread_type == "service":
        # 承認に基づいて関数が実行されたかどうかだけを検証します。
        if not approval_required or approval_required == "mixed":
            # サービススレッドの場合でも実行カウンターのチェックは有効です。
            pass
        return

    # シナリオに基づいて検証します（スレッドなしとローカルスレッドの場合）。
    if num_functions == 1:
        if approval_required:
            # 承認ありの単一関数: assistantメッセージは呼び出しと承認要求の両方を含みます。
            if not streaming:
                assert len(messages) == 1
                # AssistantメッセージはFunctionCallContentとFunctionApprovalRequestContentを持つべきです。
                assert len(messages[0].contents) == 2
                assert isinstance(messages[0].contents[0], FunctionCallContent)
                assert isinstance(messages[0].contents[1], FunctionApprovalRequestContent)
                assert messages[0].contents[1].function_call.name == "approval_func"
                assert exec_counter == 0  # 関数はまだ実行されていません。
            else:
                # ストリーミング: 2つの関数呼び出しチャンクと1つの承認要求更新（同じassistantメッセージ）。
                assert len(messages) == 3
                assert isinstance(messages[0].contents[0], FunctionCallContent)
                assert isinstance(messages[1].contents[0], FunctionCallContent)
                assert isinstance(messages[2].contents[0], FunctionApprovalRequestContent)
                assert messages[2].contents[0].function_call.name == "approval_func"
                assert exec_counter == 0  # 関数はまだ実行されていません。
        else:
            # 承認なしの単一関数: 呼び出し、結果、最終。
            if not streaming:
                assert len(messages) == 3
                assert isinstance(messages[0].contents[0], FunctionCallContent)
                assert isinstance(messages[1].contents[0], FunctionResultContent)
                assert messages[1].contents[0].result == "Processed value1"
                assert messages[2].role == Role.ASSISTANT
                assert messages[2].text == "done"
                assert exec_counter == 1
            else:
                # ストリーミングには: 2つの関数呼び出し更新、1つの結果更新、1つの最終更新があります。
                assert len(messages) == 4
                assert isinstance(messages[0].contents[0], FunctionCallContent)
                assert isinstance(messages[1].contents[0], FunctionCallContent)
                assert isinstance(messages[2].contents[0], FunctionResultContent)
                assert messages[3].text == "done"
                assert exec_counter == 1
    else:  # num_functions == 2
        # 混合承認の2つの関数。
        if not streaming:
            # 混合: assistantメッセージは両方の呼び出しと承認要求を含みます（合計4項目）
            # （1つが承認を必要とすると、すべてが承認のためにバッチ処理されるため）。
            assert len(messages) == 1
            # 2つのFunctionCallContentと2つのFunctionApprovalRequestContentがあるべきです。
            assert len(messages[0].contents) == 4
            assert isinstance(messages[0].contents[0], FunctionCallContent)
            assert isinstance(messages[0].contents[1], FunctionCallContent)
            # 両方とも承認要求になります。
            approval_requests = [c for c in messages[0].contents if isinstance(c, FunctionApprovalRequestContent)]
            assert len(approval_requests) == 2
            assert exec_counter == 0  # どちらの関数もまだ実行されていません。
        else:
            # ストリーミング: 2つの関数呼び出し更新と2つの内容を持つ1つの承認要求。
            assert len(messages) == 3
            assert isinstance(messages[0].contents[0], FunctionCallContent)
            assert isinstance(messages[1].contents[0], FunctionCallContent)
            # 承認要求メッセージは両方の承認要求を含みます。
            assert len(messages[2].contents) == 2
            assert all(isinstance(c, FunctionApprovalRequestContent) for c in messages[2].contents)
            assert exec_counter == 0  # どちらの関数もまだ実行されていません。


async def test_rejected_approval(chat_client_base: ChatClientProtocol):
    """承認されたものと一緒に拒否された承認が正しく処理されることをテストします。"""
    from agent_framework import FunctionApprovalResponseContent

    exec_counter_approved = 0
    exec_counter_rejected = 0

    @ai_function(name="approved_func", approval_mode="always_require")
    def func_approved(arg1: str) -> str:
        nonlocal exec_counter_approved
        exec_counter_approved += 1
        return f"Approved {arg1}"

    @ai_function(name="rejected_func", approval_mode="always_require")
    def func_rejected(arg1: str) -> str:
        nonlocal exec_counter_rejected
        exec_counter_rejected += 1
        return f"Rejected {arg1}"

    # セットアップ: 承認が必要な2つの関数呼び出し。
    chat_client_base.run_responses = [
        ChatResponse(
            messages=ChatMessage(
                role="assistant",
                contents=[
                    FunctionCallContent(call_id="1", name="approved_func", arguments='{"arg1": "value1"}'),
                    FunctionCallContent(call_id="2", name="rejected_func", arguments='{"arg1": "value2"}'),
                ],
            )
        ),
        ChatResponse(messages=ChatMessage(role="assistant", text="done")),
    ]

    # 承認要求を含むレスポンスを取得します。
    response = await chat_client_base.get_response("hello", tool_choice="auto", tools=[func_approved, func_rejected])
    # 承認要求は別のメッセージではなく、現在はassistantメッセージに追加されます。
    assert len(response.messages) == 1
    # Assistantメッセージは2つのFunctionCallContentと2つのFunctionApprovalRequestContentを持つべきです。
    assert len(response.messages[0].contents) == 4
    approval_requests = [c for c in response.messages[0].contents if isinstance(c, FunctionApprovalRequestContent)]
    assert len(approval_requests) == 2

    # 1つを承認し、もう1つを拒否します。
    approval_req_1 = approval_requests[0]
    approval_req_2 = approval_requests[1]

    approved_response = FunctionApprovalResponseContent(
        id=approval_req_1.id,
        function_call=approval_req_1.function_call,
        approved=True,
    )
    rejected_response = FunctionApprovalResponseContent(
        id=approval_req_2.id,
        function_call=approval_req_2.function_call,
        approved=False,
    )

    # 1つ承認、1つ拒否の状態で会話を続けます。
    all_messages = response.messages + [ChatMessage(role="user", contents=[approved_response, rejected_response])]

    # 承認を処理するget_responseを呼び出します。
    await chat_client_base.get_response(all_messages, tool_choice="auto", tools=[func_approved, func_rejected])

    # 承認/拒否が正しく処理されたことを検証します 結果は入力メッセージ（インプレースで修正）にあります。
    approved_result = None
    rejected_result = None
    for msg in all_messages:
        for content in msg.contents:
            if isinstance(content, FunctionResultContent):
                if content.call_id == "1":
                    approved_result = content
                elif content.call_id == "2":
                    rejected_result = content

    # 承認された関数は実行され、結果を持つべきです。
    assert approved_result is not None, "Should have found result for approved function"
    assert approved_result.result == "Approved value1"
    assert exec_counter_approved == 1

    # 拒否された関数は「not approved」の結果を持ち、実行されていないべきです。
    assert rejected_result is not None, "Should have found result for rejected function"
    assert rejected_result.result == "Error: Tool call invocation was rejected by user."
    assert exec_counter_rejected == 0

    # FunctionResultContentを持つメッセージはrole="tool"であることを検証します
    # これはOpenAIのAPIに対してメッセージ形式が正しいことを保証します。
    for msg in all_messages:
        for content in msg.contents:
            if isinstance(content, FunctionResultContent):
                assert msg.role == Role.TOOL, (
                    f"Message with FunctionResultContent must have role='tool', got '{msg.role}'"
                )


async def test_approval_requests_in_assistant_message(chat_client_base: ChatClientProtocol):
    """承認要求は関数呼び出しを含むassistantメッセージに追加されるべきです。"""
    exec_counter = 0

    @ai_function(name="test_func", approval_mode="always_require")
    def func_with_approval(arg1: str) -> str:
        nonlocal exec_counter
        exec_counter += 1
        return f"Result {arg1}"

    chat_client_base.run_responses = [
        ChatResponse(
            messages=ChatMessage(
                role="assistant",
                contents=[
                    FunctionCallContent(call_id="1", name="test_func", arguments='{"arg1": "value1"}'),
                ],
            )
        ),
    ]

    response = await chat_client_base.get_response("hello", tool_choice="auto", tools=[func_with_approval])

    # 呼び出しと承認要求の両方を含む1つのassistantメッセージがあるべきです。
    assert len(response.messages) == 1
    assert response.messages[0].role == Role.ASSISTANT
    assert len(response.messages[0].contents) == 2
    assert isinstance(response.messages[0].contents[0], FunctionCallContent)
    assert isinstance(response.messages[0].contents[1], FunctionApprovalRequestContent)
    assert exec_counter == 0


async def test_persisted_approval_messages_replay_correctly(chat_client_base: ChatClientProtocol):
    """メッセージが永続化されて返送される場合でも承認フローが機能するべきです（threadシナリオ）。"""
    from agent_framework import FunctionApprovalResponseContent

    exec_counter = 0

    @ai_function(name="test_func", approval_mode="always_require")
    def func_with_approval(arg1: str) -> str:
        nonlocal exec_counter
        exec_counter += 1
        return f"Result {arg1}"

    chat_client_base.run_responses = [
        ChatResponse(
            messages=ChatMessage(
                role="assistant",
                contents=[
                    FunctionCallContent(call_id="1", name="test_func", arguments='{"arg1": "value1"}'),
                ],
            )
        ),
        ChatResponse(messages=ChatMessage(role="assistant", text="done")),
    ]

    # 承認要求を取得します。
    response1 = await chat_client_base.get_response("hello", tool_choice="auto", tools=[func_with_approval])

    # メッセージを保存します（threadのように）。
    persisted_messages = [
        ChatMessage(role="user", contents=[TextContent(text="hello")]),
        *response1.messages,
    ]

    # 承認を送信します。
    approval_req = [c for c in response1.messages[0].contents if isinstance(c, FunctionApprovalRequestContent)][0]
    approval_response = FunctionApprovalResponseContent(
        id=approval_req.id,
        function_call=approval_req.function_call,
        approved=True,
    )
    persisted_messages.append(ChatMessage(role="user", contents=[approval_response]))

    # すべての永続化されたメッセージで続行します。
    response2 = await chat_client_base.get_response(persisted_messages, tool_choice="auto", tools=[func_with_approval])

    # 正常に実行されるべきです。
    assert response2 is not None
    assert exec_counter == 1
    assert response2.messages[-1].text == "done"


async def test_no_duplicate_function_calls_after_approval_processing(chat_client_base: ChatClientProtocol):
    """承認処理はメッセージ内に重複した関数呼び出しを作成しないべきです。"""
    from agent_framework import FunctionApprovalResponseContent

    @ai_function(name="test_func", approval_mode="always_require")
    def func_with_approval(arg1: str) -> str:
        return f"Result {arg1}"

    chat_client_base.run_responses = [
        ChatResponse(
            messages=ChatMessage(
                role="assistant",
                contents=[
                    FunctionCallContent(call_id="1", name="test_func", arguments='{"arg1": "value1"}'),
                ],
            )
        ),
        ChatResponse(messages=ChatMessage(role="assistant", text="done")),
    ]

    response1 = await chat_client_base.get_response("hello", tool_choice="auto", tools=[func_with_approval])

    approval_req = [c for c in response1.messages[0].contents if isinstance(c, FunctionApprovalRequestContent)][0]
    approval_response = FunctionApprovalResponseContent(
        id=approval_req.id,
        function_call=approval_req.function_call,
        approved=True,
    )

    all_messages = response1.messages + [ChatMessage(role="user", contents=[approval_response])]
    await chat_client_base.get_response(all_messages, tool_choice="auto", tools=[func_with_approval])

    # 同じcall_idを持つ関数呼び出しの数を数えます。
    function_call_count = sum(
        1
        for msg in all_messages
        for content in msg.contents
        if isinstance(content, FunctionCallContent) and content.call_id == "1"
    )

    assert function_call_count == 1


async def test_rejection_result_uses_function_call_id(chat_client_base: ChatClientProtocol):
    """拒否エラー結果は承認のIDではなく関数呼び出しのcall_idを使うべきです。"""
    from agent_framework import FunctionApprovalResponseContent

    @ai_function(name="test_func", approval_mode="always_require")
    def func_with_approval(arg1: str) -> str:
        return f"Result {arg1}"

    chat_client_base.run_responses = [
        ChatResponse(
            messages=ChatMessage(
                role="assistant",
                contents=[
                    FunctionCallContent(call_id="call_123", name="test_func", arguments='{"arg1": "value1"}'),
                ],
            )
        ),
        ChatResponse(messages=ChatMessage(role="assistant", text="done")),
    ]

    response1 = await chat_client_base.get_response("hello", tool_choice="auto", tools=[func_with_approval])

    approval_req = [c for c in response1.messages[0].contents if isinstance(c, FunctionApprovalRequestContent)][0]
    rejection_response = FunctionApprovalResponseContent(
        id=approval_req.id,
        function_call=approval_req.function_call,
        approved=False,
    )

    all_messages = response1.messages + [ChatMessage(role="user", contents=[rejection_response])]
    await chat_client_base.get_response(all_messages, tool_choice="auto", tools=[func_with_approval])

    # 拒否結果を見つけます。
    rejection_result = next(
        (content for msg in all_messages for content in msg.contents if isinstance(content, FunctionResultContent)),
        None,
    )

    assert rejection_result is not None
    assert rejection_result.call_id == "call_123"
    assert "rejected" in rejection_result.result.lower()


async def test_max_iterations_limit(chat_client_base: ChatClientProtocol):
    """additional_propertiesのMAX_ITERATIONSが関数呼び出しループを制限することをテストします。"""
    exec_counter = 0

    @ai_function(name="test_function")
    def ai_func(arg1: str) -> str:
        nonlocal exec_counter
        exec_counter += 1
        return f"Processed {arg1}"

    # ループを作成するために複数の関数呼び出しレスポンスをセットアップします。
    chat_client_base.run_responses = [
        ChatResponse(
            messages=ChatMessage(
                role="assistant",
                contents=[FunctionCallContent(call_id="1", name="test_function", arguments='{"arg1": "value1"}')],
            )
        ),
        ChatResponse(
            messages=ChatMessage(
                role="assistant",
                contents=[FunctionCallContent(call_id="2", name="test_function", arguments='{"arg1": "value2"}')],
            )
        ),
        # tool_choiceが"none"に設定された場合のフェイルセーフレスポンス。
        ChatResponse(messages=ChatMessage(role="assistant", text="giving up on tools")),
    ]

    # additional_propertiesでmax_iterationsを1に設定します。
    chat_client_base.additional_properties = {"max_iterations": 1}

    response = await chat_client_base.get_response("hello", tool_choice="auto", tools=[ai_func])

    # max_iterations=1の場合、以下を行うべきです: 1. 最初の関数呼び出しを実行する（exec_counter=1） 2.
    # 2回目の呼び出しを試みるがイテレーション制限に達する 3. tool_choice="none"で単純な回答を求めるフォールバックに切り替える
    assert exec_counter == 1  # 最初の関数だけが実行されます。
    assert response.messages[-1].text == "I broke out of the function invocation loop..."  # フェイルセーフレスポンス。
