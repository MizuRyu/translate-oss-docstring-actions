# Copyright (c) Microsoft. All rights reserved.

from collections.abc import AsyncIterable
from typing import Any

import pytest
from pydantic import BaseModel
from pytest import fixture, mark, raises

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    BaseContent,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    CitationAnnotation,
    DataContent,
    ErrorContent,
    FinishReason,
    FunctionApprovalRequestContent,
    FunctionApprovalResponseContent,
    FunctionCallContent,
    FunctionResultContent,
    HostedFileContent,
    HostedVectorStoreContent,
    Role,
    TextContent,
    TextReasoningContent,
    TextSpanRegion,
    ToolMode,
    ToolProtocol,
    UriContent,
    UsageContent,
    UsageDetails,
    ai_function,
)
from agent_framework.exceptions import AdditionItemMismatch, ContentError


@fixture
def ai_tool() -> ToolProtocol:
    """汎用のToolProtocolを返します。"""

    class GenericTool(BaseModel):
        name: str
        description: str | None = None
        additional_properties: dict[str, Any] | None = None

        def parameters(self) -> dict[str, Any]:
            """ツールのパラメータをJSONスキーマとして返します。"""
            return {
                "name": {"type": "string"},
            }

    return GenericTool(name="generic_tool", description="A generic tool")


@fixture
def ai_function_tool() -> ToolProtocol:
    """実行可能なToolProtocolを返します。"""

    @ai_function
    def simple_function(x: int, y: int) -> int:
        """2つの数値を加算する単純な関数です。"""
        return x + y

    return simple_function


# region TextContent


def test_text_content_positional():
    """TextContentクラスが正しく初期化され、BaseContentから継承していることをテストします。"""
    # TextContentのインスタンスを作成します。
    content = TextContent("Hello, world!", raw_representation="Hello, world!", additional_properties={"version": 1})

    # 型と内容をチェックします。
    assert content.type == "text"
    assert content.text == "Hello, world!"
    assert content.raw_representation == "Hello, world!"
    assert content.additional_properties["version"] == 1
    # インスタンスがBaseContentの型であることを確認します。
    assert isinstance(content, BaseContent)
    # 注：もはやPydanticのバリデーションを使用していないため、型の割り当ては動作するはずです。
    content.type = "text"  # これで問題なく動作するはずです。


def test_text_content_keyword():
    """TextContentクラスが正しく初期化され、BaseContentから継承していることをテストします。"""
    # TextContentのインスタンスを作成します。
    content = TextContent(
        text="Hello, world!", raw_representation="Hello, world!", additional_properties={"version": 1}
    )

    # 型と内容をチェックします。
    assert content.type == "text"
    assert content.text == "Hello, world!"
    assert content.raw_representation == "Hello, world!"
    assert content.additional_properties["version"] == 1
    # インスタンスがBaseContentの型であることを確認します。
    assert isinstance(content, BaseContent)
    # 注意: もはやPydanticのバリデーションを使用していないため、型の割り当ては機能するはずです
    content.type = "text"  # これで問題なく動作するはずです


# region DataContent


def test_data_content_bytes():
    """DataContentクラスが正しく初期化されることをテストします。"""
    # DataContentのインスタンスを作成する
    content = DataContent(data=b"test", media_type="application/octet-stream", additional_properties={"version": 1})

    # 型と内容を確認する
    assert content.type == "data"
    assert content.uri == "data:application/octet-stream;base64,dGVzdA=="
    assert content.has_top_level_media_type("application") is True
    assert content.has_top_level_media_type("image") is False
    assert content.additional_properties["version"] == 1

    # インスタンスがBaseContent型であることを確認する
    assert isinstance(content, BaseContent)


def test_data_content_uri():
    """URIで正しく初期化されることをDataContentクラスでテストします。"""
    # URIを使ってDataContentのインスタンスを作成する
    content = DataContent(uri="data:application/octet-stream;base64,dGVzdA==", additional_properties={"version": 1})

    # 型と内容を確認する
    assert content.type == "data"
    assert content.uri == "data:application/octet-stream;base64,dGVzdA=="
    # media_typeは現在URIから抽出される
    assert content.media_type == "application/octet-stream"
    assert content.has_top_level_media_type("application") is True
    assert content.additional_properties["version"] == 1

    # インスタンスがBaseContent型であることを確認する
    assert isinstance(content, BaseContent)


def test_data_content_invalid():
    """無効な初期化でエラーが発生することをDataContentクラスでテストします。"""
    # 無効なデータでDataContentのインスタンス作成を試みる 正しいURIではない
    with raises(ValueError):
        DataContent(uri="invalid_uri")
    # 不明なメディアタイプ
    with raises(ValueError):
        DataContent(uri="data:application/random;base64,dGVzdA==")
    # 有効なbase64データでないものも基本的なバリデーションでは受け入れられるが 現時点では重大な問題ではない


def test_data_content_empty():
    """空のデータでエラーが発生することをDataContentクラスでテストします。"""
    # 空のデータでDataContentのインスタンス作成を試みる
    with raises(ValueError):
        DataContent(data=b"", media_type="application/octet-stream")

    # 空のURIでDataContentのインスタンス作成を試みる
    with raises(ValueError):
        DataContent(uri="")


# region UriContent


def test_uri_content():
    """UriContentクラスが正しく初期化されることをテストします。"""
    content = UriContent(uri="http://example.com", media_type="image/jpg", additional_properties={"version": 1})

    # 型と内容を確認する
    assert content.type == "uri"
    assert content.uri == "http://example.com"
    assert content.media_type == "image/jpg"
    assert content.has_top_level_media_type("image") is True
    assert content.has_top_level_media_type("application") is False
    assert content.additional_properties["version"] == 1

    # インスタンスがBaseContent型であることを確認する
    assert isinstance(content, BaseContent)


# region: HostedFileContent


def test_hosted_file_content():
    """HostedFileContentクラスが正しく初期化されることをテストします。"""
    content = HostedFileContent(file_id="file-123", additional_properties={"version": 1})

    # 型と内容を確認する
    assert content.type == "hosted_file"
    assert content.file_id == "file-123"
    assert content.additional_properties["version"] == 1

    # インスタンスがBaseContent型であることを確認する
    assert isinstance(content, BaseContent)


def test_hosted_file_content_minimal():
    """最小限のパラメータでHostedFileContentクラスをテストします。"""
    content = HostedFileContent(file_id="file-456")

    # 型と内容を確認する
    assert content.type == "hosted_file"
    assert content.file_id == "file-456"
    assert content.additional_properties == {}
    assert content.raw_representation is None

    # インスタンスがBaseContent型であることを確認する
    assert isinstance(content, BaseContent)


# region: HostedVectorStoreContent


def test_hosted_vector_store_content():
    """HostedVectorStoreContentクラスが正しく初期化されることをテストします。"""
    content = HostedVectorStoreContent(vector_store_id="vs-789", additional_properties={"version": 1})

    # 型と内容を確認する
    assert content.type == "hosted_vector_store"
    assert content.vector_store_id == "vs-789"
    assert content.additional_properties["version"] == 1

    # インスタンスがBaseContent型であることを確認する
    assert isinstance(content, HostedVectorStoreContent)
    assert isinstance(content, BaseContent)


def test_hosted_vector_store_content_minimal():
    """最小限のパラメータでHostedVectorStoreContentクラスをテストします。"""
    content = HostedVectorStoreContent(vector_store_id="vs-101112")

    # 型と内容を確認する
    assert content.type == "hosted_vector_store"
    assert content.vector_store_id == "vs-101112"
    assert content.additional_properties == {}
    assert content.raw_representation is None

    # インスタンスがBaseContent型であることを確認する
    assert isinstance(content, HostedVectorStoreContent)
    assert isinstance(content, BaseContent)


# region FunctionCallContent


def test_function_call_content():
    """FunctionCallContentクラスが正しく初期化されることをテストします。"""
    content = FunctionCallContent(call_id="1", name="example_function", arguments={"param1": "value1"})

    # 型と内容を確認する
    assert content.type == "function_call"
    assert content.name == "example_function"
    assert content.arguments == {"param1": "value1"}

    # インスタンスがBaseContent型であることを確認する
    assert isinstance(content, BaseContent)


def test_function_call_content_parse_arguments():
    c1 = FunctionCallContent(call_id="1", name="f", arguments='{"a": 1, "b": 2}')
    assert c1.parse_arguments() == {"a": 1, "b": 2}
    c2 = FunctionCallContent(call_id="1", name="f", arguments="not json")
    assert c2.parse_arguments() == {"raw": "not json"}
    c3 = FunctionCallContent(call_id="1", name="f", arguments={"x": None})
    assert c3.parse_arguments() == {"x": None}


def test_function_call_content_add_merging_and_errors():
    # str + strの連結
    a = FunctionCallContent(call_id="1", name="f", arguments="abc")
    b = FunctionCallContent(call_id="1", name="f", arguments="def")
    c = a + b
    assert isinstance(c.arguments, str) and c.arguments == "abcdef"

    # dict + dictのマージ
    a = FunctionCallContent(call_id="1", name="f", arguments={"x": 1})
    b = FunctionCallContent(call_id="1", name="f", arguments={"y": 2})
    c = a + b
    assert c.arguments == {"x": 1, "y": 2}

    # 互換性のない引数の型
    a = FunctionCallContent(call_id="1", name="f", arguments="abc")
    b = FunctionCallContent(call_id="1", name="f", arguments={"y": 2})
    with raises(TypeError):
        _ = a + b

    # 互換性のないcall id
    a = FunctionCallContent(call_id="1", name="f", arguments="abc")
    b = FunctionCallContent(call_id="2", name="f", arguments="def")

    with raises(AdditionItemMismatch):
        _ = a + b


# region FunctionResultContent


def test_function_result_content():
    """FunctionResultContentクラスが正しく初期化されることをテストします。"""
    content = FunctionResultContent(call_id="1", result={"param1": "value1"})

    # 型と内容を確認する
    assert content.type == "function_result"
    assert content.result == {"param1": "value1"}

    # インスタンスがBaseContent型であることを確認する
    assert isinstance(content, BaseContent)


# region UsageDetails


def test_usage_details():
    usage = UsageDetails(input_token_count=5, output_token_count=10, total_token_count=15)
    assert usage.input_token_count == 5
    assert usage.output_token_count == 10
    assert usage.total_token_count == 15
    assert usage.additional_counts == {}


def test_usage_details_addition():
    usage1 = UsageDetails(
        input_token_count=5,
        output_token_count=10,
        total_token_count=15,
        test1=10,
        test2=20,
    )
    usage2 = UsageDetails(
        input_token_count=3,
        output_token_count=6,
        total_token_count=9,
        test1=10,
        test3=30,
    )

    combined_usage = usage1 + usage2
    assert combined_usage.input_token_count == 8
    assert combined_usage.output_token_count == 16
    assert combined_usage.total_token_count == 24
    assert combined_usage.additional_counts["test1"] == 20
    assert combined_usage.additional_counts["test2"] == 20
    assert combined_usage.additional_counts["test3"] == 30


def test_usage_details_fail():
    with raises(ValueError):
        UsageDetails(input_token_count=5, output_token_count=10, total_token_count=15, wrong_type="42.923")


def test_usage_details_additional_counts():
    usage = UsageDetails(input_token_count=5, output_token_count=10, total_token_count=15, **{"test": 1})
    assert usage.additional_counts["test"] == 1


def test_usage_details_add_with_none_and_type_errors():
    u = UsageDetails(input_token_count=1)
    # __add__にNoneを渡すとselfを返す（変更なし）
    v = u + None
    assert v is u
    # __iadd__にNoneを渡すと変更なし
    u2 = UsageDetails(input_token_count=2)
    u2 += None
    assert u2.input_token_count == 2
    # 型が間違っていると例外を発生させる
    with raises(ValueError):
        _ = u + 42  # type: ignore[arg-type]
    with raises(ValueError):
        u += 42  # type: ignore[arg-type]


# region UserInputRequest and Response


def test_function_approval_request_and_response_creation():
    """FunctionApprovalRequestContentの作成とレスポンス生成をテストします。"""
    fc = FunctionCallContent(call_id="call-1", name="do_something", arguments={"a": 1})
    req = FunctionApprovalRequestContent(id="req-1", function_call=fc)

    assert req.type == "function_approval_request"
    assert req.function_call == fc
    assert req.id == "req-1"
    assert isinstance(req, BaseContent)

    resp = req.create_response(True)

    assert isinstance(resp, FunctionApprovalResponseContent)
    assert resp.approved is True
    assert resp.function_call == fc
    assert resp.id == "req-1"


def test_function_approval_serialization_roundtrip():
    fc = FunctionCallContent(call_id="c2", name="f", arguments='{"x":1}')
    req = FunctionApprovalRequestContent(id="id-2", function_call=fc, additional_properties={"meta": 1})

    dumped = req.to_dict()
    loaded = FunctionApprovalRequestContent.from_dict(dumped)

    # 基本的なプロパティが一致することをテストします
    assert loaded.id == req.id
    assert loaded.additional_properties == req.additional_properties
    assert loaded.function_call.call_id == req.function_call.call_id
    assert loaded.function_call.name == req.function_call.name
    assert loaded.function_call.arguments == req.function_call.arguments

    # Pydanticを使わなくなったためBaseModelのバリデーションテストはスキップします Contentsのunionは完全移行時に別の方法で扱う必要があります


# region BaseContent Serialization


@mark.parametrize(
    "content_type, args",
    [
        (TextContent, {"text": "Hello, world!"}),
        (DataContent, {"data": b"Hello, world!", "media_type": "text/plain"}),
        (UriContent, {"uri": "http://example.com", "media_type": "text/html"}),
        (FunctionCallContent, {"call_id": "1", "name": "example_function", "arguments": {}}),
        (FunctionResultContent, {"call_id": "1", "result": {}}),
        (HostedFileContent, {"file_id": "file-123"}),
        (HostedVectorStoreContent, {"vector_store_id": "vs-789"}),
    ],
)
def test_ai_content_serialization(content_type: type[BaseContent], args: dict):
    content = content_type(**args)
    serialized = content.to_dict()
    deserialized = content_type.from_dict(serialized)
    # 注意: Pydanticを使わなくなったため直接の等価比較はできません 代わりにデシリアライズしたオブジェクトが同じ属性を持つか確認します
    # 元の'data'パラメータを公開しないDataContentは特別扱いします
    if content_type == DataContent and "data" in args:
        # dataで作成されたDataContentは代わりにuriとmedia_typeをチェックします
        assert hasattr(deserialized, "uri")
        assert hasattr(deserialized, "media_type")
        assert deserialized.media_type == args["media_type"]  # type: ignore
        # 'data'属性はuriに変換されるためチェックをスキップします
        for key, value in args.items():
            if key != "data":  # Skip the 'data' key for DataContent
                assert getattr(deserialized, key) == value
    else:
        # 他のコンテンツタイプは通常の属性チェックを行います
        for key, value in args.items():
            if value:
                assert getattr(deserialized, key) == value

    # 現時点ではTestModelのバリデーションはスキップします（まだPydanticを使用しているため） 移行時に更新が必要です class
    # TestModel(BaseModel): content: Contents  test_item =
    # TestModel.model_validate({"content": serialized}) assert
    # isinstance(test_item.content, content_type)


# region ChatMessage


def test_chat_message_text():
    """ChatMessageクラスがテキストコンテンツで正しく初期化されることをテストします。"""
    # 役割とテキストコンテンツでChatMessageを作成する
    message = ChatMessage(role="user", text="Hello, how are you?")

    # 型と内容を確認する
    assert message.role == Role.USER
    assert len(message.contents) == 1
    assert isinstance(message.contents[0], TextContent)
    assert message.contents[0].text == "Hello, how are you?"
    assert message.text == "Hello, how are you?"

    # インスタンスがBaseContent型であることを確認する
    assert isinstance(message.contents[0], BaseContent)


def test_chat_message_contents():
    """ChatMessageクラスが複数のコンテンツで正しく初期化されることをテストします。"""
    # 役割と複数のコンテンツでChatMessageを作成する
    content1 = TextContent("Hello, how are you?")
    content2 = TextContent("I'm fine, thank you!")
    message = ChatMessage(role="user", contents=[content1, content2])

    # 型と内容を確認する
    assert message.role == Role.USER
    assert len(message.contents) == 2
    assert isinstance(message.contents[0], TextContent)
    assert isinstance(message.contents[1], TextContent)
    assert message.contents[0].text == "Hello, how are you?"
    assert message.contents[1].text == "I'm fine, thank you!"
    assert message.text == "Hello, how are you? I'm fine, thank you!"


def test_chat_message_with_chatrole_instance():
    m = ChatMessage(role=Role.USER, text="hi")
    assert m.role == Role.USER
    assert m.text == "hi"


# region ChatResponse


def test_chat_response():
    """ChatResponseクラスがメッセージで正しく初期化されることをテストします。"""
    # ChatMessageを作成する
    message = ChatMessage(role="assistant", text="I'm doing well, thank you!")

    # メッセージでChatResponseを作成する
    response = ChatResponse(messages=message)

    # 型と内容を確認する
    assert response.messages[0].role == Role.ASSISTANT
    assert response.messages[0].text == "I'm doing well, thank you!"
    assert isinstance(response.messages[0], ChatMessage)
    # __str__はテキストを返す
    assert str(response) == response.text


class OutputModel(BaseModel):
    response: str


def test_chat_response_with_format():
    """ChatResponseクラスがメッセージで正しく初期化されることをテストします。"""
    # ChatMessageを作成する
    message = ChatMessage(role="assistant", text='{"response": "Hello"}')

    # メッセージでChatResponseを作成する
    response = ChatResponse(messages=message)

    # 型と内容を確認する
    assert response.messages[0].role == Role.ASSISTANT
    assert response.messages[0].text == '{"response": "Hello"}'
    assert isinstance(response.messages[0], ChatMessage)
    assert response.text == '{"response": "Hello"}'
    assert response.value is None
    response.try_parse_value(OutputModel)
    assert response.value is not None
    assert response.value.response == "Hello"


def test_chat_response_with_format_init():
    """ChatResponseクラスがメッセージで正しく初期化されることをテストします。"""
    # ChatMessageを作成する
    message = ChatMessage(role="assistant", text='{"response": "Hello"}')

    # メッセージでChatResponseを作成する
    response = ChatResponse(messages=message, response_format=OutputModel)

    # 型と内容を確認する
    assert response.messages[0].role == Role.ASSISTANT
    assert response.messages[0].text == '{"response": "Hello"}'
    assert isinstance(response.messages[0], ChatMessage)
    assert response.text == '{"response": "Hello"}'
    assert response.value is not None
    assert response.value.response == "Hello"


# region ChatResponseUpdate


def test_chat_response_update():
    """ChatResponseUpdateクラスがメッセージで正しく初期化されることをテストします。"""
    # ChatMessageを作成する
    message = TextContent(text="I'm doing well, thank you!")

    # メッセージでChatResponseUpdateを作成する
    response_update = ChatResponseUpdate(contents=[message])

    # 型と内容を確認する
    assert response_update.contents[0].text == "I'm doing well, thank you!"
    assert isinstance(response_update.contents[0], TextContent)
    assert response_update.text == "I'm doing well, thank you!"


def test_chat_response_updates_to_chat_response_one():
    """ChatResponseUpdateをChatResponseに変換することをテストします。"""
    # ChatMessageを作成する
    message1 = TextContent("I'm doing well, ")
    message2 = TextContent("thank you!")

    # メッセージでChatResponseUpdateを作成する
    response_updates = [
        ChatResponseUpdate(text=message1, message_id="1"),
        ChatResponseUpdate(text=message2, message_id="1"),
    ]

    # ChatResponseに変換する
    chat_response = ChatResponse.from_chat_response_updates(response_updates)

    # 型と内容を確認する
    assert len(chat_response.messages) == 1
    assert chat_response.text == "I'm doing well, thank you!"
    assert isinstance(chat_response.messages[0], ChatMessage)
    assert len(chat_response.messages[0].contents) == 1
    assert chat_response.messages[0].message_id == "1"


def test_chat_response_updates_to_chat_response_two():
    """ChatResponseUpdateをChatResponseに変換することをテストします。"""
    # ChatMessageを作成する
    message1 = TextContent("I'm doing well, ")
    message2 = TextContent("thank you!")

    # メッセージでChatResponseUpdateを作成する
    response_updates = [
        ChatResponseUpdate(text=message1, message_id="1"),
        ChatResponseUpdate(text=message2, message_id="2"),
    ]

    # ChatResponseに変換する
    chat_response = ChatResponse.from_chat_response_updates(response_updates)

    # 型と内容を確認する
    assert len(chat_response.messages) == 2
    assert chat_response.text == "I'm doing well, \nthank you!"
    assert isinstance(chat_response.messages[0], ChatMessage)
    assert chat_response.messages[0].message_id == "1"
    assert isinstance(chat_response.messages[1], ChatMessage)
    assert chat_response.messages[1].message_id == "2"


def test_chat_response_updates_to_chat_response_multiple():
    """ChatResponseUpdateをChatResponseに変換することをテストします。"""
    # ChatMessageを作成する
    message1 = TextContent("I'm doing well, ")
    message2 = TextContent("thank you!")

    # メッセージでChatResponseUpdateを作成する
    response_updates = [
        ChatResponseUpdate(text=message1, message_id="1"),
        ChatResponseUpdate(contents=[TextReasoningContent(text="Additional context")], message_id="1"),
        ChatResponseUpdate(text=message2, message_id="1"),
    ]

    # ChatResponseに変換する
    chat_response = ChatResponse.from_chat_response_updates(response_updates)

    # 型と内容を確認する
    assert len(chat_response.messages) == 1
    assert chat_response.text == "I'm doing well,  thank you!"
    assert isinstance(chat_response.messages[0], ChatMessage)
    assert len(chat_response.messages[0].contents) == 3
    assert chat_response.messages[0].message_id == "1"


def test_chat_response_updates_to_chat_response_multiple_multiple():
    """ChatResponseUpdateをChatResponseに変換することをテストします。"""
    # ChatMessageを作成する
    message1 = TextContent("I'm doing well, ", raw_representation="I'm doing well, ")
    message2 = TextContent("thank you!")

    # メッセージでChatResponseUpdateを作成する
    response_updates = [
        ChatResponseUpdate(text=message1, message_id="1"),
        ChatResponseUpdate(text=message2, message_id="1"),
        ChatResponseUpdate(contents=[TextReasoningContent(text="Additional context")], message_id="1"),
        ChatResponseUpdate(contents=[TextContent(text="More context")], message_id="1"),
        ChatResponseUpdate(text="Final part", message_id="1"),
    ]

    # ChatResponseに変換する
    chat_response = ChatResponse.from_chat_response_updates(response_updates)

    # 型と内容を確認する
    assert len(chat_response.messages) == 1
    assert isinstance(chat_response.messages[0], ChatMessage)
    assert chat_response.messages[0].message_id == "1"
    assert chat_response.messages[0].contents[0].raw_representation is not None

    assert len(chat_response.messages[0].contents) == 3
    assert isinstance(chat_response.messages[0].contents[0], TextContent)
    assert chat_response.messages[0].contents[0].text == "I'm doing well, thank you!"
    assert isinstance(chat_response.messages[0].contents[1], TextReasoningContent)
    assert chat_response.messages[0].contents[1].text == "Additional context"
    assert isinstance(chat_response.messages[0].contents[2], TextContent)
    assert chat_response.messages[0].contents[2].text == "More contextFinal part"

    assert chat_response.text == "I'm doing well, thank you! More contextFinal part"


async def test_chat_response_from_async_generator():
    async def gen() -> AsyncIterable[ChatResponseUpdate]:
        yield ChatResponseUpdate(text="Hello", message_id="1")
        yield ChatResponseUpdate(text=" world", message_id="1")

    resp = await ChatResponse.from_chat_response_generator(gen())
    assert resp.text == "Hello world"


async def test_chat_response_from_async_generator_output_format():
    async def gen() -> AsyncIterable[ChatResponseUpdate]:
        yield ChatResponseUpdate(text='{ "respon', message_id="1")
        yield ChatResponseUpdate(text='se": "Hello" }', message_id="1")

    resp = await ChatResponse.from_chat_response_generator(gen())
    assert resp.text == '{ "response": "Hello" }'
    assert resp.value is None
    resp.try_parse_value(OutputModel)
    assert resp.value is not None
    assert resp.value.response == "Hello"


async def test_chat_response_from_async_generator_output_format_in_method():
    async def gen() -> AsyncIterable[ChatResponseUpdate]:
        yield ChatResponseUpdate(text='{ "respon', message_id="1")
        yield ChatResponseUpdate(text='se": "Hello" }', message_id="1")

    resp = await ChatResponse.from_chat_response_generator(gen(), output_format_type=OutputModel)
    assert resp.text == '{ "response": "Hello" }'
    assert resp.value is not None
    assert resp.value.response == "Hello"


# region ToolMode


def test_chat_tool_mode():
    """ToolModeクラスが正しく初期化されることをテストします。"""
    # ToolModeのインスタンスを作成する
    auto_mode = ToolMode.AUTO
    required_any = ToolMode.REQUIRED_ANY
    required_mode = ToolMode.REQUIRED("example_function")
    none_mode = ToolMode.NONE

    # 型と内容を確認する
    assert auto_mode.mode == "auto"
    assert auto_mode.required_function_name is None
    assert required_any.mode == "required"
    assert required_any.required_function_name is None
    assert required_mode.mode == "required"
    assert required_mode.required_function_name == "example_function"
    assert none_mode.mode == "none"
    assert none_mode.required_function_name is None

    # インスタンスがToolMode型であることを確認する
    assert isinstance(auto_mode, ToolMode)
    assert isinstance(required_any, ToolMode)
    assert isinstance(required_mode, ToolMode)
    assert isinstance(none_mode, ToolMode)

    assert ToolMode.REQUIRED("example_function") == ToolMode.REQUIRED("example_function")
    # serializerはmodeのみを返す
    assert ToolMode.REQUIRED_ANY.serialize_model() == "required"


def test_chat_tool_mode_from_dict():
    """辞書からToolModeを作成することをテストします。"""
    mode_dict = {"mode": "required", "required_function_name": "example_function"}
    mode = ToolMode(**mode_dict)

    # 型と内容を確認する
    assert mode.mode == "required"
    assert mode.required_function_name == "example_function"

    # インスタンスがToolMode型であることを確認する
    assert isinstance(mode, ToolMode)


# region ChatOptions


def test_chat_options_init() -> None:
    options = ChatOptions()
    assert options.model_id is None


def test_chat_options_tool_choice_validation_errors():
    with raises((ContentError, TypeError)):
        ChatOptions(tool_choice="invalid-choice")


def test_chat_options_and(ai_function_tool, ai_tool) -> None:
    options1 = ChatOptions(model_id="gpt-4o", tools=[ai_function_tool], logit_bias={"x": 1}, metadata={"a": "b"})
    options2 = ChatOptions(model_id="gpt-4.1", tools=[ai_tool], additional_properties={"p": 1})
    assert options1 != options2
    options3 = options1 & options2

    assert options3.model_id == "gpt-4.1"
    assert options3.tools == [ai_function_tool, ai_tool]
    assert options3.logit_bias == {"x": 1}
    assert options3.metadata == {"a": "b"}
    assert options3.additional_properties.get("p") == 1


# region Agent Response Fixtures


@fixture
def chat_message() -> ChatMessage:
    return ChatMessage(role=Role.USER, text="Hello")


@fixture
def text_content() -> TextContent:
    return TextContent(text="Test content")


@fixture
def agent_run_response(chat_message: ChatMessage) -> AgentRunResponse:
    return AgentRunResponse(messages=chat_message)


@fixture
def agent_run_response_update(text_content: TextContent) -> AgentRunResponseUpdate:
    return AgentRunResponseUpdate(role=Role.ASSISTANT, contents=[text_content])


# region AgentRunResponse


def test_agent_run_response_init_single_message(chat_message: ChatMessage) -> None:
    response = AgentRunResponse(messages=chat_message)
    assert response.messages == [chat_message]


def test_agent_run_response_init_list_messages(chat_message: ChatMessage) -> None:
    response = AgentRunResponse(messages=[chat_message, chat_message])
    assert len(response.messages) == 2
    assert response.messages[0] == chat_message


def test_agent_run_response_init_none_messages() -> None:
    response = AgentRunResponse()
    assert response.messages == []


def test_agent_run_response_text_property(chat_message: ChatMessage) -> None:
    response = AgentRunResponse(messages=[chat_message, chat_message])
    assert response.text == "HelloHello"


def test_agent_run_response_text_property_empty() -> None:
    response = AgentRunResponse()
    assert response.text == ""


def test_agent_run_response_from_updates(agent_run_response_update: AgentRunResponseUpdate) -> None:
    updates = [agent_run_response_update, agent_run_response_update]
    response = AgentRunResponse.from_agent_run_response_updates(updates)
    assert len(response.messages) > 0
    assert response.text == "Test contentTest content"


def test_agent_run_response_str_method(chat_message: ChatMessage) -> None:
    response = AgentRunResponse(messages=chat_message)
    assert str(response) == "Hello"


# region AgentRunResponseUpdate


def test_agent_run_response_update_init_content_list(text_content: TextContent) -> None:
    update = AgentRunResponseUpdate(contents=[text_content, text_content])
    assert len(update.contents) == 2
    assert update.contents[0] == text_content


def test_agent_run_response_update_init_none_content() -> None:
    update = AgentRunResponseUpdate()
    assert update.contents == []


def test_agent_run_response_update_text_property(text_content: TextContent) -> None:
    update = AgentRunResponseUpdate(contents=[text_content, text_content])
    assert update.text == "Test contentTest content"


def test_agent_run_response_update_text_property_empty() -> None:
    update = AgentRunResponseUpdate()
    assert update.text == ""


def test_agent_run_response_update_str_method(text_content: TextContent) -> None:
    update = AgentRunResponseUpdate(contents=[text_content])
    assert str(update) == "Test content"


# region ErrorContent


def test_error_content_str():
    e1 = ErrorContent(message="Oops", error_code="E1")
    assert str(e1) == "Error E1: Oops"
    e2 = ErrorContent(message="Oops")
    assert str(e2) == "Oops"
    e3 = ErrorContent()
    assert str(e3) == "Unknown error"


# region アノテーション


def test_annotations_models_and_roundtrip():
    span = TextSpanRegion(start_index=0, end_index=5)
    cit = CitationAnnotation(title="Doc", url="http://example.com", snippet="Snippet", annotated_regions=[span])

    # コンテンツにアタッチ
    content = TextContent(text="hello", additional_properties={"v": 1})
    content.annotations = [cit]

    dumped = content.to_dict()
    loaded = TextContent.from_dict(dumped)
    assert isinstance(loaded.annotations, list)
    assert len(loaded.annotations) == 1
    # Pydanticからの移行後、アノテーションはオブジェクトとして適切に再構築されるべきです
    assert isinstance(loaded.annotations[0], CitationAnnotation)
    # アノテーションのプロパティをチェック
    loaded_cit = loaded.annotations[0]
    assert loaded_cit.type == "citation"
    assert loaded_cit.title == "Doc"
    assert loaded_cit.url == "http://example.com"
    assert loaded_cit.snippet == "Snippet"
    # annotated_regionsをチェック
    assert isinstance(loaded_cit.annotated_regions, list)
    assert len(loaded_cit.annotated_regions) == 1
    assert isinstance(loaded_cit.annotated_regions[0], TextSpanRegion)
    assert loaded_cit.annotated_regions[0].type == "text_span"
    assert loaded_cit.annotated_regions[0].start_index == 0
    assert loaded_cit.annotated_regions[0].end_index == 5


def test_function_call_merge_in_process_update_and_usage_aggregation():
    # 同じcall_idを持つ2つの関数呼び出しチャンクはマージされるべきです
    u1 = ChatResponseUpdate(contents=[FunctionCallContent(call_id="c1", name="f", arguments="{")], message_id="m")
    u2 = ChatResponseUpdate(contents=[FunctionCallContent(call_id="c1", name="f", arguments="}")], message_id="m")
    # plusの使用法
    u3 = ChatResponseUpdate(contents=[UsageContent(UsageDetails(input_token_count=1, output_token_count=2))])

    resp = ChatResponse.from_chat_response_updates([u1, u2, u3])
    assert len(resp.messages) == 1
    last_contents = resp.messages[0].contents
    assert any(isinstance(c, FunctionCallContent) for c in last_contents)
    fcs = [c for c in last_contents if isinstance(c, FunctionCallContent)]
    assert len(fcs) == 1
    assert fcs[0].arguments == "{}"
    assert resp.usage_details is not None
    assert resp.usage_details.input_token_count == 1
    assert resp.usage_details.output_token_count == 2


def test_function_call_incompatible_ids_are_not_merged():
    u1 = ChatResponseUpdate(contents=[FunctionCallContent(call_id="a", name="f", arguments="x")], message_id="m")
    u2 = ChatResponseUpdate(contents=[FunctionCallContent(call_id="b", name="f", arguments="y")], message_id="m")

    resp = ChatResponse.from_chat_response_updates([u1, u2])
    fcs = [c for c in resp.messages[0].contents if isinstance(c, FunctionCallContent)]
    assert len(fcs) == 2


# region Role & FinishReasonの基本


def test_chat_role_str_and_repr():
    assert str(Role.USER) == "user"
    assert "Role(value=" in repr(Role.USER)


def test_chat_finish_reason_constants():
    assert FinishReason.STOP.value == "stop"


def test_response_update_propagates_fields_and_metadata():
    upd = ChatResponseUpdate(
        text="hello",
        role="assistant",
        author_name="bot",
        response_id="rid",
        message_id="mid",
        conversation_id="cid",
        model_id="model-x",
        created_at="t0",
        finish_reason=FinishReason.STOP,
        additional_properties={"k": "v"},
    )
    resp = ChatResponse.from_chat_response_updates([upd])
    assert resp.response_id == "rid"
    assert resp.created_at == "t0"
    assert resp.conversation_id == "cid"
    assert resp.model_id == "model-x"
    assert resp.finish_reason == FinishReason.STOP
    assert resp.additional_properties and resp.additional_properties["k"] == "v"
    assert resp.messages[0].role == Role.ASSISTANT
    assert resp.messages[0].author_name == "bot"
    assert resp.messages[0].message_id == "mid"


def test_text_coalescing_preserves_first_properties():
    t1 = TextContent("A", raw_representation={"r": 1}, additional_properties={"p": 1})
    t2 = TextContent("B")
    upd1 = ChatResponseUpdate(text=t1, message_id="x")
    upd2 = ChatResponseUpdate(text=t2, message_id="x")
    resp = ChatResponse.from_chat_response_updates([upd1, upd2])
    # 結合後は、テキストがマージされ、最初のものからプロパティが保持された単一のTextContentが存在するべきです
    items = [c for c in resp.messages[0].contents if isinstance(c, TextContent)]
    assert len(items) >= 1
    assert items[0].text == "AB"
    assert items[0].raw_representation == {"r": 1}
    assert items[0].additional_properties == {"p": 1}


def test_function_call_content_parse_numeric_or_list():
    c_num = FunctionCallContent(call_id="1", name="f", arguments="123")
    assert c_num.parse_arguments() == {"raw": 123}
    c_list = FunctionCallContent(call_id="1", name="f", arguments="[1,2]")
    assert c_list.parse_arguments() == {"raw": [1, 2]}


def test_chat_tool_mode_eq_with_string():
    assert ToolMode.AUTO == "auto"


# region AgentRunResponse


@fixture
def agent_run_response_async() -> AgentRunResponse:
    return AgentRunResponse(messages=[ChatMessage(role="user", text="Hello")])


async def test_agent_run_response_from_async_generator():
    async def gen():
        yield AgentRunResponseUpdate(contents=[TextContent("A")])
        yield AgentRunResponseUpdate(contents=[TextContent("B")])

    r = await AgentRunResponse.from_agent_response_generator(gen())
    assert r.text == "AB"


# シリアライズおよび算術メソッドの追加カバレッジテストのregion


def test_text_content_add_comprehensive_coverage():
    """カバレッジ向上のため、さまざまな組み合わせでTextContentの__add__メソッドをテスト。"""

    # Noneのraw_representationでテスト
    t1 = TextContent("Hello", raw_representation=None, annotations=None)
    t2 = TextContent(" World", raw_representation=None, annotations=None)
    result = t1 + t2
    assert result.text == "Hello World"
    assert result.raw_representation is None
    assert result.annotations is None

    # 最初はraw_representationを持ち、2番目はNoneでテスト
    t1 = TextContent("Hello", raw_representation="raw1", annotations=None)
    t2 = TextContent(" World", raw_representation=None, annotations=None)
    result = t1 + t2
    assert result.text == "Hello World"
    assert result.raw_representation == "raw1"

    # 最初はNone、2番目はraw_representationを持つ場合のテスト
    t1 = TextContent("Hello", raw_representation=None, annotations=None)
    t2 = TextContent(" World", raw_representation="raw2", annotations=None)
    result = t1 + t2
    assert result.text == "Hello World"
    assert result.raw_representation == "raw2"

    # 両方がraw_representation（リストでない）を持つ場合のテスト
    t1 = TextContent("Hello", raw_representation="raw1", annotations=None)
    t2 = TextContent(" World", raw_representation="raw2", annotations=None)
    result = t1 + t2
    assert result.text == "Hello World"
    assert result.raw_representation == ["raw1", "raw2"]

    # 最初がリストのraw_representation、2番目が単一のテスト
    t1 = TextContent("Hello", raw_representation=["raw1", "raw2"], annotations=None)
    t2 = TextContent(" World", raw_representation="raw3", annotations=None)
    result = t1 + t2
    assert result.text == "Hello World"
    assert result.raw_representation == ["raw1", "raw2", "raw3"]

    # 両方がリストのraw_representationを持つ場合のテスト
    t1 = TextContent("Hello", raw_representation=["raw1", "raw2"], annotations=None)
    t2 = TextContent(" World", raw_representation=["raw3", "raw4"], annotations=None)
    result = t1 + t2
    assert result.text == "Hello World"
    assert result.raw_representation == ["raw1", "raw2", "raw3", "raw4"]

    # 最初が単一のraw_representation、2番目がリストのテスト
    t1 = TextContent("Hello", raw_representation="raw1", annotations=None)
    t2 = TextContent(" World", raw_representation=["raw2", "raw3"], annotations=None)
    result = t1 + t2
    assert result.text == "Hello World"
    assert result.raw_representation == ["raw1", "raw2", "raw3"]


def test_text_content_iadd_coverage():
    """カバレッジ向上のためTextContentの__iadd__メソッドをテスト。"""

    t1 = TextContent("Hello", raw_representation="raw1", additional_properties={"key1": "val1"})
    t2 = TextContent(" World", raw_representation="raw2", additional_properties={"key2": "val2"})

    original_id = id(t1)
    t1 += t2

    # インプレースで変更されるべき
    assert id(t1) == original_id
    assert t1.text == "Hello World"
    assert t1.raw_representation == ["raw1", "raw2"]
    assert t1.additional_properties == {"key1": "val1", "key2": "val2"}


def test_text_reasoning_content_add_coverage():
    """カバレッジ向上のためTextReasoningContentの__add__メソッドをテスト。"""

    t1 = TextReasoningContent("Thinking 1")
    t2 = TextReasoningContent(" Thinking 2")

    result = t1 + t2
    assert result.text == "Thinking 1 Thinking 2"


def test_text_reasoning_content_iadd_coverage():
    """カバレッジ向上のためTextReasoningContentの__iadd__メソッドをテスト。"""

    t1 = TextReasoningContent("Thinking 1")
    t2 = TextReasoningContent(" Thinking 2")

    original_id = id(t1)
    t1 += t2

    assert id(t1) == original_id
    assert t1.text == "Thinking 1 Thinking 2"


def test_comprehensive_to_dict_exclude_options():
    """さまざまなexcludeオプションでto_dictメソッドをテストし、カバレッジを向上。"""

    # exclude_noneを使ったTextContentのテスト
    text_content = TextContent("Hello", raw_representation=None, additional_properties={"prop": "val"})
    text_dict = text_content.to_dict(exclude_none=True)
    assert "raw_representation" not in text_dict
    assert text_dict["prop"] == "val"

    # カスタムexcludeセットでのテスト
    text_dict_exclude = text_content.to_dict(exclude={"additional_properties"})
    assert "additional_properties" not in text_dict_exclude
    assert "text" in text_dict_exclude

    # 追加カウントを持つUsageDetailsのテスト
    usage = UsageDetails(input_token_count=5, custom_count=10)
    usage_dict = usage.to_dict()
    assert usage_dict["input_token_count"] == 5
    assert usage_dict["custom_count"] == 10

    # UsageDetailsのexclude_noneテスト
    usage_none = UsageDetails(input_token_count=5, output_token_count=None)
    usage_dict_no_none = usage_none.to_dict(exclude_none=True)
    assert "output_token_count" not in usage_dict_no_none
    assert usage_dict_no_none["input_token_count"] == 5


def test_usage_details_iadd_edge_cases():
    """エッジケースでUsageDetailsの__iadd__をテストし、カバレッジ向上。"""

    # None値でのテスト
    u1 = UsageDetails(input_token_count=None, output_token_count=5, custom1=10)
    u2 = UsageDetails(input_token_count=3, output_token_count=None, custom2=20)

    u1 += u2
    assert u1.input_token_count == 3
    assert u1.output_token_count == 5
    assert u1.additional_counts["custom1"] == 10
    assert u1.additional_counts["custom2"] == 20

    # 追加カウントのマージをテスト
    u3 = UsageDetails(input_token_count=1, shared_count=5)
    u4 = UsageDetails(input_token_count=2, shared_count=15)

    u3 += u4
    assert u3.input_token_count == 3
    assert u3.additional_counts["shared_count"] == 20


def test_chat_message_from_dict_with_mixed_content():
    """混合コンテンツタイプでChatMessageのfrom_dictをテストし、カバレッジ向上。"""

    message_data = {
        "role": "assistant",
        "contents": [
            {"type": "text", "text": "Hello"},
            {"type": "function_call", "call_id": "call1", "name": "func", "arguments": {"arg": "val"}},
            {"type": "function_result", "call_id": "call1", "result": "success"},
        ],
    }

    message = ChatMessage.from_dict(message_data)
    assert len(message.contents) == 3  # 不明なタイプは無視される
    assert isinstance(message.contents[0], TextContent)
    assert isinstance(message.contents[1], FunctionCallContent)
    assert isinstance(message.contents[2], FunctionResultContent)

    # ラウンドトリップテスト
    message_dict = message.to_dict()
    assert len(message_dict["contents"]) == 3


def test_chat_options_edge_cases():
    """エッジケースでChatOptionsをテストし、カバレッジ向上。"""

    # ツール変換でのテスト
    def sample_tool():
        return "test"

    options = ChatOptions(tools=[sample_tool], tool_choice="auto")
    assert options.tool_choice == ToolMode.AUTO

    # ToolModeでのto_dictテスト
    options_dict = options.to_dict()
    assert "tool_choice" in options_dict

    # tool_choice辞書でのfrom_dictテスト
    data_with_dict_tool_choice = {
        "model_id": "gpt-4",
        "tool_choice": {"mode": "required", "required_function_name": "test_func"},
    }
    options_from_dict = ChatOptions.from_dict(data_with_dict_tool_choice)
    assert options_from_dict.tool_choice.mode == "required"
    assert options_from_dict.tool_choice.required_function_name == "test_func"


def test_text_content_add_type_error():
    """互換性のない型でTextContentの__add__がTypeErrorを発生させるテスト。"""
    t1 = TextContent("Hello")

    with raises(TypeError, match="Incompatible type"):
        t1 + "not a TextContent"


def test_comprehensive_serialization_methods():
    """さまざまなコンテンツタイプでfrom_dictおよびto_dictメソッドをテスト。"""

    # 全フィールドを持つTextContentのテスト
    text_data = {
        "text": "Hello world",
        "raw_representation": {"key": "value"},
        "prop": "val",
        "annotations": None,
    }
    text_content = TextContent.from_dict(text_data)
    assert text_content.text == "Hello world"
    assert text_content.raw_representation == {"key": "value"}
    assert text_content.additional_properties == {"prop": "val"}

    # ラウンドトリップテスト
    text_dict = text_content.to_dict()
    assert text_dict["text"] == "Hello world"
    assert text_dict["prop"] == "val"
    # 注意: raw_representationはto_dict()出力から常に除外されます exclude_noneでのテスト
    text_dict_no_none = text_content.to_dict(exclude_none=True)
    assert "annotations" not in text_dict_no_none

    # FunctionResultContentのテスト
    result_data = {"call_id": "call123", "result": "success", "additional_properties": {"meta": "data"}}
    result_content = FunctionResultContent.from_dict(result_data)
    assert result_content.call_id == "call123"
    assert result_content.result == "success"


def test_chat_options_tool_choice_variations():
    """さまざまなtool_choice値でChatOptionsのfrom_dictおよびto_dictをテスト。"""

    # 文字列tool_choiceでのテスト
    data = {"model_id": "gpt-4", "tool_choice": "auto", "temperature": 0.7}
    options = ChatOptions.from_dict(data)
    assert options.tool_choice == ToolMode.AUTO

    # 辞書tool_choiceでのテスト
    data_dict = {
        "model_id": "gpt-4",
        "tool_choice": {"mode": "required", "required_function_name": "test_func"},
        "temperature": 0.7,
    }
    options_dict = ChatOptions.from_dict(data_dict)
    assert options_dict.tool_choice.mode == "required"
    assert options_dict.tool_choice.required_function_name == "test_func"

    # ToolModeでのto_dictテスト
    options_dict_serialized = options_dict.to_dict()
    assert "tool_choice" in options_dict_serialized
    assert isinstance(options_dict_serialized["tool_choice"], dict)


def test_chat_message_complex_content_serialization():
    """さまざまなコンテンツタイプでChatMessageのシリアライズをテスト。"""

    # 複数のコンテンツタイプを持つメッセージを作成
    contents = [
        TextContent("Hello"),
        FunctionCallContent(call_id="call1", name="func", arguments={"arg": "val"}),
        FunctionResultContent(call_id="call1", result="success"),
    ]

    message = ChatMessage(role=Role.ASSISTANT, contents=contents)

    # to_dictのテスト
    message_dict = message.to_dict()
    assert len(message_dict["contents"]) == 3
    assert message_dict["contents"][0]["type"] == "text"
    assert message_dict["contents"][1]["type"] == "function_call"
    assert message_dict["contents"][2]["type"] == "function_result"

    # from_dictのラウンドトリップテスト
    reconstructed = ChatMessage.from_dict(message_dict)
    assert len(reconstructed.contents) == 3
    assert isinstance(reconstructed.contents[0], TextContent)
    assert isinstance(reconstructed.contents[1], FunctionCallContent)
    assert isinstance(reconstructed.contents[2], FunctionResultContent)


def test_usage_content_serialization_with_details():
    """UsageDetails変換を伴うUsageContentのfrom_dictおよびto_dictのテスト。"""

    # 辞書としてのdetailsでのfrom_dictテスト
    usage_data = {
        "type": "usage",
        "details": {
            "type": "usage_details",
            "input_token_count": 10,
            "output_token_count": 20,
            "total_token_count": 30,
            "custom_count": 5,
        },
    }
    usage_content = UsageContent.from_dict(usage_data)
    assert isinstance(usage_content.details, UsageDetails)
    assert usage_content.details.input_token_count == 10
    assert usage_content.details.additional_counts["custom_count"] == 5

    # UsageDetailsオブジェクトでのto_dictテスト
    usage_dict = usage_content.to_dict()
    assert isinstance(usage_dict["details"], dict)
    assert usage_dict["details"]["input_token_count"] == 10


def test_function_approval_response_content_serialization():
    """function_call変換を伴うFunctionApprovalResponseContentのfrom_dictおよびto_dictのテスト。"""

    # 辞書としてのfunction_callでのfrom_dictテスト
    response_data = {
        "type": "function_approval_response",
        "id": "response123",
        "approved": True,
        "function_call": {
            "type": "function_call",
            "call_id": "call123",
            "name": "test_func",
            "arguments": {"param": "value"},
        },
    }
    response_content = FunctionApprovalResponseContent.from_dict(response_data)
    assert isinstance(response_content.function_call, FunctionCallContent)
    assert response_content.function_call.call_id == "call123"

    # FunctionCallContentオブジェクトでのto_dictテスト
    response_dict = response_content.to_dict()
    assert isinstance(response_dict["function_call"], dict)
    assert response_dict["function_call"]["call_id"] == "call123"


def test_chat_response_complex_serialization():
    """複雑なネストオブジェクトを伴うChatResponseのfrom_dictおよびto_dictのテスト。"""

    # 辞書としてのmessages、finish_reason、usage_detailsでのfrom_dictテスト
    response_data = {
        "messages": [
            {"role": "user", "contents": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "contents": [{"type": "text", "text": "Hi there"}]},
        ],
        "finish_reason": {"value": "stop"},
        "usage_details": {
            "type": "usage_details",
            "input_token_count": 5,
            "output_token_count": 8,
            "total_token_count": 13,
        },
        "model_id": "gpt-4",  # Test alias handling
    }

    response = ChatResponse.from_dict(response_data)
    assert len(response.messages) == 2
    assert isinstance(response.messages[0], ChatMessage)
    assert isinstance(response.finish_reason, FinishReason)
    assert isinstance(response.usage_details, UsageDetails)
    assert response.model_id == "gpt-4"  # model_idとして保存されるべき

    # 複雑なオブジェクトでのto_dictテスト
    response_dict = response.to_dict()
    assert len(response_dict["messages"]) == 2
    assert isinstance(response_dict["messages"][0], dict)
    assert isinstance(response_dict["finish_reason"], dict)
    assert isinstance(response_dict["usage_details"], dict)
    assert response_dict["model_id"] == "gpt-4"  # model_idとしてシリアライズされるべき


def test_chat_response_update_all_content_types():
    """すべてのサポートされるコンテンツタイプでのChatResponseUpdateのfrom_dictテスト。"""

    update_data = {
        "contents": [
            {"type": "text", "text": "Hello"},
            {"type": "data", "data": b"base64data", "media_type": "text/plain"},
            {"type": "uri", "uri": "http://example.com", "media_type": "text/html"},
            {"type": "error", "error": "An error occurred"},
            {"type": "function_call", "call_id": "call1", "name": "func", "arguments": {}},
            {"type": "function_result", "call_id": "call1", "result": "success"},
            {"type": "usage", "details": {"type": "usage_details", "input_token_count": 1}},
            {"type": "hosted_file", "file_id": "file123"},
            {"type": "hosted_vector_store", "vector_store_id": "vs123"},
            {
                "type": "function_approval_request",
                "id": "req1",
                "function_call": {"type": "function_call", "call_id": "call1", "name": "func", "arguments": {}},
            },
            {
                "type": "function_approval_response",
                "id": "resp1",
                "approved": True,
                "function_call": {"type": "function_call", "call_id": "call1", "name": "func", "arguments": {}},
            },
            {"type": "text_reasoning", "text": "reasoning"},
        ]
    }

    update = ChatResponseUpdate.from_dict(update_data)
    assert len(update.contents) == 12  # unknown_typeは警告と共にスキップされる
    assert isinstance(update.contents[0], TextContent)
    assert isinstance(update.contents[1], DataContent)
    assert isinstance(update.contents[2], UriContent)
    assert isinstance(update.contents[3], ErrorContent)
    assert isinstance(update.contents[4], FunctionCallContent)
    assert isinstance(update.contents[5], FunctionResultContent)
    assert isinstance(update.contents[6], UsageContent)
    assert isinstance(update.contents[7], HostedFileContent)
    assert isinstance(update.contents[8], HostedVectorStoreContent)
    assert isinstance(update.contents[9], FunctionApprovalRequestContent)
    assert isinstance(update.contents[10], FunctionApprovalResponseContent)
    assert isinstance(update.contents[11], TextReasoningContent)


def test_agent_run_response_complex_serialization():
    """messagesおよびusage_detailsを伴うAgentRunResponseのfrom_dictおよびto_dictのテスト。"""

    response_data = {
        "messages": [
            {"role": "user", "contents": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "contents": [{"type": "text", "text": "Hi"}]},
        ],
        "usage_details": {
            "type": "usage_details",
            "input_token_count": 3,
            "output_token_count": 2,
            "total_token_count": 5,
        },
    }

    response = AgentRunResponse.from_dict(response_data)
    assert len(response.messages) == 2
    assert isinstance(response.messages[0], ChatMessage)
    assert isinstance(response.usage_details, UsageDetails)

    # to_dictのテスト
    response_dict = response.to_dict()
    assert len(response_dict["messages"]) == 2
    assert isinstance(response_dict["messages"][0], dict)
    assert isinstance(response_dict["usage_details"], dict)


def test_agent_run_response_update_all_content_types():
    """すべてのコンテンツタイプとrole処理を伴うAgentRunResponseUpdateのfrom_dictテスト。"""

    update_data = {
        "contents": [
            {"type": "text", "text": "Hello"},
            {"type": "data", "data": b"base64data", "media_type": "text/plain"},
            {"type": "uri", "uri": "http://example.com", "media_type": "text/html"},
            {"type": "error", "error": "An error occurred"},
            {"type": "function_call", "call_id": "call1", "name": "func", "arguments": {}},
            {"type": "function_result", "call_id": "call1", "result": "success"},
            {"type": "usage", "details": {"type": "usage_details", "input_token_count": 1}},
            {"type": "hosted_file", "file_id": "file123"},
            {"type": "hosted_vector_store", "vector_store_id": "vs123"},
            {
                "type": "function_approval_request",
                "id": "req1",
                "function_call": {"type": "function_call", "call_id": "call1", "name": "func", "arguments": {}},
            },
            {
                "type": "function_approval_response",
                "id": "resp1",
                "approved": True,
                "function_call": {"type": "function_call", "call_id": "call1", "name": "func", "arguments": {}},
            },
            {"type": "text_reasoning", "text": "reasoning"},
        ],
        "role": {"value": "assistant"},  # Test role as dict
    }

    update = AgentRunResponseUpdate.from_dict(update_data)
    assert len(update.contents) == 12  # unknown_typeはログに記録され無視される
    assert isinstance(update.role, Role)
    assert update.role.value == "assistant"

    # role変換を伴うto_dictのテスト
    update_dict = update.to_dict()
    assert len(update_dict["contents"]) == 12  # from_dict中にunknown_typeは無視された
    assert isinstance(update_dict["role"], dict)

    # 文字列変換としてのroleのテスト
    update_data_str_role = update_data.copy()
    update_data_str_role["role"] = "user"
    update_str = AgentRunResponseUpdate.from_dict(update_data_str_role)
    assert isinstance(update_str.role, Role)
    assert update_str.role.value == "user"


# region シリアライズ


@mark.parametrize(
    "content_class,init_kwargs",
    [
        pytest.param(
            TextContent,
            {
                "type": "text",
                "text": "Hello world",
                "raw_representation": "raw",
            },
            id="text_content",
        ),
        pytest.param(
            TextReasoningContent,
            {
                "type": "text_reasoning",
                "text": "Reasoning text",
                "raw_representation": "raw",
            },
            id="text_reasoning_content",
        ),
        pytest.param(
            DataContent,
            {
                "type": "data",
                "uri": "data:text/plain;base64,dGVzdCBkYXRh",
            },
            id="data_content_with_uri",
        ),
        pytest.param(
            DataContent,
            {
                "type": "data",
                "data": b"test data",
                "media_type": "text/plain",
            },
            id="data_content_with_bytes",
        ),
        pytest.param(
            UriContent,
            {
                "type": "uri",
                "uri": "http://example.com",
                "media_type": "text/html",
            },
            id="uri_content",
        ),
        pytest.param(
            HostedFileContent,
            {"type": "hosted_file", "file_id": "file-123"},
            id="hosted_file_content",
        ),
        pytest.param(
            HostedVectorStoreContent,
            {
                "type": "hosted_vector_store",
                "vector_store_id": "vs-789",
            },
            id="hosted_vector_store_content",
        ),
        pytest.param(
            FunctionCallContent,
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "test_func",
                "arguments": {"arg": "val"},
            },
            id="function_call_content",
        ),
        pytest.param(
            FunctionResultContent,
            {
                "type": "function_result",
                "call_id": "call-1",
                "result": "success",
            },
            id="function_result_content",
        ),
        pytest.param(
            ErrorContent,
            {
                "type": "error",
                "message": "Error occurred",
                "error_code": "E001",
            },
            id="error_content",
        ),
        pytest.param(
            UsageContent,
            {
                "type": "usage",
                "details": {
                    "type": "usage_details",
                    "input_token_count": 10,
                    "output_token_count": 20,
                    "reasoning_tokens": 5,
                },
            },
            id="usage_content",
        ),
        pytest.param(
            FunctionApprovalRequestContent,
            {
                "type": "function_approval_request",
                "id": "req-1",
                "function_call": {"type": "function_call", "call_id": "call-1", "name": "test_func", "arguments": {}},
            },
            id="function_approval_request",
        ),
        pytest.param(
            FunctionApprovalResponseContent,
            {
                "type": "function_approval_response",
                "id": "resp-1",
                "approved": True,
                "function_call": {"type": "function_call", "call_id": "call-1", "name": "test_func", "arguments": {}},
            },
            id="function_approval_response",
        ),
        pytest.param(
            ChatMessage,
            {
                "role": {"type": "role", "value": "user"},
                "contents": [
                    {"type": "text", "text": "Hello"},
                    {"type": "function_call", "call_id": "call-1", "name": "test_func", "arguments": {}},
                ],
                "message_id": "msg-123",
                "author_name": "User",
            },
            id="chat_message",
        ),
        pytest.param(
            ChatResponse,
            {
                "type": "chat_response",
                "messages": [
                    {
                        "type": "chat_message",
                        "role": {"type": "role", "value": "user"},
                        "contents": [{"type": "text", "text": "Hello"}],
                    },
                    {
                        "type": "chat_message",
                        "role": {"type": "role", "value": "assistant"},
                        "contents": [{"type": "text", "text": "Hi there"}],
                    },
                ],
                "finish_reason": {"type": "finish_reason", "value": "stop"},
                "usage_details": {
                    "type": "usage_details",
                    "input_token_count": 10,
                    "output_token_count": 20,
                    "total_token_count": 30,
                },
                "response_id": "resp-123",
                "model_id": "gpt-4",
            },
            id="chat_response",
        ),
        pytest.param(
            ChatResponseUpdate,
            {
                "contents": [
                    {"type": "text", "text": "Hello"},
                    {"type": "function_call", "call_id": "call-1", "name": "test_func", "arguments": {}},
                ],
                "role": {"type": "role", "value": "assistant"},
                "finish_reason": {"type": "finish_reason", "value": "stop"},
                "message_id": "msg-123",
                "response_id": "resp-123",
            },
            id="chat_response_update",
        ),
        pytest.param(
            AgentRunResponse,
            {
                "messages": [
                    {
                        "role": {"type": "role", "value": "user"},
                        "contents": [{"type": "text", "text": "Question"}],
                    },
                    {
                        "role": {"type": "role", "value": "assistant"},
                        "contents": [{"type": "text", "text": "Answer"}],
                    },
                ],
                "response_id": "run-123",
                "usage_details": {
                    "type": "usage_details",
                    "input_token_count": 5,
                    "output_token_count": 3,
                    "total_token_count": 8,
                },
            },
            id="agent_run_response",
        ),
        pytest.param(
            AgentRunResponseUpdate,
            {
                "contents": [
                    {"type": "text", "text": "Streaming"},
                    {"type": "function_call", "call_id": "call-1", "name": "test_func", "arguments": {}},
                ],
                "role": {"type": "role", "value": "assistant"},
                "message_id": "msg-123",
                "response_id": "run-123",
                "author_name": "Agent",
            },
            id="agent_run_response_update",
        ),
    ],
)
def test_content_roundtrip_serialization(content_class: type[BaseContent], init_kwargs: dict[str, Any]):
    """すべてのコンテンツタイプのto_dict/from_dictラウンドトリップテスト。"""
    # インスタンスを作成
    content = content_class(**init_kwargs)

    # 辞書にシリアライズ
    content_dict = content.to_dict()

    # シリアライズされた辞書にtypeキーがあることを検証
    assert "type" in content_dict
    if hasattr(content, "type"):
        assert content_dict["type"] == content.type  # type: ignore[attr-defined]

    # 辞書からデシリアライズ
    reconstructed = content_class.from_dict(content_dict)

    # typeを検証
    assert isinstance(reconstructed, content_class)
    # 動的にtype属性をチェック
    if hasattr(content, "type"):
        assert reconstructed.type == content.type  # type: ignore[attr-defined]

    # 主要属性を検証（raw_representationはシリアライズされないため除外）
    for key, value in init_kwargs.items():
        if key == "type":
            continue
        if key == "raw_representation":
            # raw_representationは意図的にシリアライズから除外される
            continue

        # 'data'パラメータで作成されたDataContentの特別な処理
        if content_class == DataContent and key == "data":
            # DataContentは'data'を'uri'に変換するため、'data'属性のチェックはスキップ
            # 代わりにuriとmedia_typeが正しく設定されていることを検証
            assert hasattr(reconstructed, "uri")
            assert hasattr(reconstructed, "media_type")
            assert reconstructed.media_type == init_kwargs.get("media_type")
            # uriにエンコードされたデータが含まれていることを検証
            assert reconstructed.uri.startswith(f"data:{init_kwargs.get('media_type')};base64,")
            continue

        reconstructed_value = getattr(reconstructed, key)

        # ネストされたSerializationMixinオブジェクトの特別な処理
        if hasattr(value, "to_dict"):
            # シリアライズされた形式を比較
            assert reconstructed_value.to_dict() == value.to_dict()
        # 辞書がオブジェクトに変換される可能性のあるリストの特別な処理
        elif isinstance(value, list) and value and isinstance(reconstructed_value, list):
            # これは辞書から作成されたオブジェクトのリストかどうかをチェック
            if isinstance(value[0], dict) and hasattr(reconstructed_value[0], "to_dict"):
                # 再構築されたオブジェクトをシリアライズして各アイテムを比較
                assert len(reconstructed_value) == len(value)

            else:
                assert reconstructed_value == value
        # UsageDetailsやFunctionCallContentのようにオブジェクトに変換される辞書の特別な処理
        elif isinstance(value, dict) and hasattr(reconstructed_value, "to_dict"):
            # 'type'キーを除外してオブジェクトのシリアライズ形式と辞書を比較
            reconstructed_dict = reconstructed_value.to_dict()
            if value:
                assert len(reconstructed_dict) == len(value)
        else:
            assert reconstructed_value == value


def test_text_content_with_annotations_serialization():
    """CitationAnnotationとTextSpanRegionを伴うTextContentのラウンドトリップシリアライズをテスト。"""
    # TextSpanRegionを作成
    region = TextSpanRegion(start_index=0, end_index=5)

    # regionを持つCitationAnnotationを作成
    citation = CitationAnnotation(
        title="Test Citation",
        url="http://example.com/citation",
        file_id="file-123",
        tool_name="test_tool",
        snippet="This is a test snippet",
        annotated_regions=[region],
        additional_properties={"custom": "value"},
    )

    # アノテーション付きTextContentを作成
    content = TextContent(
        text="Hello world", annotations=[citation], additional_properties={"content_key": "content_val"}
    )

    # 辞書にシリアライズ
    content_dict = content.to_dict()

    # 構造を検証
    assert content_dict["type"] == "text"
    assert content_dict["text"] == "Hello world"
    assert content_dict["content_key"] == "content_val"
    assert len(content_dict["annotations"]) == 1

    # アノテーション構造を検証
    annotation_dict = content_dict["annotations"][0]
    assert annotation_dict["type"] == "citation"
    assert annotation_dict["title"] == "Test Citation"
    assert annotation_dict["url"] == "http://example.com/citation"
    assert annotation_dict["file_id"] == "file-123"
    assert annotation_dict["tool_name"] == "test_tool"
    assert annotation_dict["snippet"] == "This is a test snippet"
    assert annotation_dict["custom"] == "value"

    # region構造を検証
    assert len(annotation_dict["annotated_regions"]) == 1
    region_dict = annotation_dict["annotated_regions"][0]
    assert region_dict["type"] == "text_span"
    assert region_dict["start_index"] == 0
    assert region_dict["end_index"] == 5

    # 辞書からデシリアライズ
    reconstructed = TextContent.from_dict(content_dict)

    # 再構築されたコンテンツを検証
    assert isinstance(reconstructed, TextContent)
    assert reconstructed.text == "Hello world"
    assert reconstructed.type == "text"
    assert reconstructed.additional_properties == {"content_key": "content_val"}

    # 再構築されたアノテーションを検証
    assert len(reconstructed.annotations) == 1  # type: ignore[arg-type]
    recon_annotation = reconstructed.annotations[0]  # type: ignore[index]
    assert isinstance(recon_annotation, CitationAnnotation)
    assert recon_annotation.title == "Test Citation"
    assert recon_annotation.url == "http://example.com/citation"
    assert recon_annotation.file_id == "file-123"
    assert recon_annotation.tool_name == "test_tool"
    assert recon_annotation.snippet == "This is a test snippet"
    assert recon_annotation.additional_properties == {"custom": "value"}

    # 再構築されたregionを検証
    assert len(recon_annotation.annotated_regions) == 1  # type: ignore[arg-type]
    recon_region = recon_annotation.annotated_regions[0]  # type: ignore[index]
    assert isinstance(recon_region, TextSpanRegion)
    assert recon_region.start_index == 0
    assert recon_region.end_index == 5
    assert recon_region.type == "text_span"


def test_text_content_with_multiple_annotations_serialization():
    """複数のアノテーションを持つTextContentのラウンドトリップシリアライズをテスト。"""
    # 複数のregionを作成
    region1 = TextSpanRegion(start_index=0, end_index=5)
    region2 = TextSpanRegion(start_index=6, end_index=11)

    # 複数のcitationを作成
    citation1 = CitationAnnotation(title="Citation 1", url="http://example.com/1", annotated_regions=[region1])

    citation2 = CitationAnnotation(title="Citation 2", url="http://example.com/2", annotated_regions=[region2])

    # 複数のアノテーションを持つTextContentを作成
    content = TextContent(text="Hello world", annotations=[citation1, citation2])

    # シリアライズ
    content_dict = content.to_dict()

    # 2つのアノテーションがあることを検証
    assert len(content_dict["annotations"]) == 2
    assert content_dict["annotations"][0]["title"] == "Citation 1"
    assert content_dict["annotations"][1]["title"] == "Citation 2"

    # デシリアライズ
    reconstructed = TextContent.from_dict(content_dict)

    # 再構築を検証
    assert len(reconstructed.annotations) == 2
    assert all(isinstance(ann, CitationAnnotation) for ann in reconstructed.annotations)
    assert reconstructed.annotations[0].title == "Citation 1"
    assert reconstructed.annotations[1].title == "Citation 2"
    assert all(isinstance(ann.annotated_regions[0], TextSpanRegion) for ann in reconstructed.annotations)
