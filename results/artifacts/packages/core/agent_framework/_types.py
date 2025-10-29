# Copyright (c) Microsoft. All rights reserved.

import base64
import json
import re
import sys
from collections.abc import (
    AsyncIterable,
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from copy import deepcopy
from typing import Any, ClassVar, Literal, TypeVar, cast, overload

from pydantic import BaseModel, ValidationError

from ._logging import get_logger
from ._serialization import SerializationMixin
from ._tools import ToolProtocol, ai_function
from .exceptions import AdditionItemMismatch, ContentError

if sys.version_info >= (3, 11):
    from typing import Self  # pragma: no cover
else:
    from typing_extensions import Self  # pragma: no cover


__all__ = [
    "AgentRunResponse",
    "AgentRunResponseUpdate",
    "AnnotatedRegions",
    "Annotations",
    "BaseAnnotation",
    "BaseContent",
    "ChatMessage",
    "ChatOptions",
    "ChatResponse",
    "ChatResponseUpdate",
    "CitationAnnotation",
    "Contents",
    "DataContent",
    "ErrorContent",
    "FinishReason",
    "FunctionApprovalRequestContent",
    "FunctionApprovalResponseContent",
    "FunctionCallContent",
    "FunctionResultContent",
    "HostedFileContent",
    "HostedVectorStoreContent",
    "Role",
    "TextContent",
    "TextReasoningContent",
    "TextSpanRegion",
    "ToolMode",
    "UriContent",
    "UsageContent",
    "UsageDetails",
    "prepare_function_call_results",
]

logger = get_logger("agent_framework")


# region Content Parsing Utilities


class EnumLike(type):
    """定義済み定数を持つenum風クラスを作成するためのジェネリックメタクラス。

    このメタクラスは_constantsクラス属性に基づいてクラスレベルの定数を自動的に作成します。
    各定数は(name, *args)のタプルとして定義され、
    nameは定数名、argsはコンストラクタ引数です。

    """

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]) -> "EnumLike":
        cls = super().__new__(mcs, name, bases, namespace)

        # _constantsが定義されていれば定数を作成します。
        if (const := getattr(cls, "_constants", None)) and isinstance(const, dict):
            for const_name, const_args in const.items():
                if isinstance(const_args, (list, tuple)):
                    setattr(cls, const_name, cls(*const_args))
                else:
                    setattr(cls, const_name, cls(const_args))

        return cls


def _parse_content(content_data: MutableMapping[str, Any]) -> "Contents":
    """単一のcontentデータ辞書を適切なContentオブジェクトにパースします。

    Args:
        content_data: Contentデータ（dict）

    Returns:
        Contentオブジェクト

    Raises:
        パースに失敗した場合はContentError

    """
    content_type = str(content_data.get("type"))
    match content_type:
        case "text":
            return TextContent.from_dict(content_data)
        case "data":
            return DataContent.from_dict(content_data)
        case "uri":
            return UriContent.from_dict(content_data)
        case "error":
            return ErrorContent.from_dict(content_data)
        case "function_call":
            return FunctionCallContent.from_dict(content_data)
        case "function_result":
            return FunctionResultContent.from_dict(content_data)
        case "usage":
            return UsageContent.from_dict(content_data)
        case "hosted_file":
            return HostedFileContent.from_dict(content_data)
        case "hosted_vector_store":
            return HostedVectorStoreContent.from_dict(content_data)
        case "function_approval_request":
            return FunctionApprovalRequestContent.from_dict(content_data)
        case "function_approval_response":
            return FunctionApprovalResponseContent.from_dict(content_data)
        case "text_reasoning":
            return TextReasoningContent.from_dict(content_data)
        case _:
            raise ContentError(f"Unknown content type '{content_type}'")


def _parse_content_list(contents_data: Sequence[Any]) -> list["Contents"]:
    """contentデータ辞書のリストを適切なContentオブジェクトにパースします。

    Args:
        contents_data: Contentデータのリスト（dictまたは既に構築されたオブジェクト）

    Returns:
        不明なタイプはログに記録して無視し、Contentオブジェクトのリストを返します。

    """
    contents: list["Contents"] = []
    for content_data in contents_data:
        if isinstance(content_data, dict):
            try:
                content = _parse_content(content_data)
                contents.append(content)
            except ContentError as exc:
                logger.warning(f"Skipping unknown content type or invalid content: {exc}")
        else:
            # すでにcontentオブジェクトであればそのまま保持します。
            contents.append(content_data)

    return contents


# endregion region Constants and types
_T = TypeVar("_T")
TEmbedding = TypeVar("TEmbedding")
TChatResponse = TypeVar("TChatResponse", bound="ChatResponse")
TToolMode = TypeVar("TToolMode", bound="ToolMode")
TAgentRunResponse = TypeVar("TAgentRunResponse", bound="AgentRunResponse")

CreatedAtT = str  # datetimeoffset型を使うか？それともdatetime.datetimeのようなより具体的な型か？

URI_PATTERN = re.compile(r"^data:(?P<media_type>[^;]+);base64,(?P<base64_data>[A-Za-z0-9+/=]+)$")

KNOWN_MEDIA_TYPES = [
    "application/json",
    "application/octet-stream",
    "application/pdf",
    "application/xml",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/wav",
    "image/apng",
    "image/avif",
    "image/bmp",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/svg+xml",
    "image/tiff",
    "image/webp",
    "text/css",
    "text/csv",
    "text/html",
    "text/javascript",
    "text/plain",
    "text/plain;charset=UTF-8",
    "text/xml",
]


class UsageDetails(SerializationMixin):
    """リクエスト/レスポンスの使用状況の詳細を提供します。

    Attributes:
        input_token_count: 入力のトークン数。
        output_token_count: 出力のトークン数。
        total_token_count: レスポンス生成に使用されたトークンの合計数。
        additional_counts: 追加のトークン数の辞書。kwargsで設定可能。

    Examples:
        .. code-block:: python

            from agent_framework import UsageDetails

            # 使用状況詳細の作成
            usage = UsageDetails(
                input_token_count=100,
                output_token_count=50,
                total_token_count=150,
            )
            print(usage.total_token_count)  # 150

            # 追加のカウントを含む場合
            usage = UsageDetails(
                input_token_count=100,
                output_token_count=50,
                total_token_count=150,
                reasoning_tokens=25,
            )
            print(usage.additional_counts["reasoning_tokens"])  # 25

            # 使用状況詳細の結合
            usage1 = UsageDetails(input_token_count=100, output_token_count=50)
            usage2 = UsageDetails(input_token_count=200, output_token_count=100)
            combined = usage1 + usage2
            print(combined.input_token_count)  # 300

    """

    DEFAULT_EXCLUDE: ClassVar[set[str]] = {"_extra_counts"}

    def __init__(
        self,
        input_token_count: int | None = None,
        output_token_count: int | None = None,
        total_token_count: int | None = None,
        **kwargs: int,
    ) -> None:
        """UsageDetailsインスタンスを初期化します。

        Args:
            input_token_count: 入力のトークン数。
            output_token_count: 出力のトークン数。
            total_token_count: レスポンス生成に使用されたトークンの合計数。

        Keyword Args:
            **kwargs: 追加のトークン数。キーワード引数で設定可能。
                `additional_counts`プロパティを通じて取得できます。

        """
        self.input_token_count = input_token_count
        self.output_token_count = output_token_count
        self.total_token_count = total_token_count

        # すべてのkwargsが整数であることを検証します（Pydanticの動作を保持）。
        self._extra_counts: dict[str, int] = {}
        for key, value in kwargs.items():
            if not isinstance(value, int):
                raise ValueError(f"Additional counts must be integers, got {type(value).__name__}")
            self._extra_counts[key] = value

    def to_dict(self, *, exclude_none: bool = True, exclude: set[str] | None = None) -> dict[str, Any]:
        """UsageDetailsインスタンスを辞書に変換します。

        Keyword Args:
            exclude_none: None値を出力から除外するかどうか。
            exclude: 出力から除外するフィールド名のセット。

        Returns:
            UsageDetailsインスタンスの辞書表現。

        """
        # 親クラスから基本の辞書を取得します。
        result = super().to_dict(exclude_none=exclude_none, exclude=exclude)

        # 追加のカウント（追加フィールド）を加えます。
        if exclude is None:
            exclude = set()

        for key, value in self._extra_counts.items():
            if key in exclude:
                continue
            if exclude_none and value is None:
                continue
            result[key] = value

        return result

    def __str__(self) -> str:
        """使用状況詳細の文字列表現を返します。"""
        return self.to_json()

    @property
    def additional_counts(self) -> dict[str, int]:
        """使用状況のよく知られた追加カウントを表します。これは網羅的なリストではありません。

        Remarks:
            異なるAIサービス間で同名だが無関係な追加カウントの衝突を避けるために、
            ここで明示的に定義されていないキーはAIサービス名でプレフィックスを付けるべきです。
            例："openai."や"azure."。区切り文字"."はJSONキーとして不正な文字のため選ばれました。

            時間とともに追加カウントがベースクラスに追加される可能性があります。

        """
        return self._extra_counts

    def __setitem__(self, key: str, value: int) -> None:
        """使用状況詳細に追加カウントを設定します。"""
        if not isinstance(value, int):
            raise ValueError("Additional counts must be integers.")
        self._extra_counts[key] = value

    def __add__(self, other: "UsageDetails | None") -> "UsageDetails":
        """2つの`UsageDetails`インスタンスを結合します。"""
        if not other:
            return self
        if not isinstance(other, UsageDetails):
            raise ValueError("Can only add two usage details objects together.")

        additional_counts = self.additional_counts.copy()
        if other.additional_counts:
            for key, value in other.additional_counts.items():
                additional_counts[key] = additional_counts.get(key, 0) + (value or 0)

        return UsageDetails(
            input_token_count=(self.input_token_count or 0) + (other.input_token_count or 0),
            output_token_count=(self.output_token_count or 0) + (other.output_token_count or 0),
            total_token_count=(self.total_token_count or 0) + (other.total_token_count or 0),
            **additional_counts,
        )

    def __iadd__(self, other: "UsageDetails | None") -> Self:
        if not other:
            return self
        if not isinstance(other, UsageDetails):
            raise ValueError("Can only add usage details objects together.")

        self.input_token_count = (self.input_token_count or 0) + (other.input_token_count or 0)
        self.output_token_count = (self.output_token_count or 0) + (other.output_token_count or 0)
        self.total_token_count = (self.total_token_count or 0) + (other.total_token_count or 0)

        for key, value in other.additional_counts.items():
            self.additional_counts[key] = self.additional_counts.get(key, 0) + (value or 0)

        return self

    def __eq__(self, other: object) -> bool:
        """2つのUsageDetailsインスタンスが等しいかどうかをチェックします。"""
        if not isinstance(other, UsageDetails):
            return False

        return (
            self.input_token_count == other.input_token_count
            and self.output_token_count == other.output_token_count
            and self.total_token_count == other.total_token_count
            and self.additional_counts == other.additional_counts
        )


# region BaseAnnotation


class TextSpanRegion(SerializationMixin):
    """注釈されたテキスト領域を表します。

    Examples:
        .. code-block:: python

            from agent_framework import TextSpanRegion

            # テキストスパン領域を作成
            region = TextSpanRegion(start_index=0, end_index=10)
            print(region.type)  # "text_span"

    """

    def __init__(
        self,
        *,
        start_index: int | None = None,
        end_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        """TextSpanRegionを初期化します。

        Keyword Args:
            start_index: テキストスパンの開始インデックス。
            end_index: テキストスパンの終了インデックス。
            **kwargs: 追加のキーワード引数。

        """
        self.type: Literal["text_span"] = "text_span"
        self.start_index = start_index
        self.end_index = end_index

        # 追加のkwargsを処理します。
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


AnnotatedRegions = TextSpanRegion


class BaseAnnotation(SerializationMixin):
    """すべてのAI Annotationタイプの基底クラス。"""

    DEFAULT_EXCLUDE: ClassVar[set[str]] = {"raw_representation", "additional_properties"}

    def __init__(
        self,
        *,
        annotated_regions: list[AnnotatedRegions] | list[MutableMapping[str, Any]] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """BaseAnnotationを初期化します。

        Keyword Args:
            annotated_regions: 注釈された領域のリスト。regionオブジェクトまたはdict。
            additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
            raw_representation: 基盤実装からのコンテンツのオプションの生表現。
            **kwargs: 追加のキーワード引数（additional_propertiesにマージされます）。

        """
        # annotated_regionsのdict形式からの変換を処理します（SerializationMixin対応）。
        self.annotated_regions: list[AnnotatedRegions] | None = None
        if annotated_regions is not None:
            converted_regions: list[AnnotatedRegions] = []
            for region_data in annotated_regions:
                if isinstance(region_data, MutableMapping):
                    if region_data.get("type", "") == "text_span":
                        converted_regions.append(TextSpanRegion.from_dict(region_data))
                    else:
                        logger.warning(f"Unknown region type: {region_data.get('type', '')} in {region_data}")
                else:
                    # すでにregionオブジェクトであればそのまま保持します。
                    converted_regions.append(region_data)
            self.annotated_regions = converted_regions

        # kwargs を additional_properties にマージする
        self.additional_properties = additional_properties or {}
        self.additional_properties.update(kwargs)

        self.raw_representation = raw_representation

    def to_dict(self, *, exclude: set[str] | None = None, exclude_none: bool = True) -> dict[str, Any]:
        """インスタンスを辞書に変換します。

        additional_properties のフィールドをルートレベルに抽出します。

        Keyword Args:
            exclude: シリアライズから除外するフィールド名のセット。
            exclude_none: None の値を出力から除外するかどうか。デフォルトは True。

        Returns:
            インスタンスの辞書表現。

        """
        # SerializationMixin から基本の辞書を取得する
        result = super().to_dict(exclude=exclude, exclude_none=exclude_none)

        # additional_properties をルートレベルに抽出する
        if self.additional_properties:
            result.update(self.additional_properties)

        return result


class CitationAnnotation(BaseAnnotation):
    """引用注釈を表します。

    Attributes:
        type: コンテンツのタイプ。このクラスでは常に "citation" です。
        title: 引用されたコンテンツのタイトル。
        url: 引用されたコンテンツの URL。
        file_id: 該当する場合、引用されたコンテンツのファイル識別子。
        tool_name: 該当する場合、引用を生成したツールの名前。
        snippet: 該当する場合、引用されたコンテンツの抜粋。
        annotated_regions: この引用で注釈された領域のリスト。
        additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
        raw_representation: 基盤となる実装からのコンテンツのオプションの生の表現。

    Examples:
        .. code-block:: python

            from agent_framework import CitationAnnotation, TextSpanRegion

            # 引用注釈を作成
            citation = CitationAnnotation(
                title="Agent Framework Documentation",
                url="https://example.com/docs",
                snippet="This is a relevant excerpt...",
                annotated_regions=[TextSpanRegion(start_index=0, end_index=25)],
            )
            print(citation.title)  # "Agent Framework Documentation"

    """

    def __init__(
        self,
        *,
        title: str | None = None,
        url: str | None = None,
        file_id: str | None = None,
        tool_name: str | None = None,
        snippet: str | None = None,
        annotated_regions: list[AnnotatedRegions] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """CitationAnnotation を初期化します。

        Keyword Args:
            title: 引用されたコンテンツのタイトル。
            url: 引用されたコンテンツの URL。
            file_id: 該当する場合、引用されたコンテンツのファイル識別子。
            tool_name: 該当する場合、引用を生成したツールの名前。
            snippet: 該当する場合、引用されたコンテンツの抜粋。
            annotated_regions: この引用で注釈された領域のリスト。
            additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
            raw_representation: 基盤となる実装からのコンテンツのオプションの生の表現。
            **kwargs: 追加のキーワード引数。

        """
        super().__init__(
            annotated_regions=annotated_regions,
            additional_properties=additional_properties,
            raw_representation=raw_representation,
            **kwargs,
        )
        self.title = title
        self.url = url
        self.file_id = file_id
        self.tool_name = tool_name
        self.snippet = snippet
        self.type: Literal["citation"] = "citation"


Annotations = CitationAnnotation


# region BaseContent

TContents = TypeVar("TContents", bound="BaseContent")


class BaseContent(SerializationMixin):
    """AI サービスで使用されるコンテンツを表します。

    Attributes:
        annotations: コンテンツに関連付けられたオプションの注釈。
        additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
        raw_representation: 基盤となる実装からのコンテンツのオプションの生の表現。


    """

    DEFAULT_EXCLUDE: ClassVar[set[str]] = {"raw_representation", "additional_properties"}

    def __init__(
        self,
        *,
        annotations: list[Annotations | MutableMapping[str, Any]] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """BaseContent を初期化します。

        Keyword Args:
            annotations: コンテンツに関連付けられたオプションの注釈。注釈オブジェクトまたは辞書のいずれか。
            additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
            raw_representation: 基盤となる実装からのコンテンツのオプションの生の表現。
            **kwargs: 追加のキーワード引数（additional_properties にマージされます）。

        """
        self.annotations: list[Annotations] | None = None
        # 辞書形式からの注釈変換を処理する（SerializationMixin サポート用）
        if annotations is not None:
            converted_annotations: list[Annotations] = []
            for annotation_data in annotations:
                if isinstance(annotation_data, Annotations):
                    # すでに注釈オブジェクトであれば、そのまま保持する
                    converted_annotations.append(annotation_data)
                elif isinstance(annotation_data, MutableMapping) and annotation_data.get("type", "") == "citation":
                    converted_annotations.append(CitationAnnotation.from_dict(annotation_data))
                else:
                    logger.debug(
                        f"Unknown annotation found: {annotation_data.get('type', 'no_type')}"
                        f" with data: {annotation_data}"
                    )
            self.annotations = converted_annotations

        # kwargs を additional_properties にマージする
        self.additional_properties = additional_properties or {}
        self.additional_properties.update(kwargs)

        self.raw_representation = raw_representation

    def to_dict(self, *, exclude: set[str] | None = None, exclude_none: bool = True) -> dict[str, Any]:
        """インスタンスを辞書に変換します。

        additional_properties のフィールドをルートレベルに抽出します。

        Keyword Args:
            exclude: シリアライズから除外するフィールド名のセット。
            exclude_none: None の値を出力から除外するかどうか。デフォルトは True。

        Returns:
            インスタンスの辞書表現。

        """
        # SerializationMixin から基本の辞書を取得する
        result = super().to_dict(exclude=exclude, exclude_none=exclude_none)

        # additional_properties をルートレベルに抽出する
        if self.additional_properties:
            result.update(self.additional_properties)

        return result


class TextContent(BaseContent):
    """チャット内のテキストコンテンツを表します。

    Attributes:
        text: このインスタンスが表すテキストコンテンツ。
        type: コンテンツのタイプ。このクラスでは常に "text" です。
        annotations: コンテンツに関連付けられたオプションの注釈。
        additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
        raw_representation: コンテンツのオプションの生の表現。

    Examples:
        .. code-block:: python

            from agent_framework import TextContent

            # 基本的なテキストコンテンツを作成
            text = TextContent(text="Hello, world!")
            print(text.text)  # "Hello, world!"

            # テキストコンテンツを連結
            text1 = TextContent(text="Hello, ")
            text2 = TextContent(text="world!")
            combined = text1 + text2
            print(combined.text)  # "Hello, world!"

    """

    def __init__(
        self,
        text: str,
        *,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        annotations: list[Annotations | MutableMapping[str, Any]] | None = None,
        **kwargs: Any,
    ):
        """TextContent インスタンスを初期化します。

        Args:
            text: このインスタンスが表すテキストコンテンツ。

        Keyword Args:
            additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
            raw_representation: コンテンツのオプションの生の表現。
            annotations: コンテンツに関連付けられたオプションの注釈。
            **kwargs: その他の追加キーワード引数。

        """
        super().__init__(
            annotations=annotations,
            additional_properties=additional_properties,
            raw_representation=raw_representation,
            **kwargs,
        )
        self.text = text
        self.type: Literal["text"] = "text"

    def __add__(self, other: "TextContent") -> "TextContent":
        """2つの TextContent インスタンスを連結します。

        以下の処理が行われます:
        テキストが連結されます。
        注釈が結合されます。
        追加プロパティがマージされ、共有キーの値は最初のインスタンスのものが優先されます。
        両方に raw_representation がある場合は、それらがリストにまとめられます。

        """
        if not isinstance(other, TextContent):
            raise TypeError("Incompatible type")

        # raw_representation をマージする
        if self.raw_representation is None:
            raw_representation = other.raw_representation
        elif other.raw_representation is None:
            raw_representation = self.raw_representation
        else:
            raw_representation = (
                self.raw_representation if isinstance(self.raw_representation, list) else [self.raw_representation]
            ) + (other.raw_representation if isinstance(other.raw_representation, list) else [other.raw_representation])

        # 注釈をマージする
        if self.annotations is None:
            annotations = other.annotations
        elif other.annotations is None:
            annotations = self.annotations
        else:
            annotations = self.annotations + other.annotations

        # 適切なデシリアライズのため from_dict を使って新しいインスタンスを作成する
        result_dict = {
            "text": self.text + other.text,
            "type": "text",
            "annotations": [ann.to_dict(exclude_none=False) for ann in annotations] if annotations else None,
            "additional_properties": {
                **(other.additional_properties or {}),
                **(self.additional_properties or {}),
            },
            "raw_representation": raw_representation,
        }
        return TextContent.from_dict(result_dict)

    def __iadd__(self, other: "TextContent") -> Self:
        """2つの TextContent インスタンスをインプレースで連結します。

        以下の処理が行われます:
        テキストが連結されます。
        注釈が結合されます。
        追加プロパティがマージされ、共有キーの値は最初のインスタンスのものが優先されます。
        両方に raw_representation がある場合は、それらがリストにまとめられます。

        """
        if not isinstance(other, TextContent):
            raise TypeError("Incompatible type")

        # テキストを連結する
        self.text += other.text

        # 追加プロパティをマージする（self が優先）
        if self.additional_properties is None:
            self.additional_properties = {}
        if other.additional_properties:
            # まず他方から更新し、その後 self の値を復元して優先を維持する
            self_props = self.additional_properties.copy()
            self.additional_properties.update(other.additional_properties)
            self.additional_properties.update(self_props)

        # raw_representation をマージする
        if self.raw_representation is None:
            self.raw_representation = other.raw_representation
        elif other.raw_representation is not None:
            self.raw_representation = (
                self.raw_representation if isinstance(self.raw_representation, list) else [self.raw_representation]
            ) + (other.raw_representation if isinstance(other.raw_representation, list) else [other.raw_representation])

        # 注釈をマージする
        if other.annotations:
            if self.annotations is None:
                self.annotations = []
            self.annotations.extend(other.annotations)

        return self


class TextReasoningContent(BaseContent):
    """チャット内のテキスト推論コンテンツを表します。

    Remarks:
        このクラスと `TextContent` は表面的には似ていますが、異なります。

    Attributes:
        text: このインスタンスが表すテキストコンテンツ。
        type: コンテンツのタイプ。このクラスでは常に "text_reasoning" です。
        annotations: コンテンツに関連付けられたオプションの注釈。
        additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
        raw_representation: コンテンツのオプションの生の表現。

    Examples:
        .. code-block:: python

            from agent_framework import TextReasoningContent

            # 推論コンテンツを作成
            reasoning = TextReasoningContent(text="Let me think step by step...")
            print(reasoning.text)  # "Let me think step by step..."

            # 推論コンテンツを連結
            reasoning1 = TextReasoningContent(text="First, ")
            reasoning2 = TextReasoningContent(text="second, ")
            combined = reasoning1 + reasoning2
            print(combined.text)  # "First, second, "

    """

    def __init__(
        self,
        text: str,
        *,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        annotations: list[Annotations | MutableMapping[str, Any]] | None = None,
        **kwargs: Any,
    ):
        """TextReasoningContent インスタンスを初期化します。

        Args:
            text: このインスタンスが表すテキストコンテンツ。

        Keyword Args:
            additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
            raw_representation: コンテンツのオプションの生の表現。
            annotations: コンテンツに関連付けられたオプションの注釈。
            **kwargs: その他の追加キーワード引数。

        """
        super().__init__(
            annotations=annotations,
            additional_properties=additional_properties,
            raw_representation=raw_representation,
            **kwargs,
        )
        self.text = text
        self.type: Literal["text_reasoning"] = "text_reasoning"

    def __add__(self, other: "TextReasoningContent") -> "TextReasoningContent":
        """2つの TextReasoningContent インスタンスを連結します。

        以下の処理が行われます:
        テキストが連結されます。
        注釈が結合されます。
        追加プロパティがマージされ、共有キーの値は最初のインスタンスのものが優先されます。
        両方に raw_representation がある場合は、それらがリストにまとめられます。

        """
        if not isinstance(other, TextReasoningContent):
            raise TypeError("Incompatible type")

        # raw_representation をマージする
        if self.raw_representation is None:
            raw_representation = other.raw_representation
        elif other.raw_representation is None:
            raw_representation = self.raw_representation
        else:
            raw_representation = (
                self.raw_representation if isinstance(self.raw_representation, list) else [self.raw_representation]
            ) + (other.raw_representation if isinstance(other.raw_representation, list) else [other.raw_representation])

        # 注釈をマージする
        if self.annotations is None:
            annotations = other.annotations
        elif other.annotations is None:
            annotations = self.annotations
        else:
            annotations = self.annotations + other.annotations

        # 適切なデシリアライズのため from_dict を使って新しいインスタンスを作成する
        result_dict = {
            "text": self.text + other.text,
            "type": "text_reasoning",
            "annotations": [ann.to_dict(exclude_none=False) for ann in annotations] if annotations else None,
            "additional_properties": {**(self.additional_properties or {}), **(other.additional_properties or {})},
            "raw_representation": raw_representation,
        }
        return TextReasoningContent.from_dict(result_dict)

    def __iadd__(self, other: "TextReasoningContent") -> Self:
        """2つの TextReasoningContent インスタンスをインプレースで連結します。

        以下の処理が行われます:
        テキストが連結されます。
        注釈が結合されます。
        追加プロパティがマージされ、共有キーの値は最初のインスタンスのものが優先されます。
        両方に raw_representation がある場合は、それらがリストにまとめられます。

        """
        if not isinstance(other, TextReasoningContent):
            raise TypeError("Incompatible type")

        # テキストを連結する
        self.text += other.text

        # 追加プロパティをマージする（self が優先）
        if self.additional_properties is None:
            self.additional_properties = {}
        if other.additional_properties:
            # まず他方から更新し、その後 self の値を復元して優先を維持する
            self_props = self.additional_properties.copy()
            self.additional_properties.update(other.additional_properties)
            self.additional_properties.update(self_props)

        # raw_representation をマージする
        if self.raw_representation is None:
            self.raw_representation = other.raw_representation
        elif other.raw_representation is not None:
            self.raw_representation = (
                self.raw_representation if isinstance(self.raw_representation, list) else [self.raw_representation]
            ) + (other.raw_representation if isinstance(other.raw_representation, list) else [other.raw_representation])

        # 注釈をマージする
        if other.annotations:
            if self.annotations is None:
                self.annotations = []
            self.annotations.extend(other.annotations)

        return self


class DataContent(BaseContent):
    """メディアタイプ（MIMEタイプとも呼ばれる）に関連付けられたバイナリデータコンテンツを表します。

    Important:
        これはデータ URI として表現されるバイナリデータ用であり、オンラインリソース用ではありません。
        オンラインリソースには ``UriContent`` を使用してください。

    Attributes:
        uri: このインスタンスが表すデータの URI。通常はデータ URI の形式です。
            形式は "data:{media_type};base64,{base64_data}" の形であるべきです。
        media_type: データのメディアタイプ。
        type: コンテンツのタイプ。このクラスでは常に "data" です。
        annotations: コンテンツに関連付けられたオプションの注釈。
        additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
        raw_representation: コンテンツのオプションの生の表現。

    Examples:
        .. code-block:: python

            from agent_framework import DataContent

            # バイナリデータから作成
            image_data = b"raw image bytes"
            data_content = DataContent(data=image_data, media_type="image/png")

            # データ URI から作成
            data_uri = "data:image/png;base64,iVBORw0KGgoAAAANS..."
            data_content = DataContent(uri=data_uri)

            # メディアタイプをチェック
            if data_content.has_top_level_media_type("image"):
                print("This is an image")

    """

    @overload
    def __init__(
        self,
        *,
        uri: str,
        annotations: list[Annotations | MutableMapping[str, Any]] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """URIを使ってDataContentインスタンスを初期化します。

        重要:
            これはデータURIとして表現されるバイナリデータ用であり、オンラインリソース用ではありません。
            オンラインリソースには``UriContent``を使用してください。

        キーワード引数:
            uri: このインスタンスが表すデータのURI。
                形式は "data:{media_type};base64,{base64_data}" であるべきです。
            annotations: コンテンツに関連付けられたオプションの注釈。
            additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
            raw_representation: コンテンツのオプションの生の表現。
            **kwargs: その他の任意のキーワード引数。

        """

    @overload
    def __init__(
        self,
        *,
        data: bytes,
        media_type: str,
        annotations: list[Annotations | MutableMapping[str, Any]] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """バイナリデータを使ってDataContentインスタンスを初期化します。

        重要:
            これはデータURIとして表現されるバイナリデータ用であり、オンラインリソース用ではありません。
            オンラインリソースには``UriContent``を使用してください。

        キーワード引数:
            data: このインスタンスが表すバイナリデータ。
                データはbase64エンコードされたデータURIに変換されます。
            media_type: データのメディアタイプ。
            annotations: コンテンツに関連付けられたオプションの注釈。
            additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
            raw_representation: コンテンツのオプションの生の表現。
            **kwargs: その他の任意のキーワード引数。

        """

    def __init__(
        self,
        *,
        uri: str | None = None,
        data: bytes | None = None,
        media_type: str | None = None,
        annotations: list[Annotations | MutableMapping[str, Any]] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """DataContentインスタンスを初期化します。

        重要:
            これはデータURIとして表現されるバイナリデータ用であり、オンラインリソース用ではありません。
            オンラインリソースには``UriContent``を使用してください。

        キーワード引数:
            uri: このインスタンスが表すデータのURI。
                形式は "data:{media_type};base64,{base64_data}" であるべきです。
            data: このインスタンスが表すバイナリデータ。
                データはbase64エンコードされたデータURIに変換されます。
            media_type: データのメディアタイプ。
            annotations: コンテンツに関連付けられたオプションの注釈。
            additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
            raw_representation: コンテンツのオプションの生の表現。
            **kwargs: その他の任意のキーワード引数。

        """
        if uri is None:
            if data is None or media_type is None:
                raise ValueError("Either 'data' and 'media_type' or 'uri' must be provided.")
            uri = f"data:{media_type};base64,{base64.b64encode(data).decode('utf-8')}"

        # URI形式を検証し、media_typeが提供されていない場合は抽出します
        validated_uri = self._validate_uri(uri)
        if media_type is None:
            match = URI_PATTERN.match(validated_uri)
            if match:
                media_type = match.group("media_type")

        super().__init__(
            annotations=annotations,
            additional_properties=additional_properties,
            raw_representation=raw_representation,
            **kwargs,
        )
        self.uri = validated_uri
        self.media_type = media_type
        self.type: Literal["data"] = "data"

    @classmethod
    def _validate_uri(cls, uri: str) -> str:
        """URI形式を検証し、media_typeを抽出します。

        RFC 2397に基づく最小限のデータURIパーサー: https://datatracker.ietf.org/doc/html/rfc2397。

        """
        match = URI_PATTERN.match(uri)
        if not match:
            raise ValueError(f"Invalid data URI format: {uri}")
        media_type = match.group("media_type")
        if media_type not in KNOWN_MEDIA_TYPES:
            raise ValueError(f"Unknown media type: {media_type}")
        return uri

    def has_top_level_media_type(self, top_level_media_type: Literal["application", "audio", "image", "text"]) -> bool:
        return _has_top_level_media_type(self.media_type, top_level_media_type)


class UriContent(BaseContent):
    """URIコンテンツを表します。

    重要:
        これは画像やファイルなど、URIで識別されるコンテンツに使用されます。
        （バイナリ）データURIの場合は、代わりに``DataContent``を使用してください。

    属性:
        uri: コンテンツのURI、例: 'https://example.com/image.png'。
        media_type: コンテンツのメディアタイプ、例: 'image/png', 'application/json'など。
        type: コンテンツのタイプで、このクラスでは常に"uri"です。
        annotations: コンテンツに関連付けられたオプションの注釈。
        additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
        raw_representation: コンテンツのオプションの生の表現。

    Examples:
        .. code-block:: python

            from agent_framework import UriContent

            # 画像用のURIコンテンツを作成
            image_uri = UriContent(
                uri="https://example.com/image.png",
                media_type="image/png",
            )

            # ドキュメント用のURIコンテンツを作成
            doc_uri = UriContent(
                uri="https://example.com/document.pdf",
                media_type="application/pdf",
            )

            # 画像かどうかをチェック
            if image_uri.has_top_level_media_type("image"):
                print("This is an image URI")

    """

    def __init__(
        self,
        uri: str,
        media_type: str,
        *,
        annotations: list[Annotations | MutableMapping[str, Any]] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """UriContentインスタンスを初期化します。

        備考:
            これは画像やファイルなど、URIで識別されるコンテンツに使用されます。
            （バイナリ）データURIの場合は、代わりに`DataContent`を使用してください。

        引数:
            uri: コンテンツのURI。
            media_type: コンテンツのメディアタイプ。

        キーワード引数:
            annotations: コンテンツに関連付けられたオプションの注釈。
            additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
            raw_representation: コンテンツのオプションの生の表現。
            **kwargs: その他の任意のキーワード引数。

        """
        super().__init__(
            annotations=annotations,
            additional_properties=additional_properties,
            raw_representation=raw_representation,
            **kwargs,
        )
        self.uri = uri
        self.media_type = media_type
        self.type: Literal["uri"] = "uri"

    def has_top_level_media_type(self, top_level_media_type: Literal["application", "audio", "image", "text"]) -> bool:
        """指定されたトップレベルのメディアタイプを持つかどうかを示すブール値を返します。

        引数:
            top_level_media_type: チェックするトップレベルのメディアタイプ。許可される値:
                "image", "text", "application", "audio"。


        """
        return _has_top_level_media_type(self.media_type, top_level_media_type)


def _has_top_level_media_type(
    media_type: str | None, top_level_media_type: Literal["application", "audio", "image", "text"]
) -> bool:
    if media_type is None:
        return False

    slash_index = media_type.find("/")
    span = media_type[:slash_index] if slash_index >= 0 else media_type
    span = span.strip()
    return span.lower() == top_level_media_type.lower()


class ErrorContent(BaseContent):
    """エラーを表します。

    備考:
        通常は致命的でないエラーに使用され、操作の一部で問題が発生したが、操作自体は継続可能な場合に使われます。

    属性:
        error_code: エラーに関連付けられたエラーコード。
        details: エラーに関する追加の詳細。
        message: エラーメッセージ。
        type: コンテンツのタイプで、このクラスでは常に"error"です。
        annotations: コンテンツに関連付けられたオプションの注釈。
        additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
        raw_representation: コンテンツのオプションの生の表現。

    Examples:
        .. code-block:: python

            from agent_framework import ErrorContent

            # エラーコンテンツを作成
            error = ErrorContent(
                message="Failed to process request",
                error_code="PROCESSING_ERROR",
                details="The input format was invalid",
            )
            print(str(error))  # "Error PROCESSING_ERROR: Failed to process request"

            # コードなしのエラー
            simple_error = ErrorContent(message="Something went wrong")
            print(str(simple_error))  # "Something went wrong"

    """

    def __init__(
        self,
        *,
        message: str | None = None,
        error_code: str | None = None,
        details: str | None = None,
        annotations: list[Annotations | MutableMapping[str, Any]] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """ErrorContentインスタンスを初期化します。

        キーワード引数:
            message: エラーメッセージ。
            error_code: エラーに関連付けられたエラーコード。
            details: エラーに関する追加の詳細。
            annotations: コンテンツに関連付けられたオプションの注釈。
            additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
            raw_representation: コンテンツのオプションの生の表現。
            **kwargs: その他の任意のキーワード引数。

        """
        super().__init__(
            annotations=annotations,
            additional_properties=additional_properties,
            raw_representation=raw_representation,
            **kwargs,
        )
        self.message = message
        self.error_code = error_code
        self.details = details
        self.type: Literal["error"] = "error"

    def __str__(self) -> str:
        """エラーの文字列表現を返します。"""
        return f"Error {self.error_code}: {self.message}" if self.error_code else self.message or "Unknown error"


class FunctionCallContent(BaseContent):
    """関数呼び出しリクエストを表します。

    属性:
        call_id: 関数呼び出しの識別子。
        name: 要求された関数の名前。
        arguments: 関数に提供されることを要求された引数。
        exception: 元の関数呼び出しデータをこの表現にマッピングする際に発生した例外。
        type: コンテンツのタイプで、このクラスでは常に"function_call"です。
        annotations: コンテンツに関連付けられたオプションの注釈。
        additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
        raw_representation: コンテンツのオプションの生の表現。

    Examples:
        .. code-block:: python

            from agent_framework import FunctionCallContent

            # 関数呼び出しを作成
            func_call = FunctionCallContent(
                call_id="call_123",
                name="get_weather",
                arguments={"location": "Seattle", "unit": "celsius"},
            )

            # 引数を解析
            args = func_call.parse_arguments()
            print(args["location"])  # "Seattle"

            # 文字列引数で作成（段階的な補完）
            func_call_partial_1 = FunctionCallContent(
                call_id="call_124",
                name="search",
                arguments='{"query": ',
            )
            func_call_partial_2 = FunctionCallContent(
                call_id="call_124",
                name="search",
                arguments='"latest news"}',
            )
            full_call = func_call_partial_1 + func_call_partial_2
            args = full_call.parse_arguments()
            print(args["query"])  # "latest news"

    """

    def __init__(
        self,
        *,
        call_id: str,
        name: str,
        arguments: str | dict[str, Any | None] | None = None,
        exception: Exception | None = None,
        annotations: list[Annotations | MutableMapping[str, Any]] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """FunctionCallContentインスタンスを初期化します。

        キーワード引数:
            call_id: 関数呼び出しの識別子。
            name: 要求された関数の名前。
            arguments: 関数に提供されることを要求された引数。
                引数の段階的な補完を許可するために文字列であることも可能です。
            exception: 元の関数呼び出しデータをこの表現にマッピングする際に発生した例外。
            annotations: コンテンツに関連付けられたオプションの注釈。
            additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
            raw_representation: コンテンツのオプションの生の表現。
            **kwargs: その他の任意のキーワード引数。

        """
        super().__init__(
            annotations=annotations,
            additional_properties=additional_properties,
            raw_representation=raw_representation,
            **kwargs,
        )
        self.call_id = call_id
        self.name = name
        self.arguments = arguments
        self.exception = exception
        self.type: Literal["function_call"] = "function_call"

    def parse_arguments(self) -> dict[str, Any | None] | None:
        """引数を辞書に解析します。

        JSONとして解析できない場合、または解析結果が辞書でない場合は、
        "raw"という単一のキーを持つ辞書として返されます。

        """
        if isinstance(self.arguments, str):
            # 引数が文字列の場合、JSONとして解析を試みます
            try:
                loaded = json.loads(self.arguments)
                if isinstance(loaded, dict):
                    return loaded  # type:ignore
                return {"raw": loaded}
            except (json.JSONDecodeError, TypeError):
                return {"raw": self.arguments}
        return self.arguments

    def __add__(self, other: "FunctionCallContent") -> "FunctionCallContent":
        if not isinstance(other, FunctionCallContent):
            raise TypeError("Incompatible type")
        if other.call_id and self.call_id != other.call_id:
            raise AdditionItemMismatch("", log_level=None)
        if not self.arguments:
            arguments = other.arguments
        elif not other.arguments:
            arguments = self.arguments
        elif isinstance(self.arguments, str) and isinstance(other.arguments, str):
            arguments = self.arguments + other.arguments
        elif isinstance(self.arguments, dict) and isinstance(other.arguments, dict):
            arguments = {**self.arguments, **other.arguments}
        else:
            raise TypeError("Incompatible argument types")
        return FunctionCallContent(
            call_id=self.call_id,
            name=self.name,
            arguments=arguments,
            exception=self.exception or other.exception,
            additional_properties={**(self.additional_properties or {}), **(other.additional_properties or {})},
            raw_representation=self.raw_representation or other.raw_representation,
        )


class FunctionResultContent(BaseContent):
    """関数呼び出しの結果を表します。

    属性:
        call_id: この結果が対応する関数呼び出しの識別子。
        result: 関数呼び出しの結果、または関数呼び出しが失敗した場合の一般的なエラーメッセージ。
        exception: 関数呼び出しが失敗した場合に発生した例外。
        type: コンテンツのタイプで、このクラスでは常に"function_result"です。
        annotations: コンテンツに関連付けられたオプションの注釈。
        additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
        raw_representation: コンテンツのオプションの生の表現。

    Examples:
        .. code-block:: python

            from agent_framework import FunctionResultContent

            # 成功した関数結果を作成
            result = FunctionResultContent(
                call_id="call_123",
                result={"temperature": 22, "condition": "sunny"},
            )

            # 失敗した関数結果を作成
            failed_result = FunctionResultContent(
                call_id="call_124",
                result="Function execution failed",
                exception=ValueError("Invalid location"),
            )

    """

    def __init__(
        self,
        *,
        call_id: str,
        result: Any | None = None,
        exception: Exception | None = None,
        annotations: list[Annotations | MutableMapping[str, Any]] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """FunctionResultContentインスタンスを初期化します。

        キーワード引数:
            call_id: この結果が対応する関数呼び出しの識別子。
            result: 関数呼び出しの結果、または関数呼び出しが失敗した場合の一般的なエラーメッセージ。
            exception: 関数呼び出しが失敗した場合に発生した例外。
            annotations: コンテンツに関連付けられたオプションの注釈。
            additional_properties: コンテンツに関連付けられたオプションの追加プロパティ。
            raw_representation: コンテンツのオプションの生の表現。
            **kwargs: その他の任意のキーワード引数。

        """
        super().__init__(
            annotations=annotations,
            additional_properties=additional_properties,
            raw_representation=raw_representation,
            **kwargs,
        )
        self.call_id = call_id
        self.result = result
        self.exception = exception
        self.type: Literal["function_result"] = "function_result"


class UsageContent(BaseContent):
    """チャットのRequestとResponseに関連する使用情報を表します。

    Attributes:
        details: 入力および出力のToken数やその他のカウントを含む使用情報。
        type: コンテンツの種類で、このClassでは常に"usage"です。
        annotations: コンテンツに関連付けられた任意の注釈。
        additional_properties: コンテンツに関連付けられた任意の追加プロパティ。
        raw_representation: コンテンツの任意の生の表現。

    Examples:
        .. code-block:: python

            from agent_framework import UsageContent, UsageDetails

            # 使用コンテンツを作成
            usage = UsageContent(
                details=UsageDetails(
                    input_token_count=100,
                    output_token_count=50,
                    total_token_count=150,
                ),
            )
            print(usage.details.total_token_count)  # 150

    """

    def __init__(
        self,
        details: UsageDetails | MutableMapping[str, Any],
        *,
        annotations: list[Annotations | MutableMapping[str, Any]] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """UsageContentインスタンスを初期化します。"""
        super().__init__(
            annotations=annotations,
            additional_properties=additional_properties,
            raw_representation=raw_representation,
            **kwargs,
        )
        # 必要に応じてdictをUsageDetailsに変換します。
        if isinstance(details, MutableMapping):
            details = UsageDetails.from_dict(details)
        self.details = details
        self.type: Literal["usage"] = "usage"


class HostedFileContent(BaseContent):
    """ホストされたファイルコンテンツを表します。

    Attributes:
        file_id: ホストされたファイルの識別子。
        type: コンテンツの種類で、このClassでは常に"hosted_file"です。
        additional_properties: コンテンツに関連付けられた任意の追加プロパティ。
        raw_representation: コンテンツの任意の生の表現。

    Examples:
        .. code-block:: python

            from agent_framework import HostedFileContent

            # ホストされたファイルコンテンツを作成
            file_content = HostedFileContent(file_id="file-abc123")
            print(file_content.file_id)  # "file-abc123"

    """

    def __init__(
        self,
        file_id: str,
        *,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """HostedFileContentインスタンスを初期化します。

        Args:
            file_id: ホストされたファイルの識別子。

        Keyword Args:
            additional_properties: コンテンツに関連付けられた任意の追加プロパティ。
            raw_representation: コンテンツの任意の生の表現。
            **kwargs: その他の任意のキーワード引数。

        """
        super().__init__(
            additional_properties=additional_properties,
            raw_representation=raw_representation,
            **kwargs,
        )
        self.file_id = file_id
        self.type: Literal["hosted_file"] = "hosted_file"


class HostedVectorStoreContent(BaseContent):
    """ホストされたベクターストアコンテンツを表します。

    Attributes:
        vector_store_id: ホストされたベクターストアの識別子。
        type: コンテンツの種類で、このClassでは常に"hosted_vector_store"です。
        additional_properties: コンテンツに関連付けられた任意の追加プロパティ。
        raw_representation: コンテンツの任意の生の表現。

    Examples:
        .. code-block:: python

            from agent_framework import HostedVectorStoreContent

            # ホストされたベクターストアコンテンツを作成
            vs_content = HostedVectorStoreContent(vector_store_id="vs-xyz789")
            print(vs_content.vector_store_id)  # "vs-xyz789"

    """

    def __init__(
        self,
        vector_store_id: str,
        *,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """HostedVectorStoreContentインスタンスを初期化します。

        Args:
            vector_store_id: ホストされたベクターストアの識別子。

        Keyword Args:
            additional_properties: コンテンツに関連付けられた任意の追加プロパティ。
            raw_representation: コンテンツの任意の生の表現。
            **kwargs: その他の任意のキーワード引数。

        """
        super().__init__(
            additional_properties=additional_properties,
            raw_representation=raw_representation,
            **kwargs,
        )
        self.vector_store_id = vector_store_id
        self.type: Literal["hosted_vector_store"] = "hosted_vector_store"


class BaseUserInputRequest(BaseContent):
    """すべてのユーザーRequestの基底Classです。"""

    def __init__(
        self,
        *,
        id: str,
        annotations: list[Annotations | MutableMapping[str, Any]] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """BaseUserInputRequestを初期化します。

        Keyword Args:
            id: Requestの一意識別子。
            annotations: コンテンツに関連付けられた任意の注釈。
            additional_properties: コンテンツに関連付けられた任意の追加プロパティ。
            raw_representation: コンテンツの任意の生の表現。
            **kwargs: その他の任意のキーワード引数。

        """
        if not id or len(id) < 1:
            raise ValueError("id must be at least 1 character long")
        super().__init__(
            annotations=annotations,
            additional_properties=additional_properties,
            raw_representation=raw_representation,
            **kwargs,
        )
        self.id = id
        self.type: Literal["user_input_request"] = "user_input_request"


class FunctionApprovalResponseContent(BaseContent):
    """関数呼び出しのユーザー承認に対するResponseを表します。

    Examples:
        .. code-block:: python

            from agent_framework import FunctionApprovalResponseContent, FunctionCallContent

            # 関数承認Responseを作成
            func_call = FunctionCallContent(
                call_id="call_123",
                name="send_email",
                arguments={"to": "user@example.com"},
            )
            response = FunctionApprovalResponseContent(
                approved=False,
                id="approval_001",
                function_call=func_call,
            )
            print(response.approved)  # False

    """

    def __init__(
        self,
        approved: bool,
        *,
        id: str,
        function_call: FunctionCallContent | MutableMapping[str, Any],
        annotations: list[Annotations | MutableMapping[str, Any]] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """FunctionApprovalResponseContentインスタンスを初期化します。

        Args:
            approved: 関数呼び出しが承認されたかどうか。

        Keyword Args:
            id: Requestの一意識別子。
            function_call: 承認対象の関数呼び出しコンテンツ。FunctionCallContentオブジェクトまたはdictで指定可能。
            annotations: Requestに関連付けられた任意の注釈リスト。
            additional_properties: Requestに関連付けられた任意の追加プロパティ。
            raw_representation: Requestの任意の生の表現。
            **kwargs: その他のキーワード引数。

        """
        super().__init__(
            annotations=annotations,
            additional_properties=additional_properties,
            raw_representation=raw_representation,
            **kwargs,
        )
        self.id = id
        self.approved = approved
        # 必要に応じてdictをFunctionCallContentに変換します（SerializationMixin対応）。
        if isinstance(function_call, MutableMapping):
            self.function_call = FunctionCallContent.from_dict(function_call)
        else:
            self.function_call = function_call
        # この特定のサブクラスのtypeをオーバーライドします。
        self.type: Literal["function_approval_response"] = "function_approval_response"


class FunctionApprovalRequestContent(BaseContent):
    """関数呼び出しのユーザー承認Requestを表します。

    Examples:
        .. code-block:: python

            from agent_framework import FunctionApprovalRequestContent, FunctionCallContent

            # 関数承認Requestを作成
            func_call = FunctionCallContent(
                call_id="call_123",
                name="send_email",
                arguments={"to": "user@example.com", "subject": "Hello"},
            )
            approval_request = FunctionApprovalRequestContent(
                id="approval_001",
                function_call=func_call,
            )

            # Responseを作成
            approval_response = approval_request.create_response(approved=True)
            print(approval_response.approved)  # True

    """

    def __init__(
        self,
        *,
        id: str,
        function_call: FunctionCallContent | MutableMapping[str, Any],
        annotations: list[Annotations | MutableMapping[str, Any]] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """FunctionApprovalRequestContentインスタンスを初期化します。

        Keyword Args:
            id: Requestの一意識別子。
            function_call: 承認対象の関数呼び出しコンテンツ。FunctionCallContentオブジェクトまたはdictで指定可能。
            annotations: Requestに関連付けられた任意の注釈リスト。
            additional_properties: Requestに関連付けられた任意の追加プロパティ。
            raw_representation: Requestの任意の生の表現。
            **kwargs: その他のキーワード引数。

        """
        super().__init__(
            annotations=annotations,
            additional_properties=additional_properties,
            raw_representation=raw_representation,
            **kwargs,
        )
        self.id = id
        # 必要に応じてdictをFunctionCallContentに変換します（SerializationMixin対応）。
        if isinstance(function_call, MutableMapping):
            self.function_call = FunctionCallContent.from_dict(function_call)
        else:
            self.function_call = function_call
        # この特定のサブクラスのtypeをオーバーライドします。
        self.type: Literal["function_approval_request"] = "function_approval_request"

    def create_response(self, approved: bool) -> "FunctionApprovalResponseContent":
        """関数承認Requestに対するResponseを作成します。"""
        return FunctionApprovalResponseContent(
            approved,
            id=self.id,
            function_call=self.function_call,
            additional_properties=self.additional_properties,
        )


UserInputRequestContents = FunctionApprovalRequestContent

Contents = (
    TextContent
    | DataContent
    | TextReasoningContent
    | UriContent
    | FunctionCallContent
    | FunctionResultContent
    | ErrorContent
    | UsageContent
    | HostedFileContent
    | HostedVectorStoreContent
    | FunctionApprovalRequestContent
    | FunctionApprovalResponseContent
)


def _prepare_function_call_results_as_dumpable(content: Contents | Any | list[Contents | Any]) -> Any:
    if isinstance(content, list):
        # 特にContentのリストを処理します。
        return [_prepare_function_call_results_as_dumpable(item) for item in content]
    if isinstance(content, dict):
        return {k: _prepare_function_call_results_as_dumpable(v) for k, v in content.items()}
    if hasattr(content, "to_dict"):
        return content.to_dict(exclude={"raw_representation", "additional_properties"})
    return content


def prepare_function_call_results(content: Contents | Any | list[Contents | Any]) -> str:
    """関数呼び出し結果の値を準備します。"""
    if isinstance(content, Contents):
        # BaseContentオブジェクトの場合、to_dictを使いJSONにシリアライズします。
        return json.dumps(content.to_dict(exclude={"raw_representation", "additional_properties"}))

    dumpable = _prepare_function_call_results_as_dumpable(content)
    if isinstance(dumpable, str):
        return dumpable
    # フォールバック
    return json.dumps(dumpable)


# region Chat Response constants


class Role(SerializationMixin, metaclass=EnumLike):
    """チャット対話内のメッセージの意図された役割を表します。

    Attributes:
        value: 役割の文字列表現。

    Properties:
        SYSTEM: AIシステムの動作を指示または設定する役割。
        USER: チャット対話に対するユーザー入力を提供する役割。
        ASSISTANT: システム指示やユーザープロンプトに応答する役割。
        TOOL: ツール使用要求に応じて追加情報や参照を提供する役割。

    Examples:
        .. code-block:: python

            from agent_framework import Role

            # 定義済み役割定数を使用
            system_role = Role.SYSTEM
            user_role = Role.USER
            assistant_role = Role.ASSISTANT
            tool_role = Role.TOOL

            # カスタム役割を作成
            custom_role = Role(value="custom")

            # 役割を比較
            print(system_role == Role.SYSTEM)  # True
            print(system_role.value)  # "system"

    """

    # EnumLikeメタクラスの定数設定
    _constants: ClassVar[dict[str, str]] = {
        "SYSTEM": "system",
        "USER": "user",
        "ASSISTANT": "assistant",
        "TOOL": "tool",
    }

    # 定数の型注釈
    SYSTEM: "Role"
    USER: "Role"
    ASSISTANT: "Role"
    TOOL: "Role"

    def __init__(self, value: str) -> None:
        """Roleを値で初期化します。

        Args:
            value: 役割の文字列表現。

        """
        self.value = value

    def __str__(self) -> str:
        """役割の文字列表現を返します。"""
        return self.value

    def __repr__(self) -> str:
        """役割の文字列表現を返します。"""
        return f"Role(value={self.value!r})"

    def __eq__(self, other: object) -> bool:
        """2つのRoleインスタンスが等しいかどうかをチェックします。"""
        if not isinstance(other, Role):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        """セットやdictでの使用のためRoleのハッシュ値を返します。"""
        return hash(self.value)


class FinishReason(SerializationMixin, metaclass=EnumLike):
    """チャットResponseが完了した理由を表します。

    Attributes:
        value: 完了理由の文字列表現。

    Examples:
        .. code-block:: python

            from agent_framework import FinishReason

            # 定義済み完了理由定数を使用
            stop_reason = FinishReason.STOP  # 正常完了
            length_reason = FinishReason.LENGTH  # 最大トークン数到達
            tool_calls_reason = FinishReason.TOOL_CALLS  # ツール呼び出し発生
            filter_reason = FinishReason.CONTENT_FILTER  # コンテンツフィルター発動

            # 完了理由をチェック
            if stop_reason == FinishReason.STOP:
                print("Response completed normally")

    """

    # EnumLikeメタクラスの定数設定
    _constants: ClassVar[dict[str, str]] = {
        "CONTENT_FILTER": "content_filter",
        "LENGTH": "length",
        "STOP": "stop",
        "TOOL_CALLS": "tool_calls",
    }

    # 定数の型注釈
    CONTENT_FILTER: "FinishReason"
    LENGTH: "FinishReason"
    STOP: "FinishReason"
    TOOL_CALLS: "FinishReason"

    def __init__(self, value: str) -> None:
        """FinishReasonを値で初期化します。

        Args:
            value: 完了理由の文字列表現。

        """
        self.value = value

    def __eq__(self, other: object) -> bool:
        """2つのFinishReasonインスタンスが等しいかどうかをチェックします。"""
        if not isinstance(other, FinishReason):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        """セットやdictでの使用のためFinishReasonのハッシュ値を返します。"""
        return hash(self.value)

    def __str__(self) -> str:
        """完了理由の文字列表現を返します。"""
        return self.value

    def __repr__(self) -> str:
        """完了理由の文字列表現を返します。"""
        return f"FinishReason(value={self.value!r})"


# region ChatMessage


class ChatMessage(SerializationMixin):
    """チャットメッセージを表します。

    属性:
        role: メッセージの作成者の役割。
        contents: チャットメッセージの内容アイテム。
        author_name: メッセージの作成者の名前。
        message_id: チャットメッセージのID。
        additional_properties: チャットメッセージに関連付けられた追加のプロパティ。
        raw_representation: 基盤となる実装からのチャットメッセージの生の表現。

    Examples:
        .. code-block:: python

            from agent_framework import ChatMessage, TextContent

            # テキストでメッセージを作成
            user_msg = ChatMessage(role="user", text="What's the weather?")
            print(user_msg.text)  # "What's the weather?"

            # 役割文字列でメッセージを作成
            system_msg = ChatMessage(role="system", text="You are a helpful assistant.")

            # contentsでメッセージを作成
            assistant_msg = ChatMessage(
                role="assistant",
                contents=[TextContent(text="The weather is sunny!")],
            )
            print(assistant_msg.text)  # "The weather is sunny!"

            # シリアライズ - to_dict と from_dict
            msg_dict = user_msg.to_dict()
            # {'type': 'chat_message', 'role': {'type': 'role', 'value': 'user'},
            #  'contents': [{'type': 'text', 'text': "What's the weather?"}], 'additional_properties': {}}
            restored_msg = ChatMessage.from_dict(msg_dict)
            print(restored_msg.text)  # "What's the weather?"

            # シリアライズ - to_json と from_json
            msg_json = user_msg.to_json()
            # '{"type": "chat_message", "role": {"type": "role", "value": "user"}, "contents": [...], ...}'
            restored_from_json = ChatMessage.from_json(msg_json)
            print(restored_from_json.role.value)  # "user"


    """

    DEFAULT_EXCLUDE: ClassVar[set[str]] = {"raw_representation"}

    @overload
    def __init__(
        self,
        role: Role | Literal["system", "user", "assistant", "tool"],
        *,
        text: str,
        author_name: str | None = None,
        message_id: str | None = None,
        additional_properties: MutableMapping[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """role とテキストコンテンツで ChatMessage を初期化します。

        引数:
            role: メッセージの作成者の役割。

        キーワード引数:
            text: メッセージのテキストコンテンツ。
            author_name: メッセージの作成者の名前（任意）。
            message_id: チャットメッセージのID（任意）。
            additional_properties: チャットメッセージに関連付けられた追加のプロパティ（任意）。
            raw_representation: チャットメッセージの生の表現（任意）。
            **kwargs: その他のキーワード引数。

        """

    @overload
    def __init__(
        self,
        role: Role | Literal["system", "user", "assistant", "tool"],
        *,
        contents: Sequence[Contents | Mapping[str, Any]],
        author_name: str | None = None,
        message_id: str | None = None,
        additional_properties: MutableMapping[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """role とオプションの contents で ChatMessage を初期化します。

        引数:
            role: メッセージの作成者の役割。

        キーワード引数:
            contents: メッセージに含める BaseContent アイテムのリスト（任意）。
            author_name: メッセージの作成者の名前（任意）。
            message_id: チャットメッセージのID（任意）。
            additional_properties: チャットメッセージに関連付けられた追加のプロパティ（任意）。
            raw_representation: チャットメッセージの生の表現（任意）。
            **kwargs: その他のキーワード引数。

        """

    def __init__(
        self,
        role: Role | Literal["system", "user", "assistant", "tool"] | dict[str, Any],
        *,
        text: str | None = None,
        contents: Sequence[Contents | Mapping[str, Any]] | None = None,
        author_name: str | None = None,
        message_id: str | None = None,
        additional_properties: MutableMapping[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """ChatMessage を初期化します。

        引数:
            role: メッセージの作成者の役割（Role、文字列、または辞書）。

        キーワード引数:
            text: メッセージのテキストコンテンツ（任意）。
            contents: メッセージに含める BaseContent アイテムまたは辞書のリスト（任意）。
            author_name: メッセージの作成者の名前（任意）。
            message_id: チャットメッセージのID（任意）。
            additional_properties: チャットメッセージに関連付けられた追加のプロパティ（任意）。
            raw_representation: チャットメッセージの生の表現（任意）。
            kwargs: 提供された場合、additional_properties と結合されます。

        """
        # role の変換を処理します
        if isinstance(role, dict):
            role = Role.from_dict(role)
        elif isinstance(role, str):
            role = Role(value=role)

        # contents の変換を処理します
        parsed_contents = [] if contents is None else _parse_content_list(contents)

        if text is not None:
            parsed_contents.append(TextContent(text=text))

        self.role = role
        self.contents = parsed_contents
        self.author_name = author_name
        self.message_id = message_id
        self.additional_properties = additional_properties or {}
        self.additional_properties.update(kwargs or {})
        self.raw_representation = raw_representation

    @property
    def text(self) -> str:
        """メッセージのテキストコンテンツを返します。

        備考:
            このプロパティは Contents 内のすべての TextContent オブジェクトのテキストを連結します。

        """
        return " ".join(content.text for content in self.contents if isinstance(content, TextContent))


# region ChatResponse


def _process_update(
    response: "ChatResponse | AgentRunResponse", update: "ChatResponseUpdate | AgentRunResponseUpdate"
) -> None:
    """単一のアップデートを処理し、レスポンスをその場で変更します。"""
    is_new_message = False
    if (
        not response.messages
        or (
            update.message_id
            and response.messages[-1].message_id
            and response.messages[-1].message_id != update.message_id
        )
        or (update.role and response.messages[-1].role != update.role)
    ):
        is_new_message = True

    if is_new_message:
        message = ChatMessage(role=Role.ASSISTANT, contents=[])
        response.messages.append(message)
    else:
        message = response.messages[-1]
    # アップデートのプロパティをメッセージに組み込みます。
    if update.author_name is not None:
        message.author_name = update.author_name
    if update.role is not None:
        message.role = update.role
    if update.message_id:
        message.message_id = update.message_id
    for content in update.contents:
        if (
            isinstance(content, FunctionCallContent)
            and len(message.contents) > 0
            and isinstance(message.contents[-1], FunctionCallContent)
        ):
            try:
                message.contents[-1] += content
            except AdditionItemMismatch:
                message.contents.append(content)
        elif isinstance(content, UsageContent):
            if response.usage_details is None:
                response.usage_details = UsageDetails()
            response.usage_details += content.details
        elif isinstance(content, (dict, MutableMapping)):
            try:
                cont = _parse_content(content)
                message.contents.append(cont)
            except ContentError as exc:
                logger.warning(f"Skipping unknown content type or invalid content: {exc}")
        else:
            message.contents.append(content)
    # アップデートのプロパティをレスポンスに組み込みます。
    if update.response_id:
        response.response_id = update.response_id
    if update.created_at is not None:
        response.created_at = update.created_at
    if update.additional_properties is not None:
        if response.additional_properties is None:
            response.additional_properties = {}
        response.additional_properties.update(update.additional_properties)
    if response.raw_representation is None:
        response.raw_representation = []
    if not isinstance(response.raw_representation, list):
        response.raw_representation = [response.raw_representation]
    response.raw_representation.append(update.raw_representation)
    if isinstance(response, ChatResponse) and isinstance(update, ChatResponseUpdate):
        if update.conversation_id is not None:
            response.conversation_id = update.conversation_id
        if update.finish_reason is not None:
            response.finish_reason = update.finish_reason
        if update.model_id is not None:
            response.model_id = update.model_id


def _coalesce_text_content(
    contents: list["Contents"], type_: type["TextContent"] | type["TextReasoningContent"]
) -> None:
    """任意の連続する Text または TextReasoningContent アイテムを取りまとめて単一のアイテムにします。"""
    if not contents:
        return
    coalesced_contents: list["Contents"] = []
    first_new_content: Any | None = None
    for content in contents:
        if isinstance(content, type_):
            if first_new_content is None:
                first_new_content = deepcopy(content)
            else:
                first_new_content += content
        else:
            # このコンテンツは適切なタイプではないためスキップします 既存のものをリストに書き込み、新しいものを開始します 適切なタイプが再び見つかるまで
            if first_new_content:
                coalesced_contents.append(first_new_content)
            first_new_content = None
            # しかし他のコンテンツは新しいリストに保持します
            coalesced_contents.append(content)
    if first_new_content:
        coalesced_contents.append(first_new_content)
    contents.clear()
    contents.extend(coalesced_contents)


def _finalize_response(response: "ChatResponse | AgentRunResponse") -> None:
    """必要な後処理を行いレスポンスを最終化します。"""
    for msg in response.messages:
        _coalesce_text_content(msg.contents, TextContent)
        _coalesce_text_content(msg.contents, TextReasoningContent)


class ChatResponse(SerializationMixin):
    """チャットリクエストへのレスポンスを表します。

    属性:
        messages: レスポンス内のチャットメッセージのリスト。
        response_id: チャットレスポンスのID。
        conversation_id: 会話の状態を識別するID。
        model_id: チャットレスポンスの作成に使用されたモデルID。
        created_at: チャットレスポンスのタイムスタンプ。
        finish_reason: チャットレスポンスの終了理由。
        usage_details: チャットレスポンスの使用詳細。
        structured_output: チャットレスポンスの構造化出力（該当する場合）。
        additional_properties: チャットレスポンスに関連付けられた追加のプロパティ。
        raw_representation: 基盤となる実装からのチャットレスポンスの生の表現。

    Examples:
        .. code-block:: python

            from agent_framework import ChatResponse, ChatMessage

            # シンプルなテキストレスポンスを作成
            response = ChatResponse(text="Hello, how can I help you?")
            print(response.text)  # "Hello, how can I help you?"

            # メッセージ付きのレスポンスを作成
            msg = ChatMessage(role="assistant", text="The weather is sunny.")
            response = ChatResponse(
                messages=[msg],
                finish_reason="stop",
                model_id="gpt-4",
            )

            # ストリーミングアップデートを結合
            updates = [...]  # ChatResponseUpdate オブジェクトのリスト
            response = ChatResponse.from_chat_response_updates(updates)

            # シリアライズ - to_dict と from_dict
            response_dict = response.to_dict()
            # {'type': 'chat_response', 'messages': [...], 'model_id': 'gpt-4',
            #  'finish_reason': {'type': 'finish_reason', 'value': 'stop'}}
            restored_response = ChatResponse.from_dict(response_dict)
            print(restored_response.model_id)  # "gpt-4"

            # シリアライズ - to_json と from_json
            response_json = response.to_json()
            # '{"type": "chat_response", "messages": [...], "model_id": "gpt-4", ...}'
            restored_from_json = ChatResponse.from_json(response_json)
            print(restored_from_json.text)  # "The weather is sunny."

    """

    DEFAULT_EXCLUDE: ClassVar[set[str]] = {"raw_representation", "additional_properties"}

    @overload
    def __init__(
        self,
        *,
        messages: ChatMessage | MutableSequence[ChatMessage],
        response_id: str | None = None,
        conversation_id: str | None = None,
        model_id: str | None = None,
        created_at: CreatedAtT | None = None,
        finish_reason: FinishReason | None = None,
        usage_details: UsageDetails | None = None,
        value: Any | None = None,
        response_format: type[BaseModel] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """提供されたパラメータで ChatResponse を初期化します。

        キーワード引数:
            messages: レスポンスに含める単一の ChatMessage または ChatMessage オブジェクトのシーケンス。
            response_id: チャットレスポンスのID（任意）。
            conversation_id: 会話の状態を識別するID（任意）。
            model_id: チャットレスポンスの作成に使用されたモデルID（任意）。
            created_at: チャットレスポンスのタイムスタンプ（任意）。
            finish_reason: チャットレスポンスの終了理由（任意）。
            usage_details: チャットレスポンスの使用詳細（任意）。
            value: 構造化出力の値（任意）。
            response_format: チャットレスポンスのレスポンスフォーマット（任意）。
            messages: レスポンスに含める ChatMessage オブジェクトのリスト。
            additional_properties: チャットレスポンスに関連付けられた追加のプロパティ（任意）。
            raw_representation: 基盤となる実装からのチャットレスポンスの生の表現（任意）。
            **kwargs: その他のキーワード引数。

        """

    @overload
    def __init__(
        self,
        *,
        text: TextContent | str,
        response_id: str | None = None,
        conversation_id: str | None = None,
        model_id: str | None = None,
        created_at: CreatedAtT | None = None,
        finish_reason: FinishReason | None = None,
        usage_details: UsageDetails | None = None,
        value: Any | None = None,
        response_format: type[BaseModel] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """提供されたパラメータで ChatResponse を初期化します。

        キーワード引数:
            text: レスポンスに含めるテキストコンテンツ。提供された場合、ChatMessage として追加されます。
            response_id: チャットレスポンスのID（任意）。
            conversation_id: 会話の状態を識別するID（任意）。
            model_id: チャットレスポンスの作成に使用されたモデルID（任意）。
            created_at: チャットレスポンスのタイムスタンプ（任意）。
            finish_reason: チャットレスポンスの終了理由（任意）。
            usage_details: チャットレスポンスの使用詳細（任意）。
            value: 構造化出力の値（任意）。
            response_format: チャットレスポンスのレスポンスフォーマット（任意）。
            additional_properties: チャットレスポンスに関連付けられた追加のプロパティ（任意）。
            raw_representation: 基盤となる実装からのチャットレスポンスの生の表現（任意）。
            **kwargs: その他のキーワード引数。


        """

    def __init__(
        self,
        *,
        messages: ChatMessage | MutableSequence[ChatMessage] | list[dict[str, Any]] | None = None,
        text: TextContent | str | None = None,
        response_id: str | None = None,
        conversation_id: str | None = None,
        model_id: str | None = None,
        created_at: CreatedAtT | None = None,
        finish_reason: FinishReason | dict[str, Any] | None = None,
        usage_details: UsageDetails | dict[str, Any] | None = None,
        value: Any | None = None,
        response_format: type[BaseModel] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """提供されたパラメータで ChatResponse を初期化します。

        キーワード引数:
            messages: レスポンスに含める単一の ChatMessage または ChatMessage オブジェクトのシーケンス。
            text: レスポンスに含めるテキストコンテンツ。提供された場合、ChatMessage として追加されます。
            response_id: チャットレスポンスのID（任意）。
            conversation_id: 会話の状態を識別するID（任意）。
            model_id: チャットレスポンスの作成に使用されたモデルID（任意）。
            created_at: チャットレスポンスのタイムスタンプ（任意）。
            finish_reason: チャットレスポンスの終了理由（任意）。
            usage_details: チャットレスポンスの使用詳細（任意）。
            value: 構造化出力の値（任意）。
            response_format: チャットレスポンスのレスポンスフォーマット（任意）。
            additional_properties: チャットレスポンスに関連付けられた追加のプロパティ（任意）。
            raw_representation: 基盤となる実装からのチャットレスポンスの生の表現（任意）。
            **kwargs: その他のキーワード引数。

        """
        # messages の変換を処理します
        if messages is None:
            messages = []
        elif not isinstance(messages, MutableSequence):
            messages = [messages]
        else:
            # messages リスト内の辞書を ChatMessage オブジェクトに変換します
            converted_messages: list[ChatMessage] = []
            for msg in messages:
                if isinstance(msg, dict):
                    converted_messages.append(ChatMessage.from_dict(msg))
                else:
                    converted_messages.append(msg)
            messages = converted_messages

        if text is not None:
            if isinstance(text, str):
                text = TextContent(text=text)
            messages.append(ChatMessage(role=Role.ASSISTANT, contents=[text]))

        # finish_reason の変換を処理します
        if isinstance(finish_reason, dict):
            finish_reason = FinishReason.from_dict(finish_reason)

        # usage_details の変換を処理します
        if isinstance(usage_details, dict):
            usage_details = UsageDetails.from_dict(usage_details)

        self.messages = list(messages)
        self.response_id = response_id
        self.conversation_id = conversation_id
        self.model_id = model_id
        self.created_at = created_at
        self.finish_reason = finish_reason
        self.usage_details = usage_details
        self.value = value
        self.additional_properties = additional_properties or {}
        self.additional_properties.update(kwargs or {})
        self.raw_representation: Any | list[Any] | None = raw_representation

        if response_format:
            self.try_parse_value(output_format_type=response_format)

    @classmethod
    def from_chat_response_updates(
        cls: type[TChatResponse],
        updates: Sequence["ChatResponseUpdate"],
        *,
        output_format_type: type[BaseModel] | None = None,
    ) -> TChatResponse:
        """複数の更新を1つのChatResponseに結合します。

        Example:
            .. code-block:: python

                from agent_framework import ChatResponse, ChatResponseUpdate

                # いくつかのresponse更新を作成
                updates = [
                    ChatResponseUpdate(role="assistant", text="Hello"),
                    ChatResponseUpdate(text=" How can I help you?"),
                ]

                # 更新を1つのChatResponseに結合
                response = ChatResponse.from_chat_response_updates(updates)
                print(response.text)  # "Hello How can I help you?"

        Args:
            updates: 結合するChatResponseUpdateオブジェクトのシーケンス。

        Keyword Args:
            output_format_type: レスポンステキストを構造化データに解析するためのOptionalなPydanticモデルタイプ。
        """
        msg = cls(messages=[])
        for update in updates:
            _process_update(msg, update)
        _finalize_response(msg)
        if output_format_type:
            msg.try_parse_value(output_format_type)
        return msg

    @classmethod
    async def from_chat_response_generator(
        cls: type[TChatResponse],
        updates: AsyncIterable["ChatResponseUpdate"],
        *,
        output_format_type: type[BaseModel] | None = None,
    ) -> TChatResponse:
        """複数の更新を1つのChatResponseに結合します。

        Example:
            .. code-block:: python

                from agent_framework import ChatResponse, ChatResponseUpdate, ChatClient

                client = ChatClient()  # 具体的な実装であるべき
                response = await ChatResponse.from_chat_response_generator(
                    client.get_streaming_response("Hello, how are you?")
                )
                print(response.text)

        Args:
            updates: 結合するChatResponseUpdateオブジェクトの非同期イテラブル。

        Keyword Args:
            output_format_type: レスポンステキストを構造化データに解析するためのOptionalなPydanticモデルタイプ。
        """
        msg = cls(messages=[])
        async for update in updates:
            _process_update(msg, update)
        _finalize_response(msg)
        if output_format_type:
            msg.try_parse_value(output_format_type)
        return msg

    @property
    def text(self) -> str:
        """レスポンス内のすべてのメッセージの連結されたテキストを返します。"""
        return ("\n".join(message.text for message in self.messages if isinstance(message, ChatMessage))).strip()

    def __str__(self) -> str:
        return self.text

    def try_parse_value(self, output_format_type: type[BaseModel]) -> None:
        """値が存在する場合は何もしません。そうでなければテキストを値に解析しようとします。"""
        if self.value is None:
            try:
                self.value = output_format_type.model_validate_json(self.text)  # type: ignore[reportUnknownMemberType]
            except ValidationError as ex:
                logger.debug("Failed to parse value from chat response text: %s", ex)


# region ChatResponseUpdate


class ChatResponseUpdate(SerializationMixin):
    """`ChatClient`からの単一のストリーミングレスポンスチャンクを表します。

    Attributes:
        contents: チャットレスポンス更新のコンテンツ項目。
        role: レスポンス更新の作成者の役割。
        author_name: レスポンス更新の作成者の名前。
        response_id: この更新が属するレスポンスのID。
        message_id: この更新が属するメッセージのID。
        conversation_id: この更新が属する会話の状態を識別するID。
        model_id: このレスポンス更新に関連付けられたモデルID。
        created_at: チャットレスポンス更新のタイムスタンプ。
        finish_reason: 操作の終了理由。
        additional_properties: チャットレスポンス更新に関連付けられた追加のプロパティ。
        raw_representation: 基盤となる実装からのチャットレスポンス更新の生の表現。

    Examples:
        .. code-block:: python

            from agent_framework import ChatResponseUpdate, TextContent

            # レスポンス更新を作成
            update = ChatResponseUpdate(
                contents=[TextContent(text="Hello")],
                role="assistant",
                message_id="msg_123",
            )
            print(update.text)  # "Hello"

            # テキスト省略形で更新を作成
            update = ChatResponseUpdate(text="World!", role="assistant")

            # シリアライズ - to_dict と from_dict
            update_dict = update.to_dict()
            # {'type': 'chat_response_update', 'contents': [{'type': 'text', 'text': 'Hello'}],
            #  'role': {'type': 'role', 'value': 'assistant'}, 'message_id': 'msg_123'}
            restored_update = ChatResponseUpdate.from_dict(update_dict)
            print(restored_update.text)  # "Hello"

            # シリアライズ - to_json と from_json
            update_json = update.to_json()
            # '{"type": "chat_response_update", "contents": [{"type": "text", "text": "Hello"}], ...}'
            restored_from_json = ChatResponseUpdate.from_json(update_json)
            print(restored_from_json.message_id)  # "msg_123"
    """

    DEFAULT_EXCLUDE: ClassVar[set[str]] = {"raw_representation"}

    def __init__(
        self,
        *,
        contents: Sequence[Contents | dict[str, Any]] | None = None,
        text: TextContent | str | None = None,
        role: Role | Literal["system", "user", "assistant", "tool"] | dict[str, Any] | None = None,
        author_name: str | None = None,
        response_id: str | None = None,
        message_id: str | None = None,
        conversation_id: str | None = None,
        model_id: str | None = None,
        created_at: CreatedAtT | None = None,
        finish_reason: FinishReason | dict[str, Any] | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """提供されたパラメータでChatResponseUpdateを初期化します。

        Keyword Args:
            contents: 更新に含めるOptionalなBaseContentアイテムまたは辞書のリスト。
            text: 更新に含めるOptionalなテキストコンテンツ。
            role: レスポンス更新の作成者のOptionalな役割（Role、文字列、または辞書）。
            author_name: レスポンス更新の作成者のOptionalな名前。
            response_id: この更新が属するレスポンスのOptionalなID。
            message_id: この更新が属するメッセージのOptionalなID。
            conversation_id: この更新が属する会話の状態を識別するOptionalなID。
            model_id: このレスポンス更新に関連付けられたOptionalなモデルID。
            created_at: チャットレスポンス更新のOptionalなタイムスタンプ。
            finish_reason: 操作のOptionalな終了理由。
            additional_properties: チャットレスポンス更新に関連付けられたOptionalな追加プロパティ。
            raw_representation: 基盤となる実装からのチャットレスポンス更新のOptionalな生の表現。
            **kwargs: その他の任意のキーワード引数。


        """
        # contentsの変換を処理します
        contents = [] if contents is None else _parse_content_list(contents)

        if text is not None:
            if isinstance(text, str):
                text = TextContent(text=text)
            contents.append(text)

        # roleの変換を処理します
        if isinstance(role, dict):
            role = Role.from_dict(role)
        elif isinstance(role, str):
            role = Role(value=role)

        # finish_reasonの変換を処理します
        if isinstance(finish_reason, dict):
            finish_reason = FinishReason.from_dict(finish_reason)

        self.contents = list(contents)
        self.role = role
        self.author_name = author_name
        self.response_id = response_id
        self.message_id = message_id
        self.conversation_id = conversation_id
        self.model_id = model_id
        self.created_at = created_at
        self.finish_reason = finish_reason
        self.additional_properties = additional_properties
        self.raw_representation = raw_representation

    @property
    def text(self) -> str:
        """更新内のすべてのcontentsの連結されたテキストを返します。"""
        return "".join(content.text for content in self.contents if isinstance(content, TextContent))

    def __str__(self) -> str:
        return self.text


# region AgentRunResponse


class AgentRunResponse(SerializationMixin):
    """Agentのrunリクエストに対するレスポンスを表します。

    1つ以上のレスポンスメッセージとレスポンスに関するメタデータを提供します。
    通常のレスポンスは単一のメッセージを含みますが、関数呼び出し、RAG取得、複雑なロジックを含むシナリオでは複数のメッセージを含む場合があります。

    Examples:
        .. code-block:: python

            from agent_framework import AgentRunResponse, ChatMessage

            # エージェントレスポンスを作成
            msg = ChatMessage(role="assistant", text="Task completed successfully.")
            response = AgentRunResponse(messages=[msg], response_id="run_123")
            print(response.text)  # "Task completed successfully."

            # ユーザー入力リクエストにアクセス
            user_requests = response.user_input_requests
            print(len(user_requests))  # 0

            # ストリーミング更新を結合
            updates = [...]  # AgentRunResponseUpdateオブジェクトのリスト
            response = AgentRunResponse.from_agent_run_response_updates(updates)

            # シリアライズ - to_dict と from_dict
            response_dict = response.to_dict()
            # {'type': 'agent_run_response', 'messages': [...], 'response_id': 'run_123',
            #  'additional_properties': {}}
            restored_response = AgentRunResponse.from_dict(response_dict)
            print(restored_response.response_id)  # "run_123"

            # シリアライズ - to_json と from_json
            response_json = response.to_json()
            # '{"type": "agent_run_response", "messages": [...], "response_id": "run_123", ...}'
            restored_from_json = AgentRunResponse.from_json(response_json)
            print(restored_from_json.text)  # "Task completed successfully."
    """

    DEFAULT_EXCLUDE: ClassVar[set[str]] = {"raw_representation"}

    def __init__(
        self,
        *,
        messages: ChatMessage
        | list[ChatMessage]
        | MutableMapping[str, Any]
        | list[MutableMapping[str, Any]]
        | None = None,
        response_id: str | None = None,
        created_at: CreatedAtT | None = None,
        usage_details: UsageDetails | MutableMapping[str, Any] | None = None,
        value: Any | None = None,
        raw_representation: Any | None = None,
        additional_properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """AgentRunResponseを初期化します。

        Keyword Args:
            messages: レスポンス内のチャットメッセージのリスト。
            response_id: チャットレスポンスのID。
            created_at: チャットレスポンスのタイムスタンプ。
            usage_details: チャットレスポンスの使用詳細。
            value: 適用可能な場合のagent runレスポンスの構造化出力。
            additional_properties: チャットレスポンスに関連付けられた追加のプロパティ。
            raw_representation: 基盤となる実装からのチャットレスポンスの生の表現。
            **kwargs: レスポンスに設定する追加のプロパティ。

        """
        processed_messages: list[ChatMessage] = []
        if messages is not None:
            if isinstance(messages, ChatMessage):
                processed_messages.append(messages)
            elif isinstance(messages, list):
                for message_data in messages:
                    if isinstance(message_data, ChatMessage):
                        processed_messages.append(message_data)
                    elif isinstance(message_data, MutableMapping):
                        processed_messages.append(ChatMessage.from_dict(message_data))
                    else:
                        logger.warning(f"Unknown message content: {message_data}")
            elif isinstance(messages, MutableMapping):
                processed_messages.append(ChatMessage.from_dict(messages))

        # 必要に応じてusage_detailsを辞書から変換します（SerializationMixinサポートのため）
        if isinstance(usage_details, MutableMapping):
            usage_details = UsageDetails.from_dict(usage_details)

        self.messages = processed_messages
        self.response_id = response_id
        self.created_at = created_at
        self.usage_details = usage_details
        self.value = value
        self.additional_properties = additional_properties or {}
        self.additional_properties.update(kwargs or {})
        self.raw_representation = raw_representation

    @property
    def text(self) -> str:
        """すべてのメッセージの連結されたテキストを取得します。"""
        return "".join(msg.text for msg in self.messages) if self.messages else ""

    @property
    def user_input_requests(self) -> list[UserInputRequestContents]:
        """レスポンスからすべてのBaseUserInputRequestメッセージを取得します。"""
        return [
            content
            for msg in self.messages
            for content in msg.contents
            if isinstance(content, UserInputRequestContents)
        ]

    @classmethod
    def from_agent_run_response_updates(
        cls: type[TAgentRunResponse],
        updates: Sequence["AgentRunResponseUpdate"],
        *,
        output_format_type: type[BaseModel] | None = None,
    ) -> TAgentRunResponse:
        """複数の更新を1つのAgentRunResponseに結合します。

        Args:
            updates: 結合するAgentRunResponseUpdateオブジェクトのシーケンス。

        Keyword Args:
            output_format_type: レスポンステキストを構造化データに解析するためのOptionalなPydanticモデルタイプ。
        """
        msg = cls(messages=[])
        for update in updates:
            _process_update(msg, update)
        _finalize_response(msg)
        if output_format_type:
            msg.try_parse_value(output_format_type)
        return msg

    @classmethod
    async def from_agent_response_generator(
        cls: type[TAgentRunResponse],
        updates: AsyncIterable["AgentRunResponseUpdate"],
        *,
        output_format_type: type[BaseModel] | None = None,
    ) -> TAgentRunResponse:
        """複数の更新を1つのAgentRunResponseに結合します。

        Args:
            updates: 結合するAgentRunResponseUpdateオブジェクトの非同期イテラブル。

        Keyword Args:
            output_format_type: レスポンステキストを構造化データに解析するためのOptionalなPydanticモデルタイプ。
        """
        msg = cls(messages=[])
        async for update in updates:
            _process_update(msg, update)
        _finalize_response(msg)
        if output_format_type:
            msg.try_parse_value(output_format_type)
        return msg

    def __str__(self) -> str:
        return self.text

    def try_parse_value(self, output_format_type: type[BaseModel]) -> None:
        """値が存在する場合は何もしません。そうでなければテキストを値に解析しようとします。"""
        if self.value is None:
            try:
                self.value = output_format_type.model_validate_json(self.text)  # type: ignore[reportUnknownMemberType]
            except ValidationError as ex:
                logger.debug("Failed to parse value from agent run response text: %s", ex)


# region AgentRunResponseUpdate


class AgentRunResponseUpdate(SerializationMixin):
    """Agentからの単一のストリーミングレスポンスチャンクを表します。

    Examples:
        .. code-block:: python

            from agent_framework import AgentRunResponseUpdate, TextContent

            # エージェントラン更新を作成
            update = AgentRunResponseUpdate(
                contents=[TextContent(text="Processing...")],
                role="assistant",
                response_id="run_123",
            )
            print(update.text)  # "Processing..."

            # ユーザー入力リクエストをチェック
            user_requests = update.user_input_requests

            # シリアライズ - to_dict と from_dict
            update_dict = update.to_dict()
            # {'type': 'agent_run_response_update', 'contents': [{'type': 'text', 'text': 'Processing...'}],
            #  'role': {'type': 'role', 'value': 'assistant'}, 'response_id': 'run_123'}
            restored_update = AgentRunResponseUpdate.from_dict(update_dict)
            print(restored_update.response_id)  # "run_123"

            # シリアライズ - to_json と from_json
            update_json = update.to_json()
            # '{"type": "agent_run_response_update", "contents": [{"type": "text", "text": "Processing..."}], ...}'
            restored_from_json = AgentRunResponseUpdate.from_json(update_json)
            print(restored_from_json.text)  # "Processing..."
    """

    DEFAULT_EXCLUDE: ClassVar[set[str]] = {"raw_representation"}

    def __init__(
        self,
        *,
        contents: Sequence[Contents | MutableMapping[str, Any]] | None = None,
        text: TextContent | str | None = None,
        role: Role | MutableMapping[str, Any] | str | None = None,
        author_name: str | None = None,
        response_id: str | None = None,
        message_id: str | None = None,
        created_at: CreatedAtT | None = None,
        additional_properties: MutableMapping[str, Any] | None = None,
        raw_representation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """AgentRunResponseUpdateを初期化します。

        キーワード引数:
            contents: 更新に含めるBaseContentアイテムまたは辞書のオプションリスト。
            text: 更新のテキストコンテンツ（オプション）。
            role: レスポンス更新の作成者の役割（Role、文字列、または辞書）。
            author_name: レスポンス更新の作成者名（オプション）。
            response_id: この更新が属するレスポンスのID（オプション）。
            message_id: この更新が属するメッセージのID（オプション）。
            created_at: チャットレスポンス更新のタイムスタンプ（オプション）。
            additional_properties: チャットレスポンス更新に関連付けられた追加プロパティ（オプション）。
            raw_representation: チャットレスポンス更新の生の表現（オプション）。
            kwargs: 提供された場合、additional_propertiesと結合されます。


        """
        parsed_contents: list[Contents] = [] if contents is None else _parse_content_list(contents)

        if text is not None:
            if isinstance(text, str):
                text = TextContent(text=text)
            parsed_contents.append(text)

        # 必要に応じてdictからroleを変換します（SerializationMixinのサポートのため）
        if isinstance(role, MutableMapping):
            role = Role.from_dict(role)
        elif isinstance(role, str):
            role = Role(value=role)

        self.contents = parsed_contents
        self.role = role
        self.author_name = author_name
        self.response_id = response_id
        self.message_id = message_id
        self.created_at = created_at
        self.additional_properties = additional_properties
        self.raw_representation: Any | list[Any] | None = raw_representation

    @property
    def text(self) -> str:
        """contents内のすべてのTextContentオブジェクトの連結されたテキストを取得します。"""
        return (
            "".join(content.text for content in self.contents if isinstance(content, TextContent))
            if self.contents
            else ""
        )

    @property
    def user_input_requests(self) -> list[UserInputRequestContents]:
        """レスポンスからすべてのBaseUserInputRequestメッセージを取得します。"""
        return [content for content in self.contents if isinstance(content, UserInputRequestContents)]

    def __str__(self) -> str:
        return self.text


# region ChatOptions


class ToolMode(SerializationMixin, metaclass=EnumLike):
    """チャットリクエストでツールが使用されるかどうか、およびその方法を定義します。

    Examples:
        .. code-block:: python

            from agent_framework import ToolMode

            # 事前定義されたツールモードを使用
            auto_mode = ToolMode.AUTO  # モデルがツール使用のタイミングを決定
            required_mode = ToolMode.REQUIRED_ANY  # モデルはツールを使用しなければならない
            none_mode = ToolMode.NONE  # ツールの使用は禁止

            # 特定の関数を要求
            specific_mode = ToolMode.REQUIRED(function_name="get_weather")
            print(specific_mode.required_function_name)  # "get_weather"

            # モードの比較
            print(auto_mode == "auto")  # True

    """

    # EnumLikeメタクラスの定数設定
    _constants: ClassVar[dict[str, tuple[str, ...]]] = {
        "AUTO": ("auto",),
        "REQUIRED_ANY": ("required",),
        "NONE": ("none",),
    }

    # 定数の型アノテーション
    AUTO: "ToolMode"
    REQUIRED_ANY: "ToolMode"
    NONE: "ToolMode"

    def __init__(
        self,
        mode: Literal["auto", "required", "none"] = "none",
        *,
        required_function_name: str | None = None,
    ) -> None:
        """ToolModeを初期化します。

        引数:
            mode: ツールモード - "auto"、"required"、または"none"。

        キーワード引数:
            required_function_name: requiredモード用の関数名（オプション）。

        """
        self.mode = mode
        self.required_function_name = required_function_name

    @classmethod
    def REQUIRED(cls, function_name: str | None = None) -> "ToolMode":
        """指定された関数の呼び出しを要求するToolModeを返します。"""
        return cls(mode="required", required_function_name=function_name)

    def __eq__(self, other: object) -> bool:
        """別のToolModeまたは文字列との等価性をチェックします。"""
        if isinstance(other, str):
            return self.mode == other
        if isinstance(other, ToolMode):
            return self.mode == other.mode and self.required_function_name == other.required_function_name
        return False

    def __hash__(self) -> int:
        """セットや辞書で使用するためのToolModeのハッシュを返します。"""
        return hash((self.mode, self.required_function_name))

    def serialize_model(self) -> str:
        """ToolModeをモード文字列だけにシリアライズします。"""
        return self.mode

    def __str__(self) -> str:
        """モードの文字列表現を返します。"""
        return self.mode

    def __repr__(self) -> str:
        """ToolModeの文字列表現を返します。"""
        if self.required_function_name:
            return f"ToolMode(mode={self.mode!r}, required_function_name={self.required_function_name!r})"
        return f"ToolMode(mode={self.mode!r})"


class ChatOptions(SerializationMixin):
    """AIサービスの共通リクエスト設定。

    Examples:
        .. code-block:: python

            from agent_framework import ChatOptions, ai_function

            # 基本的なチャットオプションを作成
            options = ChatOptions(
                model_id="gpt-4",
                temperature=0.7,
                max_tokens=1000,
            )


            # ツールを使う場合
            @ai_function
            def get_weather(location: str) -> str:
                '''場所の天気を取得します。'''
                return f"Weather in {location}"


            options = ChatOptions(
                model_id="gpt-4",
                tools=get_weather,
                tool_choice="auto",
            )

            # 特定のツールの呼び出しを要求
            options_required = ChatOptions(
                model_id="gpt-4",
                tools=get_weather,
                tool_choice=ToolMode.REQUIRED(function_name="get_weather"),
            )

            # オプションの組み合わせ
            base_options = ChatOptions(temperature=0.5)
            extended_options = ChatOptions(max_tokens=500, tools=get_weather)
            combined = base_options & extended_options

    """

    DEFAULT_EXCLUDE: ClassVar[set[str]] = {"_tools"}  # 内部フィールド、.toolsプロパティを使用してください

    def __init__(
        self,
        *,
        model_id: str | None = None,
        allow_multiple_tool_calls: bool | None = None,
        conversation_id: str | None = None,
        frequency_penalty: float | None = None,
        instructions: str | None = None,
        logit_bias: MutableMapping[str | int, float] | None = None,
        max_tokens: int | None = None,
        metadata: MutableMapping[str, str] | None = None,
        presence_penalty: float | None = None,
        response_format: type[BaseModel] | None = None,
        seed: int | None = None,
        stop: str | Sequence[str] | None = None,
        store: bool | None = None,
        temperature: float | None = None,
        tool_choice: ToolMode | Literal["auto", "required", "none"] | Mapping[str, Any] | None = None,
        tools: ToolProtocol
        | Callable[..., Any]
        | MutableMapping[str, Any]
        | Sequence[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]]
        | None = None,
        top_p: float | None = None,
        user: str | None = None,
        additional_properties: MutableMapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        """ChatOptionsを初期化します。

        キーワード引数:
            model_id: 使用するAIモデルID。
            allow_multiple_tool_calls: 複数のツール呼び出しを許可するかどうか。
            conversation_id: 会話ID。
            frequency_penalty: 頻度ペナルティ（-2.0から2.0の範囲）。
            instructions: 指示。システムまたは同等のメッセージに変換されます。
            logit_bias: ロジットバイアスマッピング。
            max_tokens: 最大トークン数（0より大きい必要があります）。
            metadata: メタデータマッピング。
            presence_penalty: プレゼンスペナルティ（-2.0から2.0の範囲）。
            response_format: 構造化出力レスポンスフォーマットスキーマ。有効なPydanticモデルである必要があります。
            seed: 再現性のためのランダムシード。
            stop: ストップシーケンス。
            store: 会話を保存するかどうか。
            temperature: 温度（0.0から2.0の範囲）。
            tool_choice: ツール選択モード。
            tools: 利用可能なツールのリスト。
            top_p: top-p値（0.0から1.0の範囲）。
            user: ユーザーID。
            additional_properties: プロバイダー固有の追加プロパティ。kwargsとしても渡せます。
            **kwargs: additional_propertiesに含める追加プロパティ。

        """
        # 数値制約を検証し、必要に応じて型を変換します
        if frequency_penalty is not None:
            if not (-2.0 <= frequency_penalty <= 2.0):
                raise ValueError("frequency_penalty must be between -2.0 and 2.0")
            frequency_penalty = float(frequency_penalty)
        if presence_penalty is not None:
            if not (-2.0 <= presence_penalty <= 2.0):
                raise ValueError("presence_penalty must be between -2.0 and 2.0")
            presence_penalty = float(presence_penalty)
        if temperature is not None:
            if not (0.0 <= temperature <= 2.0):
                raise ValueError("temperature must be between 0.0 and 2.0")
            temperature = float(temperature)
        if top_p is not None:
            if not (0.0 <= top_p <= 1.0):
                raise ValueError("top_p must be between 0.0 and 1.0")
            top_p = float(top_p)
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        if additional_properties is None:
            additional_properties = {}
        if kwargs:
            additional_properties.update(kwargs)

        self.additional_properties = cast(dict[str, Any], additional_properties)
        self.model_id = model_id
        self.allow_multiple_tool_calls = allow_multiple_tool_calls
        self.conversation_id = conversation_id
        self.frequency_penalty = frequency_penalty
        self.instructions = instructions
        self.logit_bias = logit_bias
        self.max_tokens = max_tokens
        self.metadata = metadata
        self.presence_penalty = presence_penalty
        self.response_format = response_format
        self.seed = seed
        self.stop = stop
        self.store = store
        self.temperature = temperature
        self.tool_choice = self._validate_tool_mode(tool_choice)
        self._tools = self._validate_tools(tools)
        self.top_p = top_p
        self.user = user

    @property
    def tools(self) -> list[ToolProtocol | MutableMapping[str, Any]] | None:
        """指定されたツールを返します。"""
        return self._tools

    @tools.setter
    def tools(
        self,
        new_tools: ToolProtocol
        | Callable[..., Any]
        | MutableMapping[str, Any]
        | Sequence[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]]
        | None,
    ) -> None:
        """ツールを設定します。"""
        self._tools = self._validate_tools(new_tools)

    @classmethod
    def _validate_tools(
        cls,
        tools: (
            ToolProtocol
            | Callable[..., Any]
            | MutableMapping[str, Any]
            | Sequence[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]]
            | None
        ),
    ) -> list[ToolProtocol | MutableMapping[str, Any]] | None:
        """toolsフィールドを解析します。"""
        if not tools:
            return None
        if not isinstance(tools, Sequence):
            if not isinstance(tools, (ToolProtocol, MutableMapping)):
                return [ai_function(tools)]
            return [tools]
        return [tool if isinstance(tool, (ToolProtocol, MutableMapping)) else ai_function(tool) for tool in tools]

    @classmethod
    def _validate_tool_mode(
        cls, tool_choice: ToolMode | Literal["auto", "required", "none"] | Mapping[str, Any] | None
    ) -> ToolMode | str | None:
        """tool_choiceフィールドが有効なToolModeであることを検証します。"""
        if not tool_choice:
            return None
        if isinstance(tool_choice, str):
            match tool_choice:
                case "auto":
                    return ToolMode.AUTO
                case "required":
                    return ToolMode.REQUIRED_ANY
                case "none":
                    return ToolMode.NONE
                case _:
                    raise ContentError(f"Invalid tool choice: {tool_choice}")
        if isinstance(tool_choice, (dict, Mapping)):
            return ToolMode.from_dict(tool_choice)  # type: ignore
        return tool_choice

    def __and__(self, other: object) -> "ChatOptions":
        """2つのChatOptionsインスタンスを結合します。

        他方のChatOptionsの値が優先されます。
        リストや辞書は結合されます。

        """
        if not isinstance(other, ChatOptions):
            return self
        other_tools = other.tools
        # tool_choiceは特殊なシリアライズメソッドを持ちます。後で修正できるようここに保存します。
        tool_choice = other.tool_choice or self.tool_choice
        # response_formatはシリアライズできないクラス型です。後で復元できるようここに保存します。
        response_format = self.response_format
        # ツールオブジェクトを保持したままselfの浅いコピーから開始します。
        combined = ChatOptions.from_dict(self.to_dict())
        combined.tool_choice = self.tool_choice
        combined.tools = list(self.tools) if self.tools else None
        combined.logit_bias = dict(self.logit_bias) if self.logit_bias else None
        combined.metadata = dict(self.metadata) if self.metadata else None
        combined.response_format = response_format

        # 他のオプションからスカラーおよびマッピングの更新を適用します。
        updated_data = other.to_dict(exclude_none=True, exclude={"tools"})
        logit_bias = updated_data.pop("logit_bias", {})
        metadata = updated_data.pop("metadata", {})
        additional_properties: dict[str, Any] = updated_data.pop("additional_properties", {})

        for key, value in updated_data.items():
            setattr(combined, key, value)

        combined.tool_choice = tool_choice
        # 他方にresponse_formatがあればそれを保持し、なければselfのものを保持します。
        if other.response_format is not None:
            combined.response_format = other.response_format
        if other.instructions:
            combined.instructions = "\n".join([combined.instructions or "", other.instructions or ""])

        combined.logit_bias = (
            {**(combined.logit_bias or {}), **logit_bias} if logit_bias or combined.logit_bias else None
        )
        combined.metadata = {**(combined.metadata or {}), **metadata} if metadata or combined.metadata else None
        if combined.additional_properties and additional_properties:
            combined.additional_properties.update(additional_properties)
        else:
            if additional_properties:
                combined.additional_properties = additional_properties
        if other_tools:
            if combined.tools is None:
                combined.tools = list(other_tools)
            else:
                for tool in other_tools:
                    if tool not in combined.tools:
                        combined.tools.append(tool)
        return combined
