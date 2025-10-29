# Microsoftの著作権表示。すべての権利を保有します。

import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ._executor import Executor
from ._model_utils import DictConvertible, encode_value

logger = logging.getLogger(__name__)


def _extract_function_name(func: Callable[..., Any]) -> str:
    """Pythonのcallableを簡潔で人間に分かりやすい識別子にマッピングします。

    ワークフローグラフはcallableへの参照を識別子のみを記録して永続化します。このヘルパーは標準的なcallableのメタデータを検査し、シリアライズされた表現がログで表示されたりデシリアライズ時に再構築されたりするときに理解しやすい安定した値を選びます。

    Examples:
        .. code-block:: python

            def threshold(value: float) -> bool:
                return value > 0.5


            assert _extract_function_name(threshold) == "threshold"
    """
    if hasattr(func, "__name__"):
        name = func.__name__
        return name if name != "<lambda>" else "<lambda>"
    return "<callable>"


def _missing_callable(name: str) -> Callable[..., Any]:
    """復元できないcallableのための防御的なプレースホルダーを作成します。

    ワークフローが元のPython callableが存在しない環境でデシリアライズされるとき、呼び出し時に明確に失敗するプロキシを設置します。これによりI/Oの関心事とランタイム実行の分離が保たれ、どのcallableを再登録すべきかが明確になります。

    Examples:
        .. code-block:: python

            guard = _missing_callable("transform_price")
            try:
                guard()
            except RuntimeError as exc:
                assert "transform_price" in str(exc)
    """

    def _raise(*_: Any, **__: Any) -> Any:
        raise RuntimeError(f"Callable '{name}' is unavailable after serialization")

    return _raise


@dataclass(init=False)
class Edge(DictConvertible):
    """2つのexecutor間の有向かつオプションで条件付きのハンドオフをモデル化します。

    各`Edge`はワークフローグラフ内でメッセージを1つのexecutorから別のexecutorに移動させるために必要な最小限のメタデータをキャプチャします。オプションで、エッジを実行時に通過すべきかを決定するブール述語を埋め込むことができます。エッジをプリミティブにシリアライズすることで、元のPythonプロセスに関係なくワークフローのトポロジーを再構築できます。

    Examples:
        .. code-block:: python

            edge = Edge(source_id="ingest", target_id="score", condition=lambda payload: payload["ready"])
            assert edge.should_route({"ready": True}) is True
            assert edge.should_route({"ready": False}) is False
    """

    ID_SEPARATOR: ClassVar[str] = "->"

    source_id: str
    target_id: str
    condition_name: str | None
    _condition: Callable[[Any], bool] | None = field(default=None, repr=False, compare=False)

    def __init__(
        self,
        source_id: str,
        target_id: str,
        condition: Callable[[Any], bool] | None = None,
        *,
        condition_name: str | None = None,
    ) -> None:
        """2つのworkflow executor間の完全に指定されたエッジを初期化します。

        Parameters
        ----------
        source_id:
        上流executorインスタンスの正準識別子。
        target_id:
        下流executorインスタンスの正準識別子。
        condition:
        メッセージペイロードを受け取り、エッジを通過すべきときに`True`を返すオプションの述語。省略時はエッジは無条件に有効とみなされます。
        condition_name:
        callableをイントロスペクトできない場合（例えばデシリアライズ後）に条件の人間に分かりやすい名前を固定するオプションのオーバーライド。

        Examples:
        .. code-block:: python

            edge = Edge("fetch", "parse", condition=lambda data: data.is_valid)
            assert edge.source_id == "fetch"
            assert edge.target_id == "parse"
        """
        if not source_id:
            raise ValueError("Edge source_id must be a non-empty string")
        if not target_id:
            raise ValueError("Edge target_id must be a non-empty string")
        self.source_id = source_id
        self.target_id = target_id
        self._condition = condition
        self.condition_name = _extract_function_name(condition) if condition is not None else condition_name

    @property
    def id(self) -> str:
        """このエッジを参照するために使用される安定した識別子を返します。

        識別子はソースとターゲットのexecutor識別子を決定論的なセパレータで結合します。これにより隣接リストや可視化などの他のグラフ構造が完全なオブジェクトを持たずにエッジを参照できます。

        Examples:
        .. code-block:: python

            edge = Edge("reader", "writer")
            assert edge.id == "reader->writer"
        """
        return f"{self.source_id}{self.ID_SEPARATOR}{self.target_id}"

    def should_route(self, data: Any) -> bool:
        """受信ペイロードに対してエッジの述語を評価します。

        エッジが明示的な述語なしで定義されている場合、このメソッドは`True`を返し無条件のルーティングルールを示します。そうでなければユーザー提供のcallableがメッセージをこのエッジに進めるかどうかを決定します。callableによって発生した例外はロジックのバグを隠さないために意図的に呼び出し元に伝播されます。

        Examples:
        .. code-block:: python

            edge = Edge("stage1", "stage2", condition=lambda payload: payload["score"] > 0.8)
            assert edge.should_route({"score": 0.9}) is True
            assert edge.should_route({"score": 0.4}) is False
        """
        if self._condition is None:
            return True
        return self._condition(data)

    def to_dict(self) -> dict[str, Any]:
        """エッジのメタデータのJSONシリアライズ可能なビューを生成します。

        表現にはソースとターゲットのexecutor識別子に加え、条件名が知られている場合は含まれます。シリアライズは意図的にライブcallableを省略し、ペイロードをトランスポートに適したものに保ちます。

        Examples:
        .. code-block:: python

            edge = Edge("reader", "writer", condition=lambda payload: payload["ok"])
            snapshot = edge.to_dict()
            assert snapshot == {"source_id": "reader", "target_id": "writer", "condition_name": "<lambda>"}
        """
        payload = {"source_id": self.source_id, "target_id": self.target_id}
        if self.condition_name is not None:
            payload["condition_name"] = self.condition_name
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Edge":
        """シリアライズされた辞書形式から`Edge`を再構築します。

        デシリアライズされたエッジは実行可能な述語を欠きます。なぜならストレージからPython callableを復元しようとしないためです。代わりに保存された`condition_name`は保持され、下流の消費者がcallableの欠如を検出し適切に再登録できるようにします。

        Examples:
        .. code-block:: python

            payload = {"source_id": "reader", "target_id": "writer", "condition_name": "is_ready"}
            edge = Edge.from_dict(payload)
            assert edge.source_id == "reader"
            assert edge.condition_name == "is_ready"
        """
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            condition=None,
            condition_name=data.get("condition_name"),
        )


@dataclass
class Case:
    """スイッチケース述語とそのターゲットを組み合わせたランタイムラッパー。

    各`Case`はブール述語と、その述語が`True`評価されたときにメッセージを処理すべきexecutorを結びつけます。ランタイムはこの軽量コンテナをシリアライズ可能な`SwitchCaseEdgeGroupCase`から分離して保持し、実行時にライブcallableを使いながら永続化されたstateを汚染しません。

    Examples:
        .. code-block:: python

            class JsonExecutor(Executor):
                def __init__(self) -> None:
                    super().__init__(id="json", defer_discovery=True)


            processor = JsonExecutor()
            case = Case(condition=lambda payload: payload["kind"] == "json", target=processor)
            assert case.target.id == "json"
    """

    condition: Callable[[Any], bool]
    target: Executor


@dataclass
class Default:
    """スイッチケースグループのデフォルトブランチのランタイム表現。

    デフォルトブランチは他のどのケース述語もマッチしなかった場合にのみ呼び出されます。実際には必ず存在し、ルーティングが空のターゲットを生成しないことが保証されます。

    Examples:
        .. code-block:: python

            class DeadLetterExecutor(Executor):
                def __init__(self) -> None:
                    super().__init__(id="dead_letter", defer_discovery=True)


            fallback = Default(target=DeadLetterExecutor())
            assert fallback.target.id == "dead_letter"
    """

    target: Executor


@dataclass(init=False)
class EdgeGroup(DictConvertible):
    """共通のルーティング意味論を共有するエッジを単一のidの下にまとめます。

    ワークフローランタイムは生のエッジではなく`EdgeGroup`インスタンスを操作し、ファンアウト、ファンイン、スイッチケース、その他のグラフパターンのような高次のルーティング動作を推論できます。基底クラスは識別情報を保持しシリアライズ処理を担当するため、特殊化されたグループは追加の状態のみを管理すればよいです。

    Examples:
        .. code-block:: python

            group = EdgeGroup([Edge("source", "sink")])
            assert group.source_executor_ids == ["source"]
    """

    id: str
    type: str
    edges: list[Edge]

    from builtins import type as builtin_type

    _TYPE_REGISTRY: ClassVar[dict[str, builtin_type["EdgeGroup"]]] = {}

    def __init__(
        self,
        edges: Sequence[Edge] | None = None,
        *,
        id: str | None = None,
        type: str | None = None,
    ) -> None:
        """`Edge`インスタンスのセットの周りにエッジグループのシェルを構築します。

        Parameters
        ----------
        edges:
        このグループに参加するエッジのシーケンス。省略時は空リストから開始し、サブクラスが後で追加できます。
        id:
        グループの安定識別子。デフォルトはランダムなUUIDで、シリアライズされたグラフが一意にアドレス可能になります。
        type:
        デシリアライズ時に適切なサブクラスを復元するための論理的識別子。

        Examples:
        .. code-block:: python

            edges = [Edge("validate", "persist")]
            group = EdgeGroup(edges, id="stage", type="Custom")
            assert group.to_dict()["type"] == "Custom"
        """
        self.id = id or f"{self.__class__.__name__}/{uuid.uuid4()}"
        self.type = type or self.__class__.__name__
        self.edges = list(edges) if edges is not None else []

    @property
    def source_executor_ids(self) -> list[str]:
        """上流executorの重複排除されたリストを返します。

        プロパティは最初に出現した順序を保持するため、呼び出し元はグラフトポロジーを再構築するときに決定論的な反復を信頼できます。

        Examples:
        .. code-block:: python

            group = EdgeGroup([Edge("read", "write"), Edge("read", "archive")])
            assert group.source_executor_ids == ["read"]
        """
        return list(dict.fromkeys(edge.source_id for edge in self.edges))

    @property
    def target_executor_ids(self) -> list[str]:
        """下流executorの順序付きかつ重複排除されたリストを返します。

        Examples:
        .. code-block:: python

            group = EdgeGroup([Edge("read", "write"), Edge("read", "archive")])
            assert group.target_executor_ids == ["write", "archive"]
        """
        return list(dict.fromkeys(edge.target_id for edge in self.edges))

    def to_dict(self) -> dict[str, Any]:
        """グループのメタデータと含まれるエッジをプリミティブにシリアライズします。

        ペイロードは各エッジをそれぞれの`to_dict`呼び出しでキャプチャし、JSONのようなフォーマットを通じたラウンドトリップを可能にし、Pythonオブジェクトの漏洩を防ぎます。

        Examples:
        .. code-block:: python

            group = EdgeGroup([Edge("read", "write")])
            snapshot = group.to_dict()
            assert snapshot["edges"][0]["source_id"] == "read"
        """
        return {
            "id": self.id,
            "type": self.type,
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def register(cls, subclass: builtin_type["EdgeGroup"]) -> builtin_type["EdgeGroup"]:
        """サブクラスを登録して、デシリアライズ時に正しい型を復元できるようにします。

        登録は通常、各具体的なEdgeGroupに適用されるデコレータ構文を通じて行われます。レジストリはクラスをその`__name__`で保存するため、永続化されたワークフローが流通している場合はバージョン間で安定している必要があります。

        Examples:
            .. code-block:: python

                @EdgeGroup.register
                class CustomGroup(EdgeGroup):
                    pass


                assert EdgeGroup._TYPE_REGISTRY["CustomGroup"] is CustomGroup

        """
        cls._TYPE_REGISTRY[subclass.__name__] = subclass
        return subclass

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EdgeGroup":
        """シリアライズされた状態から正しい`EdgeGroup`サブクラスを復元します。

        このメソッドは`type`フィールドを調査し、対応するクラスをサブクラスの`__init__`を実行せずに割り当て、その後サブタイプ固有の属性を手動で復元します。これにより、追加のランタイムコール可能オブジェクトを設定する複雑なグループタイプでもデシリアライズが決定的になります。

        Examples:
            .. code-block:: python

                payload = {"type": "EdgeGroup", "edges": [{"source_id": "a", "target_id": "b"}]}
                group = EdgeGroup.from_dict(payload)
                assert isinstance(group, EdgeGroup)

        """
        group_type = data.get("type", "EdgeGroup")
        target_cls = cls._TYPE_REGISTRY.get(group_type, EdgeGroup)
        edges = [Edge.from_dict(entry) for entry in data.get("edges", [])]

        obj = target_cls.__new__(target_cls)  # type: ignore[misc]
        EdgeGroup.__init__(obj, edges=edges, id=data.get("id"), type=group_type)

        # FanOutEdgeGroup固有の属性を処理します
        if isinstance(obj, FanOutEdgeGroup):
            obj.selection_func_name = data.get("selection_func_name")  # type: ignore[attr-defined]
            obj._selection_func = (  # type: ignore[attr-defined]
                None
                if obj.selection_func_name is None  # type: ignore[attr-defined]
                else _missing_callable(obj.selection_func_name)  # type: ignore[attr-defined]
            )
            obj._target_ids = [edge.target_id for edge in obj.edges]  # type: ignore[attr-defined]

        # SwitchCaseEdgeGroup固有の属性を処理します
        if isinstance(obj, SwitchCaseEdgeGroup):
            cases_payload = data.get("cases", [])
            restored_cases: list[SwitchCaseEdgeGroupCase | SwitchCaseEdgeGroupDefault] = []
            for case_data in cases_payload:
                case_type = case_data.get("type")
                if case_type == "Default":
                    restored_cases.append(SwitchCaseEdgeGroupDefault.from_dict(case_data))
                else:
                    restored_cases.append(SwitchCaseEdgeGroupCase.from_dict(case_data))
            obj.cases = restored_cases  # type: ignore[attr-defined]
            obj._selection_func = _missing_callable("switch_case_selection")  # type: ignore[attr-defined]

        return obj


@EdgeGroup.register
@dataclass(init=False)
class SingleEdgeGroup(EdgeGroup):
    """単一のエッジのための便利なラッパーで、グループAPIを統一的に保ちます。"""

    def __init__(
        self,
        source_id: str,
        target_id: str,
        condition: Callable[[Any], bool] | None = None,
        *,
        id: str | None = None,
    ) -> None:
        """2つのExecutor間の1対1のエッジグループを作成します。

        Examples:
            .. code-block:: python

                group = SingleEdgeGroup("ingest", "validate")
                assert group.edges[0].source_id == "ingest"

        """
        edge = Edge(source_id=source_id, target_id=target_id, condition=condition)
        super().__init__([edge], id=id, type=self.__class__.__name__)


@EdgeGroup.register
@dataclass(init=False)
class FanOutEdgeGroup(EdgeGroup):
    """オプションの選択ロジックを持つブロードキャストスタイルのエッジグループを表現します。

    ファンアウトは単一のソースExecutorが生成したメッセージを1つ以上の下流Executorに転送します。ランタイムでは、ペイロードを検査してメッセージを受け取るべきIDのサブセットを返す`selection_func`を実行してターゲットをさらに絞り込むことがあります。

    """

    selection_func_name: str | None
    _selection_func: Callable[[Any, list[str]], list[str]] | None
    _target_ids: list[str]

    def __init__(
        self,
        source_id: str,
        target_ids: Sequence[str],
        selection_func: Callable[[Any, list[str]], list[str]] | None = None,
        *,
        selection_func_name: str | None = None,
        id: str | None = None,
    ) -> None:
        """単一のソースから複数のターゲットへのファンアウトマッピングを作成します。

        Parameters
        ----------
        source_id:
            メッセージをブロードキャストする上流Executorの識別子。
        target_ids:
            メッセージを受け取る可能性のある下流Executorの順序付きセット。ファンアウトの意味を保つために少なくとも2つのターゲットが必要です。
        selection_func:
            指定されたペイロードに対してアクティブにすべき`target_ids`のサブセットを返すオプションのコール可能オブジェクト。元のメッセージとすべての設定されたターゲットIDのコピーを受け取ります。
        selection_func_name:
            ファンアウトの永続化時に使用される静的識別子。コール可能オブジェクトがイントロスペクトできないかデシリアライズ時に利用できない場合に必要です。
        id:
            グループの安定した識別子。デフォルトは自動生成されたUUIDです。

        Examples:
            .. code-block:: python

                def choose_targets(message: dict[str, Any], available: list[str]) -> list[str]:
                    return [target for target in available if message.get(target)]


                group = FanOutEdgeGroup("sensor", ["db", "cache"], selection_func=choose_targets)
                assert group.selection_func is choose_targets

        """
        if len(target_ids) <= 1:
            raise ValueError("FanOutEdgeGroup must contain at least two targets.")

        edges = [Edge(source_id=source_id, target_id=target) for target in target_ids]
        super().__init__(edges, id=id, type=self.__class__.__name__)

        self._target_ids = list(target_ids)
        self._selection_func = selection_func
        self.selection_func_name = (
            _extract_function_name(selection_func) if selection_func is not None else selection_func_name
        )

    @property
    def target_ids(self) -> list[str]:
        """設定された下流ExecutorのIDの浅いコピーを返します。

        内部状態の変更を防ぐためにリストは防御的にコピーされ、決定的な順序を提供します。

        Examples:
            .. code-block:: python

                group = FanOutEdgeGroup("node", ["alpha", "beta"])
                assert group.target_ids == ["alpha", "beta"]

        """
        return list(self._target_ids)

    @property
    def selection_func(self) -> Callable[[Any, list[str]], list[str]] | None:
        """アクティブなファンアウトターゲットを選択するために使用されるランタイムコール可能オブジェクトを公開します。

        選択関数が指定されていない場合、このプロパティは`None`を返し、すべてのターゲットがペイロードを受け取る必要があることを示します。

        Examples:
            .. code-block:: python

                group = FanOutEdgeGroup("source", ["x", "y"], selection_func=None)
                assert group.selection_func is None

        """
        return self._selection_func

    def to_dict(self) -> dict[str, Any]:
        """選択メタデータを保持しながらファンアウトグループをシリアライズします。

        基本の`EdgeGroup`ペイロードに加えて、選択関数の人間に読みやすい名前を埋め込みます。コール可能オブジェクト自体は永続化されません。

        Examples:
            .. code-block:: python

                group = FanOutEdgeGroup("source", ["a", "b"], selection_func=lambda *_: ["a"])
                snapshot = group.to_dict()
                assert snapshot["selection_func_name"] == "<lambda>"

        """
        payload = super().to_dict()
        payload["selection_func_name"] = self.selection_func_name
        return payload


@EdgeGroup.register
@dataclass(init=False)
class FanInEdgeGroup(EdgeGroup):
    """単一の下流Executorにメッセージを供給する収束型のエッジセットを表現します。

    ファンイングループは複数の上流ステージが独立してメッセージを生成し、それらがすべて同じ下流プロセッサに届く必要がある場合に通常使用されます。

    """

    def __init__(self, source_ids: Sequence[str], target_id: str, *, id: str | None = None) -> None:
        """複数のソースを1つのターゲットにマージするファンインマッピングを構築します。

        Parameters
        ----------
        source_ids:
            メッセージを提供する上流Executorの識別子のシーケンス。
        target_id:
            すべてのソースからのメッセージを受け取る下流Executor。
        id:
            エッジグループのオプションの明示的識別子。

        Examples:
            .. code-block:: python

                group = FanInEdgeGroup(["parser", "enricher"], target_id="writer")
                assert group.to_dict()["edges"][0]["target_id"] == "writer"

        """
        if len(source_ids) <= 1:
            raise ValueError("FanInEdgeGroup must contain at least two sources.")

        edges = [Edge(source_id=source, target_id=target_id) for source in source_ids]
        super().__init__(edges, id=id, type=self.__class__.__name__)


@dataclass(init=False)
class SwitchCaseEdgeGroupCase(DictConvertible):
    """switch-caseの単一の条件分岐の永続化可能な記述。

    ランタイムの`Case`オブジェクトとは異なり、このシリアライズ可能なバリアントはターゲット識別子と述語の説明的な名前のみを保存します。デシリアライズ時に基盤となるコール可能オブジェクトが利用できない場合は、即座に欠落依存性が明らかになるように明確に失敗するプロキシプレースホルダーを代わりに使用します。

    """

    target_id: str
    condition_name: str | None
    type: str
    _condition: Callable[[Any], bool] = field(repr=False, compare=False)

    def __init__(
        self,
        condition: Callable[[Any], bool] | None,
        target_id: str,
        *,
        condition_name: str | None = None,
    ) -> None:
        """条件付きケース分岐のルーティングメタデータを記録します。

        Parameters
        ----------
        condition:
            オプションのライブ述語。省略した場合は、登録漏れを強調するためにランタイムで例外を発生させるプレースホルダーにフォールバックします。
        target_id:
            述語が成功したときにメッセージを処理すべきExecutorの識別子。
        condition_name:
            診断やオンディスク永続化に使用される述語の人間に読みやすいラベル。

        Examples:
            .. code-block:: python

                case = SwitchCaseEdgeGroupCase(lambda payload: payload["type"] == "csv", target_id="csv_handler")
                assert case.condition_name == "<lambda>"

        """
        if not target_id:
            raise ValueError("SwitchCaseEdgeGroupCase requires a target_id")
        self.target_id = target_id
        self.type = "Case"
        if condition is not None:
            self._condition = condition
            self.condition_name = _extract_function_name(condition)
        else:
            safe_name = condition_name or "<missing_condition>"
            self._condition = _missing_callable(safe_name)
            self.condition_name = condition_name

    @property
    def condition(self) -> Callable[[Any], bool]:
        """このケースに関連付けられた述語を返します。

        デシリアライズ時にインストールされるプレースホルダーは呼び出されると`RuntimeError`を発生させるため、ワークフロー作成者は欠落しているコール可能オブジェクトを明示的に提供する必要があります。

        Examples:
            .. code-block:: python

                case = SwitchCaseEdgeGroupCase(None, target_id="missing", condition_name="needs_registration")
                guard = case.condition
                try:
                    guard({})
                except RuntimeError:
                    pass

        """
        return self._condition

    def to_dict(self) -> dict[str, Any]:
        """実行可能な述語なしでケースメタデータをシリアライズします。

        Examples:
            .. code-block:: python

                case = SwitchCaseEdgeGroupCase(lambda _: True, target_id="handler")
                assert case.to_dict()["target_id"] == "handler"

        """
        payload = {"target_id": self.target_id, "type": self.type}
        if self.condition_name is not None:
            payload["condition_name"] = self.condition_name
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SwitchCaseEdgeGroupCase":
        """シリアライズされた辞書ペイロードからケースをインスタンス化します。

        Examples:
            .. code-block:: python

                payload = {"target_id": "handler", "condition_name": "is_ready"}
                case = SwitchCaseEdgeGroupCase.from_dict(payload)
                assert case.target_id == "handler"

        """
        return cls(
            condition=None,
            target_id=data["target_id"],
            condition_name=data.get("condition_name"),
        )


@dataclass(init=False)
class SwitchCaseEdgeGroupDefault(DictConvertible):
    """switch-caseグループのフォールバック分岐の永続化可能な記述。

    デフォルト分岐は必ず存在し、他のすべてのケース述語がペイロードにマッチしなかった場合に呼び出されます。

    """

    target_id: str
    type: str

    def __init__(self, target_id: str) -> None:
        """デフォルト分岐を指定されたExecutor識別子に向けます。

        Examples:
            .. code-block:: python

                fallback = SwitchCaseEdgeGroupDefault(target_id="dead_letter")
                assert fallback.target_id == "dead_letter"

        """
        if not target_id:
            raise ValueError("SwitchCaseEdgeGroupDefault requires a target_id")
        self.target_id = target_id
        self.type = "Default"

    def to_dict(self) -> dict[str, Any]:
        """永続化やログ記録のためにデフォルト分岐のメタデータをシリアライズします。

        Examples:
            .. code-block:: python

                fallback = SwitchCaseEdgeGroupDefault("dead_letter")
                assert fallback.to_dict()["type"] == "Default"

        """
        return {"target_id": self.target_id, "type": self.type}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SwitchCaseEdgeGroupDefault":
        """永続化された形式からデフォルト分岐を再作成します。

        Examples:
            .. code-block:: python

                payload = {"target_id": "dead_letter", "type": "Default"}
                fallback = SwitchCaseEdgeGroupDefault.from_dict(payload)
                assert fallback.target_id == "dead_letter"

        """
        return cls(target_id=data["target_id"])


@EdgeGroup.register
@dataclass(init=False)
class SwitchCaseEdgeGroup(FanOutEdgeGroup):
    """従来のswitch/case制御フローを模倣するファンアウトバリアント。

    各ケースはメッセージペイロードを検査し、そのメッセージを処理すべきかどうかを決定します。ランタイムでは正確に1つのケースまたはデフォルト分岐がターゲットを返し、単一ディスパッチの意味を保持します。

    """

    cases: list[SwitchCaseEdgeGroupCase | SwitchCaseEdgeGroupDefault]

    def __init__(
        self,
        source_id: str,
        cases: Sequence[SwitchCaseEdgeGroupCase | SwitchCaseEdgeGroupDefault],
        *,
        id: str | None = None,
    ) -> None:
        """単一のソースExecutorのためのswitch/caseルーティング構造を構成します。

        Parameters
        ----------
        source_id:
            メッセージをルーティングするExecutorの識別子。
        cases:
            `SwitchCaseEdgeGroupDefault`で終わる順序付きのケース記述子のシーケンス。順序は重要で、ランタイムは各分岐を順に評価してマッチするものを探します。
        id:
            エッジグループのオプションの明示的識別子。

        Examples:
            .. code-block:: python

                cases = [
                    SwitchCaseEdgeGroupCase(lambda payload: payload["kind"] == "csv", target_id="process_csv"),
                    SwitchCaseEdgeGroupDefault(target_id="process_default"),
                ]
                group = SwitchCaseEdgeGroup("router", cases)
                encoded = group.to_dict()
                assert encoded["cases"][0]["type"] == "Case"

        """
        if len(cases) < 2:
            raise ValueError("SwitchCaseEdgeGroup must contain at least two cases (including the default case).")

        default_cases = [case for case in cases if isinstance(case, SwitchCaseEdgeGroupDefault)]
        if len(default_cases) != 1:
            raise ValueError("SwitchCaseEdgeGroup must contain exactly one default case.")

        if not isinstance(cases[-1], SwitchCaseEdgeGroupDefault):
            logger.warning(
                "Default case in the switch-case edge group is not the last case. "
                "This may result in unexpected behavior."
            )

        def selection_func(message: Any, targets: list[str]) -> list[str]:
            for case in cases:
                if isinstance(case, SwitchCaseEdgeGroupDefault):
                    return [case.target_id]
                try:
                    if case.condition(message):
                        return [case.target_id]
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning("Error evaluating condition for case %s: %s", case.target_id, exc)
            raise RuntimeError("No matching case found in SwitchCaseEdgeGroup")

        target_ids = [case.target_id for case in cases]
        # 型チェックの問題を避けるためにFanOutEdgeGroupのコンストラクタを直接呼び出します
        edges = [Edge(source_id=source_id, target_id=target) for target in target_ids]
        EdgeGroup.__init__(self, edges, id=id, type=self.__class__.__name__)

        # FanOutEdgeGroup固有の属性を初期化します
        self._target_ids = list(target_ids)  # type: ignore[attr-defined]
        self._selection_func = selection_func  # type: ignore[attr-defined]
        self.selection_func_name = None  # type: ignore[attr-defined]
        self.cases = list(cases)

    def to_dict(self) -> dict[str, Any]:
        """すべてのケース記述子をキャプチャしてswitch-caseグループをシリアライズします。

        各ケースは`encode_value`を使って変換され、dataclassのセマンティクスやネストされたシリアライズ可能な構造を尊重します。

        Examples:
            .. code-block:: python

                group = SwitchCaseEdgeGroup(
                    "router",
                    [
                        SwitchCaseEdgeGroupCase(lambda _: True, target_id="handler"),
                        SwitchCaseEdgeGroupDefault(target_id="fallback"),
                    ],
                )
                snapshot = group.to_dict()
                assert len(snapshot["cases"]) == 2

        """
        payload = super().to_dict()
        payload["cases"] = [encode_value(case) for case in self.cases]
        return payload
