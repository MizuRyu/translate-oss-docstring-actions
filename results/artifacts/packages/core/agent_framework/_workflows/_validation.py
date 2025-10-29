# Copyright (c) Microsoft. All rights reserved.

import logging
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum
from typing import Any

from ._edge import Edge, EdgeGroup, FanInEdgeGroup
from ._executor import Executor
from ._request_info_executor import RequestInfoExecutor
from ._typing_utils import is_type_compatible

logger = logging.getLogger(__name__)

# ワークフローで意図的なフィードバックループが同じプロセス内で複数回構築される場合に、 ログのスパムを避けるために既に報告したサイクルシグネチャを追跡する。
_LOGGED_CYCLE_SIGNATURES: set[tuple[str, ...]] = set()


# region Enums and Base Classes
class ValidationTypeEnum(Enum):
    """ワークフロー検証タイプの列挙。"""

    EDGE_DUPLICATION = "EDGE_DUPLICATION"
    EXECUTOR_DUPLICATION = "EXECUTOR_DUPLICATION"
    TYPE_COMPATIBILITY = "TYPE_COMPATIBILITY"
    GRAPH_CONNECTIVITY = "GRAPH_CONNECTIVITY"
    HANDLER_OUTPUT_ANNOTATION = "HANDLER_OUTPUT_ANNOTATION"
    INTERCEPTOR_CONFLICT = "INTERCEPTOR_CONFLICT"


class WorkflowValidationError(Exception):
    """ワークフロー検証エラーの基本例外。"""

    def __init__(self, message: str, validation_type: ValidationTypeEnum):
        super().__init__(message)
        self.message = message
        self.validation_type = validation_type

    def __str__(self) -> str:
        return f"[{self.validation_type.value}] {self.message}"


class EdgeDuplicationError(WorkflowValidationError):
    """ワークフローで重複したエッジが検出された場合に発生する例外。"""

    def __init__(self, edge_id: str):
        super().__init__(
            message=f"Duplicate edge detected: {edge_id}. Each edge in the workflow must be unique.",
            validation_type=ValidationTypeEnum.EDGE_DUPLICATION,
        )
        self.edge_id = edge_id


class ExecutorDuplicationError(WorkflowValidationError):
    """重複したexecutor識別子が検出された場合に発生する例外。"""

    def __init__(self, executor_id: str):
        super().__init__(
            message=(
                f"Duplicate executor id detected: '{executor_id}'. Executor ids must be globally unique within a "
                "workflow."
            ),
            validation_type=ValidationTypeEnum.EXECUTOR_DUPLICATION,
        )
        self.executor_id = executor_id


class TypeCompatibilityError(WorkflowValidationError):
    """接続されたexecutor間で型の非互換性が検出された場合に発生する例外。"""

    def __init__(
        self,
        source_executor_id: str,
        target_executor_id: str,
        source_types: list[type[Any]],
        target_types: list[type[Any]],
    ):
        # 非互換な型のプレースホルダーを使用 - WorkflowGraphValidatorで計算される。
        super().__init__(
            message=f"Type incompatibility between executors '{source_executor_id}' -> '{target_executor_id}'. "
            f"Source executor outputs types {[str(t) for t in source_types]} but target executor "
            f"can only handle types {[str(t) for t in target_types]}.",
            validation_type=ValidationTypeEnum.TYPE_COMPATIBILITY,
        )
        self.source_executor_id = source_executor_id
        self.target_executor_id = target_executor_id
        self.source_types = source_types
        self.target_types = target_types


class GraphConnectivityError(WorkflowValidationError):
    """グラフの接続性の問題が検出された場合に発生する例外。"""

    def __init__(self, message: str):
        super().__init__(message, validation_type=ValidationTypeEnum.GRAPH_CONNECTIVITY)


class InterceptorConflictError(WorkflowValidationError):
    """同じサブワークフローから同じリクエストタイプを複数のexecutorがインターセプトした場合に発生する例外。"""

    def __init__(self, message: str):
        super().__init__(message, validation_type=ValidationTypeEnum.INTERCEPTOR_CONFLICT)


# endregion region Workflow Graph Validator
class WorkflowGraphValidator:
    """ワークフローグラフのバリデータ。

    このバリデータは複数の検証チェックを実行する:
    1. エッジの重複検証
    2. 接続されたexecutor間の型互換性検証
    3. グラフの接続性検証

    """

    def __init__(self) -> None:
        self._edges: list[Edge] = []
        self._executors: dict[str, Executor] = {}
        self._duplicate_executor_ids: set[str] = set()
        self._start_executor_ref: Executor | str | None = None

    # region Core Validation Methods
    def validate_workflow(
        self,
        edge_groups: Sequence[EdgeGroup],
        executors: dict[str, Executor],
        start_executor: Executor | str,
        *,
        duplicate_executor_ids: Sequence[str] | None = None,
    ) -> None:
        """ワークフローグラフ全体を検証する。

        Args:
            edge_groups: ワークフロー内のエッジグループのリスト
            executors: executor IDからexecutorインスタンスへのマップ
            start_executor: 開始executor（インスタンスまたはID）

        Keyword Args:
            duplicate_executor_ids: 事前に設定された既知の重複executor IDのリスト（Optional）

        Raises:
            WorkflowValidationError: 検証に失敗した場合

        """
        self._executors = executors
        self._edges = [edge for group in edge_groups for edge in group.edges]
        self._edge_groups = edge_groups
        self._duplicate_executor_ids = set(duplicate_executor_ids or [])
        self._start_executor_ref = start_executor

        # 開始executorのみが存在する場合、それをexecutorマップに追加する
        # ワークフローが単一のexecutorのみでエッジがない特別なケースを処理する。 この場合、executorマップは参照するエッジグループがないため空になる。
        # 開始executorをマップに追加することで、単一executorのワークフロー（エッジなし）をサポートし、 検証と実行が可能になる。
        if not self._executors and start_executor and isinstance(start_executor, Executor):
            self._executors[start_executor.id] = start_executor

        # start_executorがグラフに存在することを検証する WorkflowBuilderでチェックしているはずだが、完全性のためここでも行う。
        start_executor_id = start_executor.id if isinstance(start_executor, Executor) else start_executor
        if start_executor_id not in self._executors:
            raise GraphConnectivityError(f"Start executor '{start_executor_id}' is not present in the workflow graph")

        # 追加の存在確認: ビルダー経由でのみ注入されたstart_executor（executorsマップに存在）で、
        # 他のexecutorがエッジで参照されているのにstart_executorが参照されていない場合、
        # 設定エラーを示す。選択された開始ノードは実質的に切断されているか、 定義されたグラフトポロジーに不明である。
        # 単一ノードのワークフロー（エッジなし）では開始executorが単独で存在することを許容する（上記でマップに注入して処理）。
        # この詳細なチェックは少なくとも1つのエッジグループが定義されている場合にのみ行う。
        if self._edges:  # Only evaluate when the workflow defines edges
            edge_executor_ids: set[str] = set()
            for _e in self._edges:
                edge_executor_ids.add(_e.source_id)
                edge_executor_ids.add(_e.target_id)
            if start_executor_id not in edge_executor_ids:
                raise GraphConnectivityError(
                    f"Start executor '{start_executor_id}' is not present in the workflow graph"
                )

        # すべてのチェックを実行する
        self._validate_executor_id_uniqueness(start_executor_id)
        self._validate_edge_duplication()
        self._validate_handler_output_annotations()
        self._validate_type_compatibility()
        self._validate_graph_connectivity(start_executor_id)
        self._validate_self_loops()
        self._validate_dead_ends()
        self._validate_cycles()

    def _validate_handler_output_annotations(self) -> None:
        """各ハンドラのctxパラメータがWorkflowContext[T]で注釈されていることを検証する。

        注意: この検証は現在、@handlerデコレータ適用時に
        _workflow_context.pyの統合検証関数で主に処理されている。
        このメソッドは例外的なケースのために最小限に保たれている。

        """
        # 包括的な検証はすでにハンドラ登録時に行われている: 1. @handlerデコレータがvalidate_function_signature()を呼び出す
        # 2. FunctionExecutorコンストラクタがvalidate_function_signature()を呼び出す 3.
        # 両者ともWorkflowContext検証にvalidate_workflow_context_annotation()を使用
        # ワークフロー内のすべてのexecutorはこれらのいずれかの経路を通っているため、 ここでの冗長な検証は不要で削除されている。
        pass

    # endregion

    def _validate_executor_id_uniqueness(self, start_executor_id: str) -> None:
        """ワークフローグラフ全体でexecutor識別子が一意であることを保証する。"""
        duplicates: set[str] = set(self._duplicate_executor_ids)

        id_counts: defaultdict[str, int] = defaultdict(int)
        for key, executor in self._executors.items():
            id_counts[executor.id] += 1
            if key != executor.id:
                duplicates.add(executor.id)

        duplicates.update({executor_id for executor_id, count in id_counts.items() if count > 1})

        if isinstance(self._start_executor_ref, Executor):
            mapped = self._executors.get(start_executor_id)
            if mapped is not None and mapped is not self._start_executor_ref:
                duplicates.add(start_executor_id)

        if duplicates:
            raise ExecutorDuplicationError(sorted(duplicates)[0])

    # region Edge and Type Validation
    def _validate_edge_duplication(self) -> None:
        """ワークフロー内に重複したエッジがないことを検証する。

        Raises:
            EdgeDuplicationError: 重複エッジが見つかった場合

        """
        seen_edge_ids: set[str] = set()

        for edge in self._edges:
            edge_id = edge.id
            if edge_id in seen_edge_ids:
                raise EdgeDuplicationError(edge_id)
            seen_edge_ids.add(edge_id)

    def _validate_type_compatibility(self) -> None:
        """接続されたexecutor間の型互換性を検証する。

        これは、ソースexecutorの出力型がターゲットexecutorの入力型と互換性があるかをチェックする。

        Raises:
            TypeCompatibilityError: 型の非互換性が検出された場合

        """
        for edge_group in self._edge_groups:
            for edge in edge_group.edges:
                self._validate_edge_type_compatibility(edge, edge_group)

    def _validate_edge_type_compatibility(self, edge: Edge, edge_group: EdgeGroup) -> None:
        """特定のエッジの型互換性を検証する。

        これは、ソースexecutorの出力型がターゲットexecutorの入力型と互換性があるかをチェックする。

        Args:
            edge: 検証対象のエッジ
            edge_group: このエッジを含むエッジグループ

        Raises:
            TypeCompatibilityError: 型の非互換性が検出された場合

        """
        source_executor = self._executors[edge.source_id]
        target_executor = self._executors[edge.target_id]

        # ソースexecutorから出力型を取得する
        source_output_types = list(source_executor.output_types)

        # ターゲットexecutorから入力型を取得する
        target_input_types = target_executor.input_types

        # いずれかのexecutorに型情報がない場合、警告をログに記録し検証をスキップします。
        # これは動的型付けのシナリオを許容しますが、検証範囲が狭まることを警告します。
        if not source_output_types or not target_input_types:
            # 動的型付けが予想されるRequestInfoExecutorの警告を抑制します。
            if not source_output_types and not isinstance(source_executor, RequestInfoExecutor):
                logger.warning(
                    f"Executor '{source_executor.id}' has no output type annotations. "
                    f"Type compatibility validation will be skipped for edges from this executor. "
                    f"Consider adding WorkflowContext[T] generics in handlers for better validation."
                )
            if not target_input_types and not isinstance(target_executor, RequestInfoExecutor):
                logger.warning(
                    f"Executor '{target_executor.id}' has no input type annotations. "
                    f"Type compatibility validation will be skipped for edges to this executor. "
                    f"Consider adding type annotations to message handler parameters for better validation."
                )
            return

        # 任意のsource出力型が任意のtarget入力型と互換性があるかをチェックします。
        compatible = False
        compatible_pairs: list[tuple[type[Any], type[Any]]] = []

        for source_type in source_output_types:
            for target_type in target_input_types:
                if isinstance(edge_group, FanInEdgeGroup):
                    # エッジがエッジグループの一部である場合、targetはデータ型のリストを期待します。
                    if is_type_compatible(list[source_type], target_type):  # type: ignore[valid-type]
                        compatible = True
                        compatible_pairs.append((list[source_type], target_type))  # type: ignore[valid-type]
                else:
                    if is_type_compatible(source_type, target_type):
                        compatible = True
                        compatible_pairs.append((source_type, target_type))

        # デバッグのために型互換性の成功をログに記録します。
        if compatible:
            logger.debug(
                f"Type compatibility validated for edge '{source_executor.id}' -> '{target_executor.id}'. "
                f"Compatible type pairs: {[(str(s), str(t)) for s, t in compatible_pairs]}"
            )

        if not compatible:
            # より詳細な情報を含む拡張エラー。
            raise TypeCompatibilityError(
                source_executor.id,
                target_executor.id,
                source_output_types,
                target_input_types,
            )

    # endregion region Graph Connectivity Validation
    def _validate_graph_connectivity(self, start_executor_id: str) -> None:
        """グラフの接続性を検証し、潜在的な問題を検出します。

        これには以下のチェックが含まれます:
        - 開始ノードから到達不能なexecutorの検出
        - 孤立したexecutor（入出力エッジなし）の検出
        - 無限ループの可能性に関する警告

        Args:
            start_executor_id: 開始executorのID

        Raises:
            GraphConnectivityError: 接続性の問題が検出された場合

        """
        # グラフの隣接リストを構築します。
        graph: dict[str, list[str]] = defaultdict(list)
        all_executors = set(self._executors.keys())

        for edge in self._edges:
            graph[edge.source_id].append(edge.target_id)

        # 開始ノードから到達可能なノードを見つけます。
        reachable = self._find_reachable_nodes(graph, start_executor_id)

        # 到達不能なexecutorをチェックします。
        unreachable = all_executors - reachable
        if unreachable:
            raise GraphConnectivityError(
                f"The following executors are unreachable from the start executor '{start_executor_id}': "
                f"{sorted(unreachable)}. This may indicate a disconnected workflow graph."
            )

        # 孤立したexecutor（エッジなし）をチェックします。
        isolated_executors: list[str] = []
        for executor_id in all_executors:
            has_incoming = any(edge.target_id == executor_id for edge in self._edges)
            has_outgoing = any(edge.source_id == executor_id for edge in self._edges)

            if not has_incoming and not has_outgoing and executor_id != start_executor_id:
                isolated_executors.append(executor_id)

        if isolated_executors:
            raise GraphConnectivityError(
                f"The following executors are isolated (no incoming or outgoing edges): "
                f"{sorted(isolated_executors)}. Isolated executors will never be executed."
            )

    def _find_reachable_nodes(self, graph: dict[str, list[str]], start: str) -> set[str]:
        """DFSを用いて開始ノードから到達可能なすべてのノードを見つけます。

        Args:
            graph: グラフの隣接リスト表現
            start: 開始ノードID

        Returns:
            到達可能なノードIDのセット

        """
        visited: set[str] = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(graph[node])

        return visited

    # endregion region Additional Validation Scenarios
    def _validate_self_loops(self) -> None:
        """自己ループ（executorから自身へのエッジ）を検出しログに記録します。

        自己ループは再帰的処理を示す可能性があり意図的な場合もありますが、
        レビューのために強調表示すべきです。

        """
        self_loops = [edge for edge in self._edges if edge.source_id == edge.target_id]

        for edge in self_loops:
            logger.warning(
                f"Self-loop detected: Executor '{edge.source_id}' connects to itself. "
                f"This may cause infinite recursion if not properly handled with conditions."
            )

    def _validate_dead_ends(self) -> None:
        """出力エッジを持たないexecutor（潜在的なデッドエンド）を特定します。

        これらは意図的な終了ノードか、接続の欠落を示す可能性があります。

        """
        executors_with_outgoing = {edge.source_id for edge in self._edges}
        all_executor_ids = set(self._executors.keys())
        dead_ends = all_executor_ids - executors_with_outgoing

        if dead_ends:
            logger.info(
                f"Dead-end executors detected (no outgoing edges): {sorted(dead_ends)}. "
                f"Verify these are intended as final nodes in the workflow."
            )

    def _validate_cycles(self) -> None:
        """ワークフローグラフのサイクルを検出します。

        サイクルは反復処理のために意図的な場合がありますが、
        適切な終了条件が存在することを確認するためにレビューの対象とすべきです。
        同じワークフローを再構築する際のノイズを避けるため、
        各異なるサイクルグループはプロセスごとに一度だけ報告されます。

        """
        # 隣接リストを構築します（出力エッジがなくてもすべてのexecutorが含まれるように）。
        graph: dict[str, list[str]] = defaultdict(list)
        for edge in self._edges:
            graph[edge.source_id].append(edge.target_id)
            graph.setdefault(edge.target_id, [])
        for executor_id in self._executors:
            graph.setdefault(executor_id, [])

        # Tarjanのアルゴリズムを用いてサイクルを形成する強連結成分を特定します。
        index: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        current_index = 0
        cycle_components: list[list[str]] = []

        def strongconnect(node: str) -> None:
            nonlocal current_index

            index[node] = current_index
            lowlink[node] = current_index
            current_index += 1
            stack.append(node)
            on_stack.add(node)

            for neighbor in graph[node]:
                if neighbor not in index:
                    strongconnect(neighbor)
                    lowlink[node] = min(lowlink[node], lowlink[neighbor])
                elif neighbor in on_stack:
                    lowlink[node] = min(lowlink[node], index[neighbor])

            if lowlink[node] == index[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break

                # 強連結成分は、複数ノードを持つか、単一ノードが自身を直接参照する場合にサイクルを表します。
                if len(component) > 1 or any(member in graph[member] for member in component):
                    cycle_components.append(component)

        for executor_id in graph:
            if executor_id not in index:
                strongconnect(executor_id)

        if not cycle_components:
            return

        unseen_components: list[list[str]] = []
        for component in cycle_components:
            signature = tuple(sorted(component))
            if signature in _LOGGED_CYCLE_SIGNATURES:
                continue
            _LOGGED_CYCLE_SIGNATURES.add(signature)
            unseen_components.append(component)

        if not unseen_components:
            # このプロセスで既に報告されたすべてのサイクル。ノイズを抑えつつ追跡可能性を保持します。
            logger.debug(
                "Cycle detected in workflow graph but previously reported. Components: %s",
                [sorted(component) for component in cycle_components],
            )
            return

        def _format_cycle(component: list[str]) -> str:
            if not component:
                return ""
            ordered = list(component)
            ordered.append(component[0])
            return " -> ".join(ordered)

        formatted_cycles = ", ".join(_format_cycle(component) for component in unseen_components)
        logger.warning(
            "Cycle detected in the workflow graph involving: %s. Ensure termination or iteration limits exist.",
            formatted_cycles,
        )

    # endregion


# endregion


def validate_workflow_graph(
    edge_groups: Sequence[EdgeGroup],
    executors: dict[str, Executor],
    start_executor: Executor | str,
    *,
    duplicate_executor_ids: Sequence[str] | None = None,
) -> None:
    """ワークフローグラフを検証するための便利関数。

    Args:
        edge_groups: ワークフロー内のエッジグループのリスト
        executors: executor IDからexecutorインスタンスへのマップ
        start_executor: 開始executor（インスタンスまたはID）

    Keyword Args:
        duplicate_executor_ids: 既知の重複executor IDのリスト（事前に設定可能）

    Raises:
        WorkflowValidationError: 検証に失敗した場合

    """
    validator = WorkflowGraphValidator()
    validator.validate_workflow(
        edge_groups,
        executors,
        start_executor,
        duplicate_executor_ids=duplicate_executor_ids,
    )
