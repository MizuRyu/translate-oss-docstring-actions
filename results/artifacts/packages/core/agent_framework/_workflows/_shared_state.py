# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


class SharedState:
    """ワークフローで共有される状態を管理するクラス。

    SharedStateは、ワークフロー実行中にExecutor間で共有する必要がある状態データへのスレッドセーフなアクセスを提供します。

    予約済みキー:
        以下のキーは内部フレームワーク用に予約されており、ユーザーコードでの変更は避けてください：

        - `_executor_state`: チェックポイント用のExecutor状態を格納（Runnerが管理）

    警告:
        アンダースコア(_)で始まるキーは内部フレームワーク操作用に予約されている可能性があるため使用しないでください。

    """

    def __init__(self) -> None:
        """共有状態を初期化します。"""
        self._state: dict[str, Any] = {}
        self._shared_state_lock = asyncio.Lock()

    async def set(self, key: str, value: Any) -> None:
        """共有状態に値を設定します。"""
        async with self._shared_state_lock:
            await self.set_within_hold(key, value)

    async def get(self, key: str) -> Any:
        """共有状態から値を取得します。"""
        async with self._shared_state_lock:
            return await self.get_within_hold(key)

    async def has(self, key: str) -> bool:
        """共有状態にキーが存在するか確認します。"""
        async with self._shared_state_lock:
            return await self.has_within_hold(key)

    async def delete(self, key: str) -> None:
        """共有状態からキーを削除します。"""
        async with self._shared_state_lock:
            await self.delete_within_hold(key)

    async def clear(self) -> None:
        """共有状態をすべてクリアします。"""
        async with self._shared_state_lock:
            self._state.clear()

    async def export_state(self) -> dict[str, Any]:
        """共有状態全体のシリアライズされたコピーを取得します。"""
        async with self._shared_state_lock:
            return dict(self._state)

    async def import_state(self, state: dict[str, Any]) -> None:
        """シリアライズされた状態辞書から共有状態を復元します。

        これにより現在の状態全体が提供された状態で置き換えられます。

        """
        async with self._shared_state_lock:
            self._state.update(state)

    @asynccontextmanager
    async def hold(self) -> AsyncIterator["SharedState"]:
        """複数操作のために共有状態のロックを保持するコンテキストマネージャ。

        使用例:
            async with shared_state.hold():
                await shared_state.set_within_hold("key", value)
                value = await shared_state.get_within_hold("key")

        """
        async with self._shared_state_lock:
            yield self

    # ロックを取得しない安全でないメソッド（hold()コンテキスト内で使用）。
    async def set_within_hold(self, key: str, value: Any) -> None:
        """ロックを取得せずに値を設定します（安全でないためhold()コンテキスト内で使用してください）。"""
        self._state[key] = value

    async def get_within_hold(self, key: str) -> Any:
        """ロックを取得せずに値を取得します（安全でないためhold()コンテキスト内で使用してください）。"""
        if key not in self._state:
            raise KeyError(f"Key '{key}' not found in shared state.")
        return self._state[key]

    async def has_within_hold(self, key: str) -> bool:
        """ロックを取得せずにキーの存在を確認します（安全でないためhold()コンテキスト内で使用してください）。"""
        return key in self._state

    async def delete_within_hold(self, key: str) -> None:
        """ロックを取得せずにキーを削除します（安全でないためhold()コンテキスト内で使用してください）。"""
        if key in self._state:
            del self._state[key]
        else:
            raise KeyError(f"Key '{key}' not found in shared state.")
