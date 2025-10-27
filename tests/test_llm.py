import asyncio
import os
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm import (  # noqa: E402
    ConcurrencyLimiter,
    TranslationRequestError,
    _get_concurrency_limiter,
    _get_github_model_policy,
    translate_batch,
)


class LlmIntegrationTests(unittest.TestCase):
    def test_translate_batch_with_azure_fallback(self) -> None:
        """GitHub Models + Azure Inferenceフォールバック統合テスト"""
        # given: GitHub ModelsとAzure Inferenceの両方が設定されている
        required_github = ["GITHUB_MODELS_ENDPOINT", "GITHUB_TOKEN", "GITHUB_MODELS_MODEL"]
        required_azure = ["AZURE_INFERENCE_ENDPOINT", "AZURE_INFERENCE_CREDENTIAL", "AZURE_INFERENCE_MODEL"]

        if not all(os.getenv(key) for key in required_github):
            self.skipTest("GitHub Modelsの接続情報が設定されていません")
        if not all(os.getenv(key) for key in required_azure):
            self.skipTest("Azure Inferenceの接続情報が設定されていません")

        entries = [
            {
                "path": "tests/data/sample.py",
                "kind": "comment",
                "text": "Hello world",
                "meta": {},
            }
        ]
        system_prompt = "Return text as-is."

        # when: translate_batchを実際のAPIで実行する
        translations, error, stats = asyncio.run(
            translate_batch(system_prompt, entries, is_mock=False)
        )

        # then: 翻訳が成功し、結果が返される
        self.assertIsNone(error)
        self.assertIsNotNone(translations)
        self.assertEqual(len(translations or []), len(entries))
        self.assertGreaterEqual(stats.get("primary_requests", 0), 0)


class ConcurrencyLimiterTests(unittest.TestCase):
    def test_limiter_keeps_parallel_tasks_under_limit(self) -> None:
        """同時実行数が制限を超えない"""
        async def scenario() -> int:
            # given: 同時実行数2のリミッタを取得する
            limiter = _get_concurrency_limiter("openai/gpt-4.1-mini", 2)
            self.assertIsNotNone(limiter)
            active = 0
            max_active = 0
            lock = asyncio.Lock()

            async def worker() -> None:
                nonlocal active, max_active
                async with limiter.slot():  # type: ignore[union-attr]
                    async with lock:
                        active += 1
                        max_active = max(max_active, active)
                    # when: リミッタ内で短時間処理を行う
                    await asyncio.sleep(0.01)
                    async with lock:
                        active -= 1

            # when: 5つのタスクを並列実行する
            await asyncio.gather(*(worker() for _ in range(5)))
            
            # then: 同時実行の最大値が制限以下である
            return max_active

        max_active = asyncio.run(scenario())
        self.assertLessEqual(max_active, 2)
        self.assertEqual(max_active, 2)

    def test_invalid_concurrency_value_is_rejected(self) -> None:
        """0以下の同時実行数は拒否される"""
        # given: 0以下の同時実行数
        # when: ConcurrencyLimiterを生成しようとする
        # then: ValueErrorが発生する
        with self.assertRaises(ValueError):
            ConcurrencyLimiter(0)


class GithubModelPolicyTests(unittest.TestCase):
    def test_unknown_model_raises_translation_request_error(self) -> None:
        """未登録モデルはTranslationRequestErrorを発生させる"""
        # given: 未登録のモデル名
        # when: _get_github_model_policyを呼び出す
        # then: TranslationRequestErrorが発生する
        with self.assertRaises(TranslationRequestError):
            _get_github_model_policy("unknown/model")


class TranslateBatchMockTests(unittest.TestCase):
    """translate_batch関数のモックテスト"""

    def test_mock_mode_returns_original_text(self) -> None:
        """モックモードでは元のテキスト+(mock)を返す"""
        # given: モックモードが有効で、2つのエントリがある
        entries = [
            {"path": "test.py", "kind": "comment", "text": "Hello", "meta": {}},
            {"path": "test.py", "kind": "docstring", "text": "World", "meta": {}},
        ]
        system_prompt = "Translate to Japanese"

        # when: translate_batchをモックモードで実行する
        translations, error, stats = asyncio.run(
            translate_batch(system_prompt, entries, is_mock=True)
        )

        # then: 元のテキストに"(mock)"が付加されて返される
        self.assertIsNone(error)
        self.assertIsNotNone(translations)
        self.assertEqual(len(translations), 2)
        self.assertEqual(translations[0], "Hello (mock)")
        self.assertEqual(translations[1], "World (mock)")
        self.assertEqual(stats["primary_requests"], 0)

    @patch("llm._invoke_client")
    @patch("llm._get_concurrency_limiter")
    @patch("llm.get_github_client")
    def test_translate_batch_success_with_primary(
        self, mock_get_client, mock_limiter, mock_invoke
    ) -> None:
        """Primary成功時の翻訳"""
        # given: Primaryクライアントが正常に翻訳を返す設定
        mock_limiter_instance = MagicMock()
        
        @asynccontextmanager
        async def mock_slot():
            yield
        
        mock_limiter_instance.slot = mock_slot
        mock_limiter.return_value = mock_limiter_instance

        # Primaryクライアントが"こんにちは"を返すようにモック
        mock_invoke.return_value = ["こんにちは"]

        entries = [{"path": "test.py", "kind": "comment", "text": "Hello", "meta": {}}]
        system_prompt = "Translate to Japanese"

        # when: translate_batchを実行する
        translations, error, stats = asyncio.run(
            translate_batch(system_prompt, entries, is_mock=False)
        )

        # then: Primaryクライアントから正常に翻訳結果が返される
        self.assertIsNone(error)
        self.assertIsNotNone(translations)
        self.assertEqual(len(translations), 1)
        self.assertEqual(translations[0], "こんにちは")
        self.assertGreater(stats["primary_requests"], 0)


if __name__ == "__main__":
    unittest.main()
