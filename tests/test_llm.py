import asyncio
import os
import sys
import unittest
from pathlib import Path

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

        translations, error, stats = asyncio.run(
            translate_batch(system_prompt, entries, is_mock=False)
        )

        self.assertIsNone(error)
        self.assertIsNotNone(translations)
        self.assertEqual(len(translations or []), len(entries))
        self.assertGreaterEqual(stats.get("primary_requests", 0), 0)


class ConcurrencyLimiterTests(unittest.TestCase):
    def test_limiter_keeps_parallel_tasks_under_limit(self) -> None:
        async def scenario() -> int:
            # given：同時実行数2のリミッタを取得する
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
                    # when：リミッタ内で短時間処理を行う
                    await asyncio.sleep(0.01)
                    async with lock:
                        active -= 1

            await asyncio.gather(*(worker() for _ in range(5)))
            # then：同時実行の最大値が制限以下である
            return max_active

        max_active = asyncio.run(scenario())
        self.assertLessEqual(max_active, 2)
        self.assertEqual(max_active, 2)

    def test_invalid_concurrency_value_is_rejected(self) -> None:
        # given：0以下の同時実行数
        with self.assertRaises(ValueError):
            ConcurrencyLimiter(0)


class GithubModelPolicyTests(unittest.TestCase):
    def test_unknown_model_raises_translation_request_error(self) -> None:
        # given：未登録モデル名
        with self.assertRaises(TranslationRequestError):
            _get_github_model_policy("unknown/model")


if __name__ == "__main__":
    unittest.main()
