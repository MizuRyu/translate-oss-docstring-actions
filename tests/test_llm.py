import asyncio
import os
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in os.sys.path:
    os.sys.path.insert(0, PROJECT_ROOT)

from llm import translate_batch  # noqa: E402


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
            translate_batch(system_prompt, entries, mock_mode=False)
        )

        self.assertIsNone(error)
        self.assertIsNotNone(translations)
        self.assertEqual(len(translations or []), len(entries))
        self.assertGreaterEqual(stats.get("primary_requests", 0), 0)


if __name__ == "__main__":
    unittest.main()
