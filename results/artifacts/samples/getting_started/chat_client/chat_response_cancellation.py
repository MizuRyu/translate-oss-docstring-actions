# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework.openai import OpenAIChatClient

"""
Chat Response Cancellation Example

Demonstrates proper cancellation of streaming chat responses during execution.
Shows asyncio task cancellation and resource cleanup techniques.
"""


async def main() -> None:
    """
    1秒後にチャットリクエストをキャンセルすることを示します。
    チャットリクエストのタスクを作成し、少し待ってからキャンセルして適切なクリーンアップを示します。

    設定:
    - OpenAIモデルID: "model_id"パラメータまたは"OPENAI_CHAT_MODEL_ID"環境変数を使用
    - OpenAI APIキー: "api_key"パラメータまたは"OPENAI_API_KEY"環境変数を使用

    """
    chat_client = OpenAIChatClient()

    try:
        task = asyncio.create_task(chat_client.get_response(messages=["Tell me a fantasy story."]))
        await asyncio.sleep(1)
        task.cancel()
        await task
    except asyncio.CancelledError:
        print("Request was cancelled")


if __name__ == "__main__":
    asyncio.run(main())
