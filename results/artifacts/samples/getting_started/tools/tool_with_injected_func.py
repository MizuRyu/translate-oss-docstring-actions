# Copyright (c) Microsoft. All rights reserved.
# type: ignore
"""
依存性注入を用いたAIFunction Toolの例

この例は、agentフレームワークの依存性注入システムを使ってAIFunctionツールを作成する方法を示す。
初期化時に関数を提供する代わりに、実際の呼び出し可能な関数は辞書定義からのデシリアライズ時に注入される。

注意:
    この例で使われているシリアライズとデシリアライズの機能は現在開発中である。
    今後のバージョンでAPIが変更される可能性があり、機能の改善と拡張が続けられる予定である。
    依存性注入パターンの最新情報については最新のドキュメントを参照されたい。

使い方:
    このスクリプトを実行すると、辞書定義から関数を実行時に注入してAIFunctionツールを作成する方法がわかる。
    Agentはこのツールを使って算術演算を行う。
"""

import asyncio

from agent_framework import AIFunction
from agent_framework.openai import OpenAIResponsesClient

definition = {
    "type": "ai_function",
    "name": "add_numbers",
    "description": "Add two numbers together.",
    "input_model": {
        "properties": {
            "a": {"description": "The first number", "type": "integer"},
            "b": {"description": "The second number", "type": "integer"},
        },
        "required": ["a", "b"],
        "title": "func_input",
        "type": "object",
    },
}


async def main() -> None:
    """関数を注入してツールを作成することを示すメイン関数。"""

    def func(a, b) -> int:
        """2つの数を加算する。"""
        return a + b

    # 依存性注入を使ってAIFunctionツールを作成する 'definition'辞書はシリアライズされたツール設定を含み、
    # 実際の関数実装はdependencies経由で提供される。  依存性構造: {"ai_function": {"name:add_numbers":
    # {"func": func}}} - "ai_function": ツールタイプ識別子に一致 - "name:add_numbers":
    # name="add_numbers"のツールに対するインスタンス固有の注入 - "func": 注入される関数を受け取るパラメータ名
    tool = AIFunction.from_dict(definition, dependencies={"ai_function": {"name:add_numbers": {"func": func}}})

    agent = OpenAIResponsesClient().create_agent(
        name="FunctionToolAgent", instructions="You are a helpful assistant.", tools=tool
    )
    response = await agent.run("What is 5 + 3?")
    print(f"Response: {response.text}")


if __name__ == "__main__":
    asyncio.run(main())
