# AGENTS.md

このファイルは、AIコーディングエージェントがこのPythonリポジトリで作業する際のガイダンスを提供します。

## プロジェクト哲学

**関数指向の設計:**
- オブジェクトの状態管理が必要な場合を除き、クラスよりも関数を優先する
- コードは最小限に保つ - 不要な早期リターンやnullチェックを避ける
- 特に指示がない限り、コメントやコミットは日本語で記述する

**コード制約:**
- 関数の長さ: ≤ 40行（推奨: 20-30行）
- ファイルの長さ: ≤ 400行
- ネストの深さ: ≤ 3レベル
- パラメータ数: ≤ 5個
- クラスの責務: 単一責任原則（SRP）を厳守

## アーキテクチャ概要

これは、libcstによるAST操作とGitHub Models/Azure AI InferenceによるLLM翻訳を使用して、Pythonコードベースのdocstringとコメントを翻訳するPython CLIツールです。

**コアデータフロー:**

1. **Extract** (`src/extract.py`) - libcstでPythonファイルを解析し、docstring/コメントを抽出
2. **Translate** (`src/translate.py`) - LLM APIを使用して抽出テキストをバッチ翻訳
3. **Replace** (`src/replace.py`) - 位置情報ベースの置換で翻訳結果をソースファイルに適用

**主要なデータ構造:**

- 抽出エントリはJSONL形式: `path`, `kind`, `text`, `line_start`, `line_end`, `col_offset`
- 翻訳バッチはトークン制限でグループ化（1リクエストあたり最大2,500トークン）
- GitHub Modelsシーケンス: `openai/gpt-4.1` → `openai/gpt-4.1-mini` → Azure Fallback

**外部依存関係:**

- `libcst` - AST解析とコード修正
- `tiktoken` - トークンカウント
- GitHub Models API（無料枠）とAzure AI Inference（フォールバック）
- レート制限: RPM (10/15), RPD (150), 同時実行数 (2/5)

## 開発コマンド

**テストと品質管理:**

- `make test` - unittestで全テストを実行
- `make extract` - docstring/コメントの抽出のみ実行
- `make translate` - 抽出データの翻訳のみ実行
- `make replace` - 翻訳結果のソースファイルへの適用のみ実行
- `make pipeline` - 全パイプライン（extract→translate→replace）を実行
- `make clean` - 生成ファイル（out/）を削除

**開発時の使用方法:**

- `uv run python main.py extract test_data --output out/extracted.jsonl`
- `uv run python main.py translate out/extracted.jsonl --output out/translated.jsonl`
- `uv run python main.py replace out/translated.jsonl --output-dir out/translated_sources`

**環境変数:**

- `GITHUB_TOKEN` - GitHub Models APIトークン（必須）
- `GITHUB_MODELS_ENDPOINT` - GitHub Modelsエンドポイント（デフォルト: https://models.inference.ai.azure.com）
- `AZURE_INFERENCE_ENDPOINT` - Azure AI Inferenceエンドポイント（フォールバック用）
- `AZURE_INFERENCE_CREDENTIAL` - Azure AI Inference APIキー（フォールバック用）
- `AZURE_INFERENCE_MODEL` - Azureモデル名（デフォルト: gpt-4.1-mini）
- `LOG_LEVEL` - ログレベル（DEBUG/INFO/WARNING/ERROR）

## コードスタイル

**命名規則:**

- 変数/関数: `snake_case`（例: `parse_config`, `user_input`）
- クラス: `PascalCase`（例: `DataProcessor`, `InputValidator`）
- 定数: `UPPER_SNAKE_CASE`（例: `MAX_RETRY_COUNT`, `API_ENDPOINT`）
- プライベート関数/メソッド: `_leading_underscore`（例: `_validate_data`）

**型ヒント:**

- 関数のパラメータと戻り値には必ず型ヒントを使用
- 柔軟な辞書には`Dict[str, Any]`を使用
- `typing`モジュールから`List[Type]`, `Optional[Type]`を使用
- ファイルパスには`pathlib`の`Path`を使用

**エラーハンドリング:**

- ドメイン固有の例外を使用（例: `DataProcessingError`, `ProcessingTimeoutError`）
- 外部ライブラリの例外は`raise ... from e`でカスタム例外でラップ
- try-exceptブロックは最小限に - 例外が発生する可能性のある操作のみをラップ
- 例外を発生させる前にコンテキスト付きでログを記録

**インポート規則:**

- 標準ライブラリ、サードパーティ、ローカルインポートの順
- プロジェクトモジュールには絶対インポートを使用（例: `from extract import run`）
- インポートは論理的にグループ化し、グループ内でアルファベット順に配置

**マジックナンバー:**

- コード内にマジックナンバーを使用しない - 常に定数を定義
- 例: `if len(text) > 5000:`ではなく`MAX_TEXT_LENGTH = 5000`

**エクスポート規則:**

- 他のモジュールで実際に使用される関数/クラスのみをエクスポート
- 内部ユーティリティはプライベートのまま（エクスポートしない）
- パブリックAPIの表面積を最小化

## Git/PR運用

**Author情報の付与:**

commit時は必ずAuthor情報（name/email）を明示的に設定してコマンド実行すること：

```bash
GIT_AUTHOR_NAME="codex" \
GIT_AUTHOR_EMAIL="codex@noreply.github.com" \
git commit -m "feat: 新機能の追加"
```

**コミットの分割:**

- 一度に変更を`git add .`でまとめない
- ファイル名を指定して`git add`する
- 適切な粒度で複数コミットに分割することを推奨

**コミットメッセージフォーマット:**

```bash
# type(scope): summary

feat(parser): 新しいパーサーを追加
fix(validator): バリデーションのバグ修正
refactor(processor): 処理クラスの責務分離
test(parser): パーサーの境界値テスト追加
docs(readme): インストール手順を更新
chore(deps): 依存パッケージを更新
```

## 生成AI利用時の注意

userの指示が曖昧である場合、良い例のように解釈してください。

**小さな単位での指示:**

```python
# 悪い例: 大きすぎる指示
"""
DataPipelineクラス全体を実装してください。
データ取得、処理、検証、保存、すべての機能を含めてください。
"""

# 良い例: 関数単位での指示
"""
process_data関数を実装してください。
入力: data (str) - 処理対象データ
出力: ProcessedData オブジェクト
制約: 
- 最大5000文字まで処理
- 空入力ではValidationError
- 外部APIを使用
"""
```

