# comment-translator (新構成)

libcst を利用して Python ファイルから docstring / コメント / ログメッセージを抽出し、LLM で翻訳した後に元ソースへ反映するツールです。抽出・翻訳・反映の 3 フェーズを CLI 1 本で扱えます。

## 必要要件

- Python 3.12 以上
- [uv](https://github.com/astral-sh/uv)
- Azure 翻訳を使用する場合は Azure AI Inference にアクセス可能なトークン（`AZURE_INFERENCE_TOKEN` または `GITHUB_TOKEN`）

プロジェクト直下の `uv.toml` で uv のキャッシュ先をローカル (`.uv-cache/`) に固定しています。環境の権限が限定されている場合でも動作させるため、削除しないでください。

## 使い方

1. 依存関係の同期（初回のみ）

   ```bash
   uv sync
   ```

2. 抽出（例: `test_data` 以下を対象）

   ```bash
   uv run python main.py extract test_data \
     --output out/extracted.jsonl \
     --include-log-messages \
     --verbose \
     --log-level INFO
   ```

   - `--include-log-messages`：`print` や `logger.*` を抽出対象に含めます（旧 `--include-runtime-messages`）。
   - `--verbose`：抽出時の追加ログを有効化します（旧 `--include-debug-logs`）。
   - `--exclude` で glob 指定によりファイルやフォルダを除外可能です。

3. 翻訳

   ```bash
   uv run python main.py translate out/extracted.jsonl \
     --output out/translated.jsonl \
     --failed-output out/unprocessed.jsonl \
     --batch-size 5 \
     --mock \
     --log-level INFO
   ```

   - `--mock` を付けると LLM を呼び出さずに `(mock)` を付けた結果を返し、`out/unprocessed.jsonl` には残さず完走します。
   - 実際の GitHub Models を利用する場合は `--mock` を外し、環境変数 `GITHUB_MODELS_ENDPOINT` / `GITHUB_TOKEN` / `GITHUB_MODELS_MODEL` を設定してください。レート制限に遭った場合は `AZURE_INFERENCE_ENDPOINT` / `AZURE_INFERENCE_CREDENTIAL` / `AZURE_INFERENCE_MODEL` が揃っていれば Azure Inference に自動フォールバックします。
   - トークン上限 (既定 2,500) を超えるエントリは即座に `out/unprocessed.jsonl` へ書き出され、警告ログが表示されます。

4. 反映

   ```bash
   uv run python main.py replace out/translated.jsonl \
     --output-dir out/translated_sources \
     --root test_data \
     --mode indirect \
     --log-level INFO
   ```

   - `--root` には元ソースのルートディレクトリを指定してください。生成ファイルは `translated_sources/<元パス>` の構造で出力されます。
   - コメントは原文のブロック単位で置き換えられ、textwrap による折り返しが適用されます。
   - docstring は元のインデント／末尾改行を再現したうえで置換します。
   - `--mode direct` は未実装です。

5. クリーンアップ（任意）

   結果ファイルを再生成したい場合は、`out/extracted.jsonl` / `out/translated.jsonl` / `out/unprocessed.jsonl` / `out/translated_sources/` を削除してから再実行してください。

## テスト

mock 翻訳を使ったシナリオテストと、抽出／置換それぞれの単体テストを `unittest` ベースで用意しています。外部 API には接続しません。

```bash
uv run python -m unittest \
  tests.test_extract \
  tests.test_translate \
  tests.test_replace \
  tests.test_pipeline
```

## ログ出力

- `LOG_LEVEL` 環境変数、または各コマンドの `--log-level` オプションでログレベルを調整できます。
- すべてのステージで実行時間や件数の統計を JSON 形式でログ出力します。
- 翻訳ステージではバッチごとにログを出し、トータルの LLM リクエスト回数も表示します。

## 既知の制限

- コメントの差し戻しは原文ブロックを文字列置換しているため、翻訳後に元のコメントブロックがソースコード上から完全に消えている場合は置換できず警告が表示されます。
- `translate` の Azure モードは構造化出力 (`response_format=json_schema`) を利用します。利用するエンドポイントが `2024-08-01-preview` 以降の API バージョンに対応していることをご確認ください。

## GitHub Actions

### 環境設定（初回のみ）

本番ワークフロー `.github/workflows/translate.yml` は `translation-approval` という environment 承認を前提にしています。

**ghコマンドで設定（推奨）:**

```bash
# 1. 現在のユーザーIDを取得
USER_ID=$(gh api user --jq '.id')

# 2. translation-approval環境に承認者を設定
gh api \
  --method PUT \
  repos/MizuRyu/translate-oss-docstring-actions/environments/translation-approval \
  --input - << EOF
{
  "wait_timer": 0,
  "reviewers": [
    {
      "type": "User",
      "id": $USER_ID
    }
  ]
}
EOF

# 3. 設定を確認
gh api repos/MizuRyu/translate-oss-docstring-actions/environments/translation-approval \
  | jq '.protection_rules[].reviewers[].reviewer.login'
```

**または、GitHub UIで設定:**

1. GitHubリポジトリの Settings → Environments へアクセス
2. `translation-approval` 環境をクリック
3. "Required reviewers" で承認者を設定（自分自身でもOK）
4. 保存

この設定により、translate jobは承認されるまで実行待機状態になります。

### ワークフロー実行

- ワークフローは「抽出→トークンサマリ提示→**承認待ち**→翻訳・反映」の順で進みます
- prepare jobが完了すると、トークン数サマリがStep Summaryに表示されます
- translate jobは承認待ちになり、GitHub UIの "Review deployments" ボタンが表示されます
- 承認後にのみ翻訳が実行されます
- Artifacts として `out/` 配下の JSONL やログをダウンロード可能です
- 指定した `artifact_dir`（既定値 `translated`）にも成果物がコピーされます

### テストワークフロー

検証用の `.github/workflows/translate-test-local.yml` では、`mock_mode` や `max_records` を指定して安全に動作確認できます。`act` を利用したローカル実行時もこちらを使用するとスムーズです。

### act での検証例

actでローカル実行する場合、Makefileを利用すると便利です：

```bash
make translate-test
```

または、直接actコマンドで実行：

```bash
act workflow_dispatch \
  -W .github/workflows/translate-test.yml \
  --container-architecture linux/amd64 \
  --bind \
  --input repository_url=https://github.com/microsoft/agent-framework.git \
  --input subdirectory=python \
  --input max_records=5 \
  --input mock_mode=true \
  --input artifact_dir=translated
```

**重要**: `--bind`オプションを指定することで、コンテナ内で生成された成果物がローカルの`translated/`ディレクトリに保存されます。

実行後、`translated/` 配下に `extracted.jsonl`, `translated.jsonl`, `unprocessed.jsonl`, `token-summary.json`, および `translated_sources/` などが保存されます。必要に応じて `git status` を確認して差分をレビューしてください。
