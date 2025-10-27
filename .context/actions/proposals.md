## Proposal

- アクション全体を「checkout → 抽出 → トークンサマリ提示 → ユーザー承認 → 翻訳（test mode対応）→ 反映」というステージに分割し、各ステージの入出力を JSONL で明確化する。
- `translate.yml`（本番）と `translate-test.yml`（検証）を分離し、検証ワークフローでは test mode で LLM 呼び出し回数を制限して差分確認を行いやすくする。
- リポジトリ URL／サブディレクトリ指定を workflow 入力として受け取り、`actions/checkout` ではなく追加の `git clone` で `workspace/<target>` に配置、成果物 JSONL は `out/` 配下の一時領域に保存する。
- 抽出直後にトークンサマリ（件数・総トークン・平均等）を計算し、承認ステップ（environment 承認 or github-script）でユーザーに提示して Human-in-the-loop を実現する。
- `translate.py` の `exclude_terms` に workflow 入力の自然言語文字列をそのまま渡し、`.format()` でテンプレートに埋め込む形を採用する。
- `act` でのローカル検証を想定し、Secrets 不要でモックが通る `translate-test.yml` を先に整備、本番用は Secrets 未設定時に fail-fast する。

## TODO

- [ ] `translate.yml` / `translate-test.yml` のジョブ構成図を作成し、ステージごとの入力・出力パスを明記する。
- [ ] `extract → token report → approval → translate → replace` 各ステップで使用する CLI コマンド・環境変数・成果物のパスを整理する。
- [ ] Human-in-the-loop 承認をどの機構（environment 承認 or github-script）で実装するか決め、サンプルを試作する。
- [ ] test mode 用の最大実行回数パラメータを workflow `inputs` に追加し、CLI への渡し方を定義する。
- [ ] JSONL やログの保存パスを `out/` 以下に統一し、Artifacts へアップロードするかどうか（本番／検証それぞれ）を決定する。
- [ ] `act` で動かすための Secrets 擬似値設定などローカル検証手順を整理し、後で README に反映する計画を立てる。

---

### 2025-10-27 メモ
- `translate.yml` (本番) と `translate-test.yml` (検証) を追加。`translate.yml` は `translation-approval` 環境で承認を要し、抽出結果のトークンサマリを表示後に翻訳を実行する。成果物は `out/` 配下のみを artifact 化し、JSONL 本体はGitHub上にログとしては残さない。
- `translate-test.yml` は `mock_mode` や `max_records` を入力で制御できる検証フロー。`act` などローカル検証でも Secrets 不要で実行可能。
- 両ワークフローとも `exclude_terms` を自然言語で受け取り、テンプレートに埋め込む形式に統一。今後 workflow_dispatch 入力を増やす場合も `.format()` で対応できる。
- 成果物は `artifact_dir` 入力（既定値 `translated`）にコピーしつつ、必要に応じて `upload-artifact` で取得できる構成にした。
