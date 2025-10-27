# Actions TODO

- [ ] translate-test.yml: subdirectory入力の安全チェック（../ や絶対パス禁止）を本番同様に追加する。
- [ ] translate-test.yml: translation_limit の扱い（文字列入力、"0"で無制限）に統一し、ドキュメントにも明記する。
- [ ] 2つのworkflowにある checkout / uv sync / git clone の共通ステップを composite action 化する計画を検討する。
- [x] translate-test.yml でも token-summary を artifact に含めるか検討し、行動方針を決める。
- [ ] 失敗 JSONL のリカバリ用 workflow を別途設計する（仕様に記載された将来タスク）。
- [ ] rsync で `--ignore-existing` 以外にも安全策が必要か再検証し、必要ならより厳格な同期方法を検討する。
