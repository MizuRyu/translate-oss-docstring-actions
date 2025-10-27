# リポジトリ整備計画 (Refactoring & Cleanup Plan)

**目標**: メンテナンス可能で、見た目が綺麗なコードベースにする

## 📋 Todo List

### ✅ Phase 1: Makefile整備とログ改善（完了）
- [x] `make extract/translate/replace` - 各パイプライン実行
- [x] `make pipeline` - 全パイプライン実行
- [x] `make test` 系 - テストコマンド群
- [x] `make clean` - 生成ファイル削除
- [x] `make act-local` - actでローカルワークフローテスト
- [x] `make help` - ヘルプ表示
- [x] `run.log` ファイル生成を削除
- [x] `.gitignore` 整理

### 🎯 Phase 2: ログ統一とワークフロー改善（次）
**優先度**: High  
**目標**: ログの見た目を統一し、進捗が分かりやすくする

#### 2.1 ログフォーマット統一
- [ ] 各モジュール（extract/translate/replace）のログ形式を統一
- [ ] プログレス表示の改善
  - [ ] 現在: `[Extract] 1/5 filename.py`
  - [ ] 改善: `[Extract] 1/5 (20%) filename.py - 10 items found`
- [ ] 処理完了時のサマリー統一
  - [ ] 処理ファイル数
  - [ ] 抽出/翻訳件数
  - [ ] 成功/失敗件数
  - [ ] 処理時間

#### 2.2 ワークフロー承認画面の改善
- [ ] トークンサマリーに実行予測情報を追加
  - [ ] 翻訳予定件数（translation_limit考慮）
  - [ ] 推定処理時間
  - [ ] 推定コスト（概算）
- [ ] 承認画面のフォーマット改善
  ```
  ## 📊 翻訳実行サマリ
  
  ### 抽出結果
  - 総ファイル数: 150件
  - 抽出件数: 450件
  - 総トークン数: 12,500
  
  ### 翻訳実行予定
  - 翻訳件数: 10件（--limit 10指定）
  - 推定トークン使用量: 278トークン
  - 推定処理時間: 約30秒
  
  ### 設定
  - Mock Mode: false
  - Repository: microsoft/agent-framework
  ```

#### 2.3 エラーハンドリングとリカバリー
- [ ] 途中で止まった場合の進捗保存
  - [ ] どこまで処理したかログに出力
  - [ ] 失敗したアイテムをunprocessed.jsonlに記録（既存機能確認）
- [ ] リトライ機能の確認・改善
- [ ] エラーメッセージの分かりやすさ向上

### 📝 Phase 3: コード品質向上
**優先度**: Medium

#### 3.1 重複コードの削減
- [ ] ログ出力用のヘルパー関数作成
- [ ] サマリー表示の共通化
- [ ] プログレス表示の共通化

#### 3.2 マジックナンバー・文字列の定数化
- [ ] 繰り返し使われる文字列を定数化
- [ ] デフォルト値を constants.py に集約

#### 3.3 関数分割
- [ ] 長い関数を小さな関数に分割
- [ ] 責務の明確化

### 🧪 Phase 4: テストの充実
**優先度**: Medium

- [ ] エッジケースのテスト追加
- [ ] エラーケースのテスト追加
- [ ] モックテストの充実

### 📚 Phase 5: ドキュメント最小限整備
**優先度**: Low

- [ ] README.mdの改善（使用例の充実）
- [ ] CHANGELOG.mdの作成
- [ ] トラブルシューティングセクション

---

## 🎯 優先順位

### High Priority（今やる）
1. ✅ Phase 1: Makefile整備とログ改善（完了）
2. Phase 2: ログ統一とワークフロー改善

### Medium Priority（その次）
3. Phase 3: コード品質向上
4. Phase 4: テストの充実

### Low Priority（余裕があれば）
5. Phase 5: ドキュメント最小限整備

---

## 📝 Notes

### 現在の問題点
- ログフォーマットが統一されていない
- ワークフロー承認画面が情報不足
  - translation_limit指定時、実際何件処理されるか不明
  - 推定処理時間が分からない
- エラー時の進捗保存が不明瞭
- 重複コードが多い（ログ出力等）
- archive/* が残る（不要）

### 整備方針
1. **見た目の統一**: すべてのログを同じフォーマットに
2. **情報の充実**: 実行前に何が起こるか分かるように
3. **エラー対応**: 途中で止まっても状況が分かるように
4. **コード整理**: 重複を減らし、保守しやすく
- [ ] README.mdの改善
  - [ ] バッジ追加（CI status, coverage, version等）
  - [ ] クイックスタートセクション
  - [ ] 使用例の充実
  - [ ] スクリーンショット/デモ
- [ ] GitHub Issues/PR テンプレート作成
- [ ] CI/CDの整備
  - [ ] テスト自動実行
  - [ ] Lint/Format自動チェック
  - [ ] カバレッジレポート
- [ ] pre-commit hooks設定
- [ ] pyproject.tomlのメタデータ充実
  - [ ] プロジェクト説明
  - [ ] キーワード
  - [ ] classifiers
  - [ ] repository URL

### 🔍 Phase 7: コード品質向上
- [ ] type hints完全化
- [ ] docstringの統一（Google/NumPy/Sphinxスタイル選択）
- [ ] Linter設定（ruff, mypy等）
- [ ] Formatter設定（black, ruff format等）
- [ ] 不要なコメント削除
- [ ] magic numberの定数化

## 🎯 優先順位

### High Priority (今すぐやる)
1. Makefile整備
2. ログ出力改善（run.log削除）
3. テストメンテナンス
4. 成果物管理

### Medium Priority (次のフェーズ)
5. ドキュメント整備
6. OSSブラッシュアップ

### Low Priority (余裕があれば)
7. コード品質向上

---

## 📝 Notes

### 現在の問題点
- `run.log` が自動生成されるがgitignoreされていない
- Makefileが最小限（translate-testのみ）
- テストが最新コードと同期していない可能性
- 成果物が散らばっている（out/, translated/）
- ドキュメントがREADME.mdのみ
- CI/CDが未整備

### 理想的な構成
```
project/
├── .context/           # 計画・設計ドキュメント
├── .github/
│   ├── workflows/
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── docs/              # ドキュメント
│   ├── proposals/
│   ├── architecture/
│   ├── contributing.md
│   └── development.md
├── src/               # ソースコード
├── tests/             # テスト
├── scripts/           # ユーティリティスクリプト
├── out/               # 一時出力（gitignore）
├── archives/          # アーカイブ（gitignore）
├── Makefile          # タスクランナー
├── pyproject.toml    # プロジェクト設定
├── README.md         # メインドキュメント
├── CHANGELOG.md      # 変更履歴
├── LICENSE           # ライセンス
└── CODE_OF_CONDUCT.md
```
