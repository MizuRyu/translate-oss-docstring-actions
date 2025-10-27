# リポジトリ整備計画 (Refactoring & Cleanup Plan)

**目標**: OSSプロジェクトとして美しく、メンテナンス可能な状態にする

## 📋 Todo List

### 🔧 Phase 1: Makefile整備
- [ ] `make extract` - 抽出処理を実行
- [ ] `make translate` - 翻訳処理を実行
- [ ] `make replace` - 反映処理を実行
- [ ] `make pipeline` - 全パイプライン実行（extract → translate → replace）
- [ ] `make test` - 全テスト実行
- [ ] `make test-extract` - extract単体テスト
- [ ] `make test-translate` - translate単体テスト
- [ ] `make test-replace` - replace単体テスト
- [ ] `make test-pipeline` - パイプライン統合テスト
- [ ] `make clean` - 生成ファイル削除
- [ ] `make act-test` - actでローカルワークフローテスト（translate-testをリネーム）
- [ ] 各コマンドで生成されたファイルをアーカイブディレクトリに保存

### 🧪 Phase 2: テストメンテナンス
- [ ] `tests/test_extract.py` - 最新コードに合わせて修正
- [ ] `tests/test_translate.py` - 最新コードに合わせて修正
- [ ] `tests/test_replace.py` - 最新コードに合わせて修正
- [ ] `tests/test_pipeline.py` - 最新コードに合わせて修正
- [ ] テストデータの整理
- [ ] カバレッジ計測の追加

### 📝 Phase 3: ログ出力改善
- [ ] `run.log` ファイル生成を削除
- [ ] ログはstdoutのみ、またはユーザー指定のパスに出力
- [ ] ログレベルのデフォルト値見直し
- [ ] 各モジュールのロガー設定統一

### 📁 Phase 4: 成果物管理
- [ ] `out/` ディレクトリ構造の明確化
- [ ] `archives/` または `runs/` にタイムスタンプ付きで保存
- [ ] `.gitignore` の更新（out/, archives/, translated/ など）

### 📚 Phase 5: ドキュメント整備
- [ ] `docs/` ディレクトリ作成
- [ ] `docs/proposals/` - 機能提案
- [ ] `docs/architecture/` - アーキテクチャ設計
- [ ] `docs/contributing.md` - コントリビューションガイド
- [ ] `docs/development.md` - 開発者向けガイド
- [ ] `CHANGELOG.md` - 変更履歴
- [ ] `CODE_OF_CONDUCT.md` - 行動規範

### 🎨 Phase 6: OSSとしてのブラッシュアップ
- [ ] LICENSE確認（既存あり）
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
