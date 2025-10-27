# Phase 1実行計画: Makefile整備とログ改善

## 🎯 目標
1. Makefileを充実させ、開発体験を向上
2. run.log自動生成を削除
3. 成果物を適切に管理

## 📝 タスク詳細

### Task 1.1: Makefileコマンド追加
**優先度**: High  
**想定時間**: 30分

#### 追加するコマンド
```makefile
# 基本コマンド
make extract        # docstring/コメント抽出
make translate      # 翻訳実行
make replace        # ソースへ反映
make pipeline       # 全パイプライン実行

# テストコマンド
make test           # 全テスト実行
make test-extract   # extract単体テスト
make test-translate # translate単体テスト
make test-replace   # replace単体テスト
make test-pipeline  # パイプライン統合テスト

# ユーティリティ
make clean          # 生成ファイル削除
make act-local      # actでローカルテスト（translate-testからリネーム）
make archive        # 成果物をアーカイブ
make help           # ヘルプ表示
```

#### 実装方針
- 各コマンドは明確なデフォルト値を持つ
- 環境変数でカスタマイズ可能に
- 成果物は `out/` に出力、完了後に `archives/$(date +%Y%m%d_%H%M%S)/` へコピー

### Task 1.2: run.log生成の削除
**優先度**: High  
**想定時間**: 15分

#### 調査
- [ ] `run.log` がどこで生成されているか確認
- [ ] ログ設定ファイル/コードを特定

#### 修正
- [ ] ファイルハンドラーを削除
- [ ] stdout/stderrのみに出力
- [ ] 必要であれば `--log-file` オプションでユーザーが指定可能に

### Task 1.3: 成果物管理
**優先度**: High  
**想定時間**: 20分

#### ディレクトリ構造
```
project/
├── out/              # 一時出力（gitignore）
│   ├── extracted.jsonl
│   ├── translated.jsonl
│   ├── unprocessed.jsonl
│   ├── token-summary.json
│   └── translated_sources/
├── archives/         # アーカイブ（gitignore）
│   └── 20251027_123456/  # タイムスタンプ
│       ├── extracted.jsonl
│       ├── translated.jsonl
│       ├── unprocessed.jsonl
│       ├── token-summary.json
│       └── translated_sources/
└── translated/      # GitHub Actions用（gitignore）
```

#### .gitignore更新
```
out/
archives/
translated/
*.log
.uv-cache/
__pycache__/
```

### Task 1.4: テストメンテナンス
**優先度**: Medium  
**想定時間**: 1時間

#### チェック項目
- [ ] `tests/test_extract.py` が動作するか
- [ ] `tests/test_translate.py` が動作するか
- [ ] `tests/test_replace.py` が動作するか
- [ ] `tests/test_pipeline.py` が動作するか
- [ ] 最新のCLI引数に対応しているか（batch_size, mode削除）
- [ ] アサーションが適切か

## 🔄 実行順序

1. **run.log調査** (Task 1.2) - 5分
2. **Makefile作成** (Task 1.1) - 30分
3. **run.log削除** (Task 1.2) - 10分
4. **.gitignore更新** (Task 1.3) - 5分
5. **make test実行して確認** (Task 1.4) - 10分
6. **テスト修正** (Task 1.4) - 50分
7. **動作確認** - 10分
8. **コミット** - 5分

**合計想定時間**: 約2時間

## ✅ 完了条件

- [ ] すべてのmakeコマンドが動作する
- [ ] `run.log` が生成されない
- [ ] `make test` が通る
- [ ] `make pipeline` で全フロー実行できる
- [ ] 成果物が適切に保存される
- [ ] .gitignoreが更新されている
- [ ] READMEに新しいmakeコマンドが記載されている

## 📋 次のフェーズ

Phase 1完了後:
- Phase 2: ドキュメント整備
- Phase 3: CI/CD追加
- Phase 4: OSSブラッシュアップ
