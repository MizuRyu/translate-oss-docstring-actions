# Phase 2実行計画: ログ統一とワークフロー改善

## 🎯 目標
1. ログフォーマットを統一し、進捗が一目で分かるようにする
2. ワークフロー承認画面で実行予定を明確にする
3. エラー時の状況を把握しやすくする

## 📝 タスク詳細

### Task 2.1: ログフォーマット統一
**優先度**: High  
**想定時間**: 1時間

#### 現状調査
各モジュールの現在のログ出力を確認:

```python
# extract.py
logger.info("[Extract] %d/%d %s", index, total_files, path)

# translate.py  
logger.info("[Translate] バッチ %d/%d (件数=%d)", index, len(batches), len(batch))

# replace.py
logger.info("[Replace] %d/%d %s", index, total_files, path)
```

#### 改善案
**共通ヘルパー関数を作成**:

```python
# src/log_utils.py (新規作成)
def log_progress(stage: str, current: int, total: int, item: str, detail: str = ""):
    """統一されたプログレス表示"""
    percentage = int(current / total * 100) if total > 0 else 0
    msg = f"[{stage}] {current}/{total} ({percentage}%) {item}"
    if detail:
        msg += f" - {detail}"
    logger.info(msg)

def log_summary(stage: str, stats: dict):
    """統一されたサマリー表示"""
    logger.info("=" * 60)
    logger.info(f"[{stage}] Complete")
    logger.info("=" * 60)
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
```

#### 適用箇所
- [ ] `src/extract.py`: log_progress, log_summary使用
- [ ] `src/translate.py`: log_progress, log_summary使用
- [ ] `src/replace.py`: log_progress, log_summary使用

### Task 2.2: ワークフロー承認画面改善
**優先度**: High  
**想定時間**: 1時間

#### 現状
```yaml
cat >> "$GITHUB_STEP_SUMMARY" << EOF
## 📊 トークン使用量サマリ

| 項目 | 値 |
|------|------|
| 抽出件数 | **${ITEMS}件** |
| 総トークン数 | **${TOKENS}** |
| 平均トークン数 | **${AVG_TOKENS}** |
EOF
```

#### 改善案
1. **scripts/summarize_tokens.py を拡張**
   - `translation_limit` を考慮
   - 実際に翻訳される件数を計算
   - 推定処理時間を計算

2. **ワークフローで詳細表示**
   ```yaml
   ## 📊 翻訳実行サマリ
   
   ### 抽出結果
   - 総ファイル数: 150件
   - 抽出件数: 450件
   - 総トークン数: 12,500
   
   ### 翻訳実行予定
   - 翻訳件数: 10件（--limit 10指定）
   - 対象トークン数: 278トークン
   - 推定バッチ数: 2バッチ
   - 推定処理時間: 約30秒
   
   ### 設定
   - Mock Mode: false
   - Exclude Terms: Agent, Thread, Client
   ```

#### 実装手順
1. `scripts/summarize_tokens.py` 修正
   - `--limit` オプション追加
   - 推定処理時間計算（バッチ数 × 平均処理時間）
   - 出力JSONに追加情報
2. `.github/workflows/translate.yml` 修正
   - prepare jobで `--limit` を渡す
   - Step Summaryのフォーマット改善

### Task 2.3: エラーハンドリング確認
**優先度**: Medium  
**想定時間**: 30分

#### 確認項目
- [ ] `translate.py`が途中で失敗した場合
  - どこまで処理したかログに出る？
  - `unprocessed.jsonl`に失敗分が記録される？
- [ ] `replace.py`が途中で失敗した場合
  - どこまで処理したかログに出る？
  - 部分的に生成されたファイルの扱いは？

#### 改善
- 各モジュールで処理完了時に必ずサマリーを出力
- 失敗時もどこまで処理したかログ出力

### Task 2.4: .gitignore整理
**優先度**: Low  
**想定時間**: 5分

#### 不要な項目削除
- `archive/*` → 削除（archives/に統一）
- 整理して見やすく

## 🔄 実行順序

1. **現状調査** (Task 2.3) - 10分
2. **log_utils.py作成** (Task 2.1) - 20分
3. **各モジュールへの適用** (Task 2.1) - 30分
4. **summarize_tokens.py拡張** (Task 2.2) - 30分
5. **ワークフロー修正** (Task 2.2) - 30分
6. **.gitignore整理** (Task 2.4) - 5分
7. **動作確認** - 20分
8. **コミット** - 5分

**合計想定時間**: 約2.5時間

## ✅ 完了条件

- [ ] すべてのログが統一されたフォーマットで出力される
- [ ] プログレスに%表示が追加される
- [ ] ワークフロー承認画面に実行予定情報が表示される
- [ ] エラー時に進捗状況が分かる
- [ ] .gitignoreが整理されている
- [ ] テストが通る

## 📋 次のフェーズ

Phase 2完了後:
- Phase 3: コード品質向上（重複コード削減）
- Phase 4: テストの充実
- Phase 5: ドキュメント最小限整備
