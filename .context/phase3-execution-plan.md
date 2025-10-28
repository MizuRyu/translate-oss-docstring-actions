# Phase 3実行計画: テスト強化

## 🎯 目標
メインケースのカバレッジを90%以上にする

## 📊 現状カバレッジ（Phase 2完了時点）

全体: 72%

主要モジュール:
- ✅ `util.py`: 100%
- ✅ `cli.py`: 85%
- ⚠️ `replace.py`: 79%
- ⚠️ `log_utils.py`: 77%
- ⚠️ `translate.py`: 74%
- ⚠️ `extract.py`: 70%
- ❌ `llm.py`: 41% ← **最優先**

## 📝 タスク詳細

### Task 3.1: llm.py のテスト強化
**優先度**: High  
**現状**: 41% → **目標**: 90%  
**想定時間**: 2時間

#### 未カバー箇所
- LLM通信部分（translate_batch関数）
- フォールバック処理
- エラーハンドリング（RateLimitError, TimeoutError）
- リトライロジック

#### 追加テストケース
1. **正常系**:
   - [ ] translate_batch成功（モック）
   - [ ] バッチ翻訳が正しく並列実行される
   - [ ] システムプロンプトが正しく適用される

2. **フォールバック**:
   - [ ] Primary失敗 → Fallback成功
   - [ ] Primary Rate Limit → Fallback使用
   - [ ] フォールバックカウントが正しく記録される

3. **エラーケース**:
   - [ ] RateLimitError発生時の挙動
   - [ ] TimeoutError発生時の挙動
   - [ ] 両方失敗時のエラーハンドリング
   - [ ] 不正なレスポンス形式

### Task 3.2: extract.py のテスト強化
**優先度**: Medium  
**現状**: 70% → **目標**: 90%  
**想定時間**: 1時間

#### 未カバー箇所
- 複雑なAST構造の処理
- エッジケース（空のdocstring、特殊文字）
- エラーハンドリング

#### 追加テストケース
1. **正常系**:
   - [ ] クラスdocstring抽出
   - [ ] ネストした関数のdocstring
   - [ ] 複数行コメント

2. **エッジケース**:
   - [ ] 空のdocstring
   - [ ] 特殊文字を含むコメント
   - [ ] 非UTF-8エンコーディング（想定外）

### Task 3.3: translate.py のテスト強化
**優先度**: Medium  
**現状**: 74% → **目標**: 90%  
**想定時間**: 1.5時間

#### 未カバー箇所
- バッチ分割ロジック（トークン上限）
- 失敗アイテムの処理
- unprocessed.jsonl書き込み

#### 追加テストケース
1. **バッチ分割**:
   - [ ] トークン上限超過時の分割
   - [ ] 単一アイテムがMAX_TOKEN超過

2. **エラーケース**:
   - [ ] 翻訳結果件数不一致
   - [ ] 一部失敗時のunprocessed.jsonl書き込み

### Task 3.4: replace.py のテスト強化
**優先度**: Medium  
**現状**: 79% → **目標**: 90%  
**想定時間**: 1時間

#### 未カバー箇所
- 複雑なAST変換
- インデント処理
- エッジケース

#### 追加テストケース
1. **正常系**:
   - [ ] クラスdocstring置換
   - [ ] ネストした関数のdocstring置換

2. **エッジケース**:
   - [ ] インデントが複雑なコード
   - [ ] タブとスペース混在

### Task 3.5: log_utils.py のテスト強化
**優先度**: Low  
**現状**: 77% → **目標**: 90%  
**想定時間**: 30分

#### 追加テストケース
- [ ] log_progress: 各種パラメータパターン
- [ ] log_summary: 空の統計情報
- [ ] log_error: 詳細情報の表示

## 🔄 実行順序

1. **Task 3.1: llm.pyテスト強化** - 2時間
2. **Task 3.3: translate.pyテスト強化** - 1.5時間
3. **Task 3.2: extract.pyテスト強化** - 1時間
4. **Task 3.4: replace.pyテスト強化** - 1時間
5. **Task 3.5: log_utils.pyテスト強化** - 30分
6. **カバレッジ測定と確認** - 30分
7. **コミット** - 10分

**合計想定時間**: 約6.5時間

## 📋 実装方針

### モックの活用
```python
from unittest.mock import patch, AsyncMock

@patch("src.llm.ChatCompletionsClient")
def test_translate_batch_success(self, mock_client):
    """翻訳バッチが成功する"""
    mock_instance = AsyncMock()
    mock_instance.complete.return_value = MockResponse(...)
    mock_client.return_value = mock_instance
    
    # テスト実行
```

### カバレッジ測定用コマンド追加
```makefile
# Makefile (将来的に追加)
coverage:
    uv run coverage run -m unittest discover -s tests
    uv run coverage report -m
```

## ✅ 完了条件

- [ ] llm.py: 90%以上
- [ ] extract.py: 90%以上
- [ ] translate.py: 90%以上
- [ ] replace.py: 90%以上
- [ ] log_utils.py: 90%以上
- [ ] 全体カバレッジ: 85%以上
- [ ] 全テスト通過

## 📋 次のフェーズ

Phase 3完了後:
- Phase 4: コード品質向上（重複コード削減）
- Phase 5: ドキュメント最小限整備
