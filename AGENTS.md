# AIエージェント仕様書作成システム - コーディング規約

## 0. プロジェクト固有ルール
### ルール
- クラスはあまり使わないで、関数志向で実装すること。
- 無駄な早期リターンやnullチェックは行わないこと。コードが増えるので、最小限のコードとしてください。
- コメントやコミット、開発で用いる言語は基本的に指示がない箇所は日本語を使用すること。

---

以下は基本的なコーディング規約です。常に従ってください。

## 1. 構造・サイズの上限

### 1.1 関数・メソッド
- **関数長**: ≤ 40行（推奨: 20-30行）
- 早期return/guard節でネスト削減
- 150文字以上の重複コードを回避

```python
# 良い例
def process_requirements(requirements: Dict[str, Any]) -> ProcessedRequirements:
    """要件を処理して構造化データを返す"""
    if not requirements:
        return ProcessedRequirements.empty()
    
    # Guard節で早期return
    if not self._validate_requirements(requirements):
        raise InvalidRequirementsError("要件の検証に失敗しました")
    
    # メイン処理を小関数に分割
    parsed = self._parse_requirements(requirements)
    validated = self._validate_parsed_data(parsed)
    return self._structure_requirements(validated)
```

### 1.2 ファイル・クラス
- **ファイル長**: ≤ 400行
- **ネスト深さ**: ≤ 3
- **パラメータ数**: ≤ 5
- **クラス責務**: 単一責任原則（SRP）厳守

```python
# 悪い例（God Class）
class SpecificationManager:  # ❌ 複数責務
    def generate_scenario(self): ...
    def review_requirements(self): ...
    def save_to_database(self): ...
    def send_notification(self): ...

# 良い例（責務分離）
class ScenarioGenerator:  # ✅ 単一責務
    """シナリオ生成に特化"""
    def generate(self, overview: str) -> Scenario: ...

class RequirementsReviewer:  # ✅ 単一責務
    """要件レビューに特化"""
    def review(self, requirements: Requirements) -> ReviewResult: ...
```

## 2. 命名・可読性

### 2.1 命名規則
```python
# 関数名: 動詞+目的語
def parse_scenario(text: str) -> Scenario: ...
def validate_requirements(reqs: Requirements) -> bool: ...
def sync_agents(agent_list: List[Agent]) -> None: ...

# クラス名: 名詞（役割を表す）
class ScenarioGenerator: ...
class RequirementsValidator: ...
class AgentOrchestrator: ...

# 定数: 大文字スネークケース
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT_SECONDS = 120
AZURE_OPENAI_ENDPOINT = "https://..."

# 変数名: 意味が明確な名前
user_input = get_user_input()  # ✅
ui = get_user_input()  # ❌ 不明瞭
```

### 2.2 魔法数の禁止
```python
# 悪い例
if len(text) > 5000:  # ❌ 魔法数
    truncate_text(text)

# 良い例
MAX_TEXT_LENGTH = 5000  # ✅ 定数化

if len(text) > MAX_TEXT_LENGTH:
    truncate_text(text)
```

## 3. コメント＆ドキュメント（AI補完最適化）

### 3.1 関数・クラスのドキュメント
```python
class ReviewTeamChat(SelectorGroupChat):
    """
    レビューチーム用のグループチャット管理クラス
    
    複数のレビュアーエージェントを協調させ、
    要件や設計を多角的にレビューする
    """
    
    def run_review(self, requirements: str) -> ReviewResult:
        """
        要件レビューを実行する
        
        Args:
            requirements: レビュー対象の要件テキスト
            
        Returns:
            ReviewResult: 各観点からのレビュー結果
            
        Raises:
            ReviewTimeoutError: レビューがタイムアウトした場合
            AgentNotAvailableError: 必要なエージェントが利用不可の場合
        """
        # 実装
```

### 3.2 意図コメント
```python
def process_parallel_reviews(self, reviews: List[Review]) -> ConsolidatedReview:
    """並列レビュー結果を統合する"""
    
    # セキュリティレビューを優先的に処理
    # 理由: セキュリティ問題は他の観点より重要度が高いため
    security_issues = self._extract_security_issues(reviews)
    if security_issues.is_critical():
        return ConsolidatedReview.critical_failure(security_issues)
    
    # 通常の統合処理に進む
    return self._merge_reviews(reviews)
```

## 4. 例外・エラー設計

### ドメイン固有例外
```python
# ドメイン固有の例外定義
class SpecificationError(Exception):
    """仕様書作成システムの基底例外"""
    pass

class ScenarioGenerationError(SpecificationError):
    """シナリオ生成時のエラー"""
    pass

class ReviewTimeoutError(SpecificationError):
    """レビュータイムアウトエラー"""
    pass

# 使用例
def generate_scenario(overview: str) -> Scenario:
    """シナリオを生成する"""
    try:
        # 最小スコープでtry-except
        response = await agent.generate(overview)
    except OpenAIError as e:
        # ドメイン固有例外でラップ
        raise ScenarioGenerationError(f"シナリオ生成に失敗: {e}") from e
    
    return self._parse_scenario(response)
```

## 5. 依存・再利用

### 既存関数の合成
```python
# 既存関数を組み合わせて新機能を実現
def process_and_review(self, text: str) -> ReviewedContent:
    """処理とレビューを組み合わせた機能"""
    
    # 既存関数の合成
    parsed = self.parse_content(text)  # 既存
    validated = self.validate_content(parsed)  # 既存
    reviewed = self.review_content(validated)  # 既存
    
    return ReviewedContent(
        original=text,
        parsed=parsed,
        validation_result=validated,
        review_result=reviewed
    )
```

## 6. Git/PR運用

### 6.1 Author情報の付与
- commit時は以下の形式を原則とし、必ずAuthor情報(name/email)を明示的に設定してコマンド実行すること。  

  ```bash
  GIT_AUTHOR_NAME="codex" \
  GIT_AUTHOR_EMAIL="codex@noreply.github.com" \
  git commit -m "feat: 自動生成コードの更新"
  ```

### 6.2 コミットの分割
- 一度に変更を`git add .`でまとめようとしないでください。
- ファイル名を指定して`git add`するようにしてください。
- commitは適切な粒度で行ってください。複数に分割されることは許可されています。

### 6.3 コミットメッセージ
そのうえで以下のmsgにすること。
```bash
# フォーマット: type(scope): summary

feat(agents): シナリオ生成エージェントを追加
fix(review): レビュータイムアウトの修正
refactor(workflow): ワークフロークラスの責務分離
test(scenario): シナリオ生成の境界値テスト追加
docs(readme): インストール手順を更新
chore(deps): AutoGenを0.7.3にアップデート
```

## 7 生成AI利用時の注意
- userの指示が曖昧である場合、良い例のように解釈してください。

###  小さな単位での指示
```python
# 悪い例: 大きすぎる指示
"""
SpecificationWorkflowクラス全体を実装してください。
シナリオ生成、要件定義、レビュー、設計、すべての機能を含めてください。
"""

# 良い例: 関数単位での指示
"""
generate_scenario関数を実装してください。
入力: overview (str) - システム概要
出力: Scenario オブジェクト
制約: 
- 最大5000文字まで処理
- 空入力ではValidationError
- Azure OpenAI APIを使用
"""
```
