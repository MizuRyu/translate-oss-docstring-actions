.PHONY: help extract translate replace pipeline clean test test-extract test-translate test-replace test-pipeline act-local archive

# デフォルト設定
TARGET_REPO ?= target-repo/python
OUT_DIR ?= out
ARCHIVE_DIR ?= archives

# ヘルプ
help:
	@echo "📚 Available commands:"
	@echo ""
	@echo "  🔧 Pipeline commands:"
	@echo "    make extract       - Extract docstrings and comments"
	@echo "    make translate     - Translate extracted content"
	@echo "    make replace       - Apply translations to source"
	@echo "    make pipeline      - Run full pipeline (extract → translate → replace)"
	@echo ""
	@echo "  🧪 Test commands:"
	@echo "    make test          - Run all tests"
	@echo "    make test-extract  - Run extract tests only"
	@echo "    make test-translate - Run translate tests only"
	@echo "    make test-replace  - Run replace tests only"
	@echo "    make test-pipeline - Run pipeline tests only"
	@echo ""
	@echo "  🛠️  Utility commands:"
	@echo "    make clean         - Remove generated files"
	@echo "    make archive       - Archive current outputs with timestamp"
	@echo "    make act-local     - Run workflow locally with act"
	@echo ""
	@echo "  📝 Environment variables:"
	@echo "    TARGET_REPO        - Target directory (default: target-repo/python)"
	@echo "    OUT_DIR            - Output directory (default: out)"
	@echo "    ARCHIVE_DIR        - Archive directory (default: archives)"

# 抽出
extract:
	@echo "🔍 Extracting docstrings and comments..."
	mkdir -p $(OUT_DIR)
	uv run python main.py extract $(TARGET_REPO) \
		--output $(OUT_DIR)/extracted.jsonl \
		--log-level INFO
	@echo "✅ Extraction complete: $(OUT_DIR)/extracted.jsonl"

# 翻訳
translate:
	@echo "🌐 Translating content..."
	mkdir -p $(OUT_DIR)
	uv run python main.py translate $(OUT_DIR)/extracted.jsonl \
		--output $(OUT_DIR)/translated.jsonl \
		--failed-output $(OUT_DIR)/unprocessed.jsonl \
		--mock \
		--log-level INFO
	@echo "✅ Translation complete: $(OUT_DIR)/translated.jsonl"

# 反映
replace:
	@echo "📝 Applying translations..."
	mkdir -p $(OUT_DIR)
	uv run python main.py replace $(OUT_DIR)/translated.jsonl \
		--output-dir $(OUT_DIR)/translated_sources \
		--root $(TARGET_REPO) \
		--log-level INFO
	@echo "✅ Replace complete: $(OUT_DIR)/translated_sources/"

# 全パイプライン
pipeline: extract translate replace
	@echo ""
	@echo "🎉 Pipeline complete!"
	@echo "  📁 Outputs in: $(OUT_DIR)/"

# クリーンアップ
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf $(OUT_DIR)
	rm -rf translated
	rm -rf target-repo
	rm -f *.log
	@echo "✅ Clean complete"

# アーカイブ
archive:
	@echo "📦 Archiving outputs..."
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	DEST_DIR="$(ARCHIVE_DIR)/$$TIMESTAMP"; \
	mkdir -p "$$DEST_DIR"; \
	if [ -d "$(OUT_DIR)" ]; then \
		cp -r $(OUT_DIR)/* "$$DEST_DIR/" 2>/dev/null || true; \
		echo "✅ Archived to: $$DEST_DIR"; \
	else \
		echo "⚠️  No outputs to archive ($(OUT_DIR) not found)"; \
	fi

# テスト実行
test:
	@echo "🧪 Running all tests..."
	uv run python -m unittest discover -s tests -p "test_*.py" -v

test-extract:
	@echo "🧪 Running extract tests..."
	uv run python -m unittest tests.test_extract -v

test-translate:
	@echo "🧪 Running translate tests..."
	uv run python -m unittest tests.test_translate -v

test-replace:
	@echo "🧪 Running replace tests..."
	uv run python -m unittest tests.test_replace -v

test-pipeline:
	@echo "🧪 Running pipeline tests..."
	uv run python -m unittest tests.test_pipeline -v

# actでローカルワークフローテスト
act-local:
	@echo "🎬 Running workflow locally with act..."
	mkdir -p translated
	act workflow_dispatch \
		-W .github/workflows/translate-test-local.yml \
		--container-architecture linux/amd64 \
		--bind \
		--input repository_url=https://github.com/microsoft/agent-framework.git \
		--input subdirectory=python \
		--input max_records=5 \
		--input mock_mode=true \
		--input artifact_dir=translated
	@echo "✅ Act execution complete. Check translated/ directory"
