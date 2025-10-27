.PHONY: translate-test

translate-test:
	mkdir -p translated
	act workflow_dispatch \
	  -W .github/workflows/translate-test.yml \
	  --container-architecture linux/amd64 \
	  --bind \
	  --input repository_url=https://github.com/microsoft/agent-framework.git \
	  --input subdirectory=python \
	  --input max_records=5 \
	  --input mock_mode=true \
	  --input artifact_dir=translated
