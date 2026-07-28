.PHONY: install test full-experiment doctor serve

install:
	python3.11 -m pip install uv
	uv sync --extra dev --extra inference --frozen

test:
	uv run python -m pytest

doctor:
	uv run bimer doctor --deployment configs/deployment-v2.json --artifact-root . --offline

serve:
	uv run bimer serve --deployment configs/deployment-v2.json --artifact-root .

full-experiment:
	MANIFEST="$(MANIFEST)" FEATURES="$(FEATURES)" OUTPUT="$(OUTPUT)" DEVICE="$(DEVICE)" ./scripts/run_full_suite.sh
