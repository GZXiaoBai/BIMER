.PHONY: install test full-experiment

install:
	python3.11 -m pip install -e '.[dev,inference]'

test:
	python -m pytest

full-experiment:
	MANIFEST="$(MANIFEST)" FEATURES="$(FEATURES)" OUTPUT="$(OUTPUT)" DEVICE="$(DEVICE)" ./scripts/run_full_suite.sh

