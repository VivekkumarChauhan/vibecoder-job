# AI Narrative Video Director — Makefile
# All commands assume Python 3.11+ and pip-installed dependencies.

PYTHON = python
PYTEST = $(PYTHON) -m pytest
RUFF = $(PYTHON) -m ruff
MYPY = $(PYTHON) -m mypy

.PHONY: all install lint test test-unit test-rules test-fault test-groq test-integration \
        test-xml run demo demo-missing-camera demo-frozen-frame demo-audio-drift \
        demo-corrupt-showtype validate-xml timeline clean

all: lint test

install:
	pip install -r requirements.txt

# ─── Linting ─────────────────────────────────────────────────────────────────
lint:
	$(RUFF) check . --fix
	$(MYPY) pipeline/ utils/ main.py

# ─── Test suites ─────────────────────────────────────────────────────────────
test:
	$(PYTEST) -m "unit or rules or fault or groq or integration" --cov=pipeline --cov=utils \
	          --cov-report=term-missing -q

test-unit:
	$(PYTEST) -m unit -q

test-rules:
	$(PYTEST) -m rules -v

test-fault:
	$(PYTEST) -m fault -v

test-groq:
	$(PYTEST) -m groq -v

test-integration:
	$(PYTEST) -m integration -v

test-xml:
	$(PYTEST) tests/test_stage6_validator.py -v

# ─── Pipeline run ────────────────────────────────────────────────────────────
run:
	@echo "Usage: python main.py --input <SyncMaster.mp4> --show-type show_type.txt"
	$(PYTHON) main.py --input sample/SyncMaster.mp4 --show-type sample/show_type.txt

# ─── Standalone demos ────────────────────────────────────────────────────────
demo-missing-camera:
	$(PYTHON) fault_injection/inject_missing_camera.py

demo-frozen-frame:
	$(PYTHON) fault_injection/inject_frozen_frame.py

demo-audio-drift:
	$(PYTHON) fault_injection/inject_audio_drift.py

demo-corrupt-showtype:
	$(PYTHON) fault_injection/inject_corrupt_showtype.py

validate-xml:
	$(PYTHON) validate_xml.py output.fcpxml

timeline:
	$(PYTHON) generate_timeline_html.py

# ─── Clean ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
