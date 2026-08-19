# AI Narrative Video Director

An AI-powered backend system that analyzes a synchronized multicam podcast recording
("SyncMaster"), understands the conversation, selects appropriate camera angles, and produces a
production-ready Adobe Premiere Pro FCPXML v1.10 rough cut automatically.

> Deterministic editorial logic + AI only where genuine judgment is needed (vision
> classification, narrative understanding). See `ARCHITECTURE.md` for the full stage design and
> `SCALING.md` for how this runs at ~500 episodes/month.

---

## Table of Contents

- [Quick Run (TL;DR)](#quick-run-tldr)
- [Requirements](#requirements)
- [Setup Guide](#setup-guide)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Running Individual Stages](#running-individual-stages)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Live Demo Scripts](#live-demo-scripts)

---

## Quick Run (TL;DR)

```bash
# 1. Clone and enter the project
git clone <repo-url> ai-narrative-director
cd ai-narrative-director

# 2. Create venv and install
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Set your Groq API key
cp .env.example .env
echo "GROQ_API_KEY=your_key_here" >> .env

# 4. Run the full test suite (confirms everything works)
make test

# 5. Run the pipeline on your files
python -m director.run \
  --input path/to/SyncMaster.mp4 \
  --show-type path/to/show_type.txt \
  --output-dir ./output

# 6. Check results
ls ./output
# output.fcpxml  editing_report.json  timeline.html  needs_review.json
```

That's it — `output.fcpxml` is ready to import into Premiere Pro.

---

## Requirements

- **Python 3.11+**
- **FFmpeg** installed and on your `PATH`
- **Groq API key** (free — sign up at [console.groq.com](https://console.groq.com))
- ~2 GB free disk space for model weights + cache (faster-whisper, Resemblyzer)
- No GPU required (runs on CPU; GPU is an optional speed upgrade — see `SCALING.md`)
- No local LLM hosting required — all LLM reasoning runs via the Groq API

---

## Setup Guide

### 1. Install system dependencies

**macOS**
```bash
brew install ffmpeg tesseract
```

**Ubuntu / Debian**
```bash
sudo apt update
sudo apt install -y ffmpeg tesseract-ocr libsndfile1
```

**Windows**
- Install [FFmpeg](https://ffmpeg.org/download.html) and add it to `PATH`.
- Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and add it to `PATH`.

### 2. Create a Python virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `faster-whisper`, `resemblyzer`, `scikit-learn` (clustering),
`opencv-python`, `pytesseract`, `imagehash`, `groq`, `pydantic`, `lxml`, `pyyaml`, `structlog`,
`pytest`, `ruff`, `mypy`.

> **Note on speaker diarization**: this project uses **Resemblyzer embeddings + clustering**
> (fully open, no license gate) as the default, with face-activity detection from Stage 1 as a
> cross-check for "current speaker." `pyannote.audio` is supported as an optional swap-in if you
> have a Hugging Face token and have accepted its model license — see `docs/DIARIZATION.md`.

### 4. Get a Groq API key

1. Sign up free at [console.groq.com](https://console.groq.com).
2. Create an API key.
3. Copy `.env.example` to `.env` and add it:
   ```bash
   cp .env.example .env
   ```
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```

### 5. Verify installation

```bash
python -m director.doctor
```

This runs a quick self-check: confirms FFmpeg/Tesseract are on `PATH`, confirms the Groq key is
valid with a minimal test call, and confirms all Python dependencies import cleanly. Fix
anything it flags before proceeding.

---

## Configuration

All editorial thresholds live in `editorial_rules.yaml` — **never hardcoded in code**. Key
sections you'll likely touch:

```yaml
reaction_shot_duration_s: [3, 5]
refresh_interval_s: 45
refresh_wide_duration_s: 3
wide_shot_max_pct:
  nav_thethi_show: 0.20
  cracking_maturity_code: null   # no cap; SBS-driven instead

physical_adjustment_triggers:
  - mic_adjust
  - face_scratch
  - lip_lick
  - posture_change
  - shoe_exposure
  - off_screen_glance_gt_3s

groq:
  budget_per_run: 50
  model_text: "llama-3.3-70b-versatile"     # confirm current free-tier model name
  model_vision: "llama-3.2-90b-vision"      # optional, confirm availability
  retry:
    max_attempts: 3
    base_backoff_s: 2
    max_backoff_s: 30

self_correction:
  max_iterations: 3

human_review:
  confidence_threshold: 0.6
```

Change a value, save, and re-run — no code changes or redeploy needed (see
[Running Individual Stages](#running-individual-stages) to re-run only the affected stages).

---

## Running the Pipeline

### Full run

```bash
python -m director.run \
  --input path/to/SyncMaster.mp4 \
  --show-type path/to/show_type.txt \
  --output-dir ./output \
  --config editorial_rules.yaml
```

**Flags:**
| Flag | Description |
|---|---|
| `--input` | Path to the SyncMaster multicam recording |
| `--show-type` | Path to `show_type.txt` (`The Nav Thethi Show` or `Cracking the Maturity Code`) |
| `--output-dir` | Where outputs are written |
| `--config` | Path to rules YAML (default: `editorial_rules.yaml`) |
| `--force` | Emit XML even if validation fails (with warnings in report) |
| `--no-cache` | Ignore cached intermediate JSON, force full re-run |
| `--groq-budget` | Override `budget_per_run` for this run |

### What happens during a run

1. Ingest → camera discovery → speaker mapping → narrative understanding (Groq calls happen
   here, cached to disk)
2. Director generates cuts → Critic checks them → self-correction loop if needed
3. XML Generator produces `output.fcpxml`
4. Validator checks it; refuses to write a broken file unless `--force`
5. `editing_report.json` and `timeline.html` are written alongside it

A ~60-minute episode typically takes 5–20 minutes on CPU, depending on transcription length.

---

## Running Individual Stages

Because every stage reads/writes cached JSON, you can re-run just one part — this is the fast
path for rule tuning and debugging (see `ARCHITECTURE.md` for the full stage list).

```bash
# Re-run only the Director + XML Generator after a rules.yaml change
python -m director.stage director \
  --narrative ./output/narrative.json \
  --cameras ./output/camera_inventory.json \
  --config editorial_rules.yaml \
  --out ./output/cut_list.json

python -m director.stage xmlgen \
  --cuts ./output/cut_list.json \
  --out ./output/output.fcpxml

python -m director.stage validate \
  --xml ./output/output.fcpxml \
  --cameras ./output/camera_inventory.json
```

This makes zero Groq API calls — only Stages 4–6 touch code, no AI involved.

---

## Testing

```bash
# Everything: lint, types, unit, integration, rule regression, fault injection, XML validation
make test
```

Equivalent to:
```bash
ruff check .
mypy .
pytest -m "not integration"        # fast unit + rule regression tests
pytest -m integration               # full pipeline test on tiny sample media
pytest -m fault_injection           # simulated failures (missing camera, Groq errors, etc.)
python scripts/validate_xml.py output/output.fcpxml   # standalone XML check
```

Most tests run against small hand-built JSON fixtures and **do not require real video** — see
`tests/fixtures/` and `docs/TESTING.md` for the full breakdown of what's mocked vs. what needs
real (tiny, synthetic) media.

`make test` must show all categories passing before you consider a change done.

---

## Project Structure

```
ai-narrative-director/
├── director/
│   ├── stages/
│   │   ├── ingest.py
│   │   ├── camera_discovery.py
│   │   ├── speaker_mapping.py
│   │   ├── narrative_understanding.py
│   │   ├── director.py          # Editorial Decision Engine
│   │   ├── critic.py
│   │   ├── xml_generator.py
│   │   └── validator.py
│   ├── schemas.py                # Pydantic models (inter-stage contracts)
│   ├── groq_client.py            # cached, retried, budgeted Groq wrapper
│   ├── run.py                    # full pipeline CLI
│   ├── stage.py                  # single-stage CLI
│   └── doctor.py                 # environment self-check
├── editorial_rules.yaml
├── scripts/
│   ├── validate_xml.py
│   ├── inject_missing_camera.py
│   ├── inject_frozen_frame.py
│   └── inject_groq_failure.py
├── tests/
│   ├── fixtures/
│   ├── test_rules_*.py
│   ├── test_fault_injection.py
│   └── test_integration.py
├── ARCHITECTURE.md
├── SCALING.md
├── docs/
│   ├── TESTING.md
│   └── DIARIZATION.md
├── editorial_rules.yaml
├── requirements.txt
├── Makefile
├── .env.example
└── README.md
```

---

## Outputs

| File | Description |
|---|---|
| `output.fcpxml` | The multicam rough cut — import directly into Premiere Pro |
| `editing_report.json` | Camera inventory, speaker mapping, cuts, warnings, off-camera segments, metadata (quality score, Groq usage, cache hit rate) |
| `timeline.html` | Optional visual timeline of cuts and camera usage % |
| `needs_review.json` | Low-confidence cuts flagged for human approval (only if any exist) |

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `GROQ_API_KEY not set` | Check `.env` exists and is loaded; run `python -m director.doctor` |
| Groq `429` errors during run | Free-tier rate limit hit — the pipeline retries with backoff automatically; if persistent, lower `groq.budget_per_run` or check `groq_budget` usage in `editing_report.json` |
| `output.fcpxml` fails to import into Premiere | Run `python scripts/validate_xml.py output/output.fcpxml` for an itemized error list before opening a support issue |
| Camera roles misdetected | Confirm `show_type.txt` content matches expected values exactly; check `camera_inventory.json` confidence scores in the report |
| Diarization seems inaccurate | Default (Resemblyzer) is lighter-weight than pyannote; see `docs/DIARIZATION.md` for swapping in pyannote if you have an HF token |
| Pipeline is slow | Expected on CPU for long episodes; see `SCALING.md` for the optional GPU whisper upgrade |

---

## Live Demo Scripts

For walking through debugging/robustness live (see `ARCHITECTURE.md` for the full rationale):

```bash
python scripts/inject_missing_camera.py     # simulates a dropped camera feed
python scripts/inject_frozen_frame.py       # simulates a frozen camera
python scripts/inject_groq_failure.py       # simulates rate-limit/timeout/malformed JSON
python scripts/validate_xml.py output/output.fcpxml   # standalone XML validation, no Premiere needed
```

Each prints a before/after showing detection → graceful fallback → warning logged, with no
crash and no invalid XML produced.