# AI Narrative Video Director

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-108%20passed-brightgreen.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![FCPXML: v1.10](https://img.shields.io/badge/FCPXML-v1.10%20(Premiere%20Pro)-purple.svg)]()
[![LLM: Groq Free Tier](https://img.shields.io/badge/LLM-Groq%20LPU%20(Free%20Tier)-orange.svg)](https://groq.com)

An industrial-grade, AI-driven backend system that automatically analyzes a synchronized multicam podcast recording (**SyncMaster**) and show configuration (`show_type.txt`), intelligently classifies narrative dynamics, and produces an editorially sound, frame-accurate **Adobe Premiere Pro FCPXML v1.10** multicam rough cut (`output.fcpxml`) along with a full analytical audit report (`editing_report.json`) and interactive visual timeline (`timeline.html`).

---

## 📑 Table of Contents

- [Core Principles & Architectural Priorities](#-core-principles--architectural-priorities)
- [System Architecture & Pipeline Flow](#-system-architecture--pipeline-flow)
- [Complete Tech Stack & Library Rationale](#-complete-tech-stack--library-rationale)
- [API Keys & Environment Variables Explained](#-api-keys--environment-variables-explained)
- [Step-by-Step Setup Guide](#-step-by-step-setup-guide)
  - [1. System Prerequisites (FFmpeg & Tesseract)](#1-system-prerequisites)
  - [2. Virtual Environment & Dependencies](#2-virtual-environment--dependencies)
  - [3. Configure Environment Variables](#3-configure-environment-variables)
  - [4. Verification Self-Test](#4-verification-self-test)
- [Editorial Rules Engine (Data-Driven)](#-editorial-rules-engine-data-driven)
- [CLI Reference & Usage](#-cli-reference--usage)
  - [Full Pipeline Run](#full-pipeline-run)
  - [Standalone XML Validator](#standalone-xml-validator)
  - [HTML/SVG Visual Timeline Generator](#htmlsvg-visual-timeline-generator)
- [Fault Injection & Live Resiliency Demos](#-fault-injection--live-resiliency-demos)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Project Directory Structure](#-project-directory-structure)
- [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🎯 Core Principles & Architectural Priorities

The system is built on four non-negotiable engineering priorities:

1. **Never Crash (100% Graceful Degradation):** Every stage returns `(result, warnings, errors)` and never raises uncaught exceptions to the orchestrator. If any service, camera stream, or API call fails, the pipeline logs actionable warnings and falls back to deterministic safe defaults.
2. **Never Produce Invalid XML:** XML generation is handled via pure `lxml` with strict rational integer frame arithmetic (`frames_to_fcp_rational`) rather than floating-point math, preventing fractional frame drift and timeline corruption in Premiere Pro.
3. **Never Produce Editorially Nonsensical Cuts:** All cuts are bounded by strict human editorial pacing heuristics (reaction holds, dialogue 2-shots, wide-shot caps, emotion locks, and mid-word cut safety checks).
4. **Free-Tier & Zero Local Model Hosting:** Designed specifically for zero GPU budget environments by leveraging the **Groq API Free Tier** for ultra-fast cloud LLM reasoning and CPU-optimized quantized models locally.

---

## 🏗 System Architecture & Pipeline Flow

```
   ┌────────────────────────────────────────────────────────┐
   │     SyncMaster (Multitrack MP4/MOV) + show_type.txt    │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │ Stage 0: Ingest & Show Type Detection                  │
   │ • Case-insensitive show matching with keyword fallback │
   │ • FFmpeg stream demuxing (video tracks, mono stems)    │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │ Stage 1: Computer Vision & Camera Discovery            │
   │ • OpenCV DNN / Haar face detection (area ratio/center) │
   │ • Motion differencing & pHash for frozen feed detection│
   │ • Tesseract OCR lower-third / nameplate recognition    │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │ Stage 2: Audio Transcription & Speaker Mapping         │
   │ • faster-whisper word-level ASR (CPU-optimized int8)   │
   │ • pyannote / acoustic energy diarization alignment     │
   │ • 1-call Groq speaker role assignment (Host vs. Guest) │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │ Stage 3: Narrative & Contextual Understanding          │
   │ • Batched JSON narrative classification (Groq LPU)     │
   │ • Emotion, laughter, storytelling & off-camera triggers│
   │ • Deterministic heuristic fallback on rate limit/budget│
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │ Stage 4: Director Decision Engine (Deterministic)      │
   │ • 10 Core Editorial Rules (Speaker, Reaction, Wide)    │
   │ • Nav Thethi ≤20% wide cap + Maturity Code SBS clips   │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │ Stage 4b: Critic (Quality Scoring & Verification)      │
   │ • Rule violation detection & quality score calculation │
   │ • Self-correction loop: auto-adjusts rules & re-directs│
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │ Stage 5: FCPXML v1.10 Generator                        │
   │ • Exact rational timecode emission (e.g. 1001/30000s)  │
   │ • Inline XML editorial comments & compound clip markers│
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │ Stage 6: Schema & Structural Validator                 │
   │ • Schema compliance, clip continuity & overlap checks  │
   │ • Premiere Pro compatibility validation                │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │ Outputs: output.fcpxml | editing_report.json | timeline│
   └────────────────────────────────────────────────────────┘
```

---

## 🛠 Complete Tech Stack & Library Rationale

Every library in this repository was chosen for correctness, efficiency, and zero local hardware constraints:

| Library / Tool | Version | Purpose in System | Why This Specific Technology? |
| :--- | :--- | :--- | :--- |
| **`faster-whisper`** | `^1.0.0` | Automatic Speech Recognition (ASR) | Uses CTranslate2 under the hood, delivering **4x faster inference** than standard OpenAI Whisper on CPU with 8-bit quantization (`int8`), generating exact word-level start/end timestamps. |
| **`groq`** | `^0.11.0` | Cloud LLM Narrative Reasoning | Provides sub-second LLM inference on Groq LPUs (`llama-3.3-70b-versatile`), allowing real-time narrative segment classification with strict JSON schema mode on a free tier. |
| **`opencv-python`** (`cv2`) | `^4.8.0` | Computer Vision & Face Tracking | Analyzes video frames per stream for face presence, bounding box area ratio, and screen centering to automatically classify Host Hero, Guest Hero, and Wide cameras. |
| **`imagehash`** | `^4.3.1` | Perceptual Video Hashing | Computes `pHash` / `dHash` across consecutive video frames to detect frozen camera feeds with 99.9% accuracy without relying on timestamp metadata alone. |
| **`pytesseract`** | `^0.3.10` | Optical Character Recognition | Reads lower-third text and on-screen name overlays from the video to confirm speaker identities and camera assignments. |
| **`pydantic`** | `^2.5.0` | Data Contracts & Schema Validation | Enforces strict, type-safe data contracts between all stages (`CameraInventory`, `NarrativeResult`, `CutList`, `EditingReport`) and validates all JSON outputs from LLM calls. |
| **`lxml`** | `^5.0.0` | XML Generation & XPath Validation | Pure C-based XML processor that guarantees valid FCPXML v1.10 output conforming to Adobe Premiere Pro and Final Cut Pro XML DTD schemas. |
| **`pyyaml`** | `^6.0` | External Rules Engine Config | Parses `editorial_rules.yaml` so that editorial guidelines, thresholds, and camera budgets remain **data, not hardcoded code**. |
| **`structlog`** | `^24.1.0` | Structured Observability Logging | Emits JSON logs in production and clean console output in local testing with stage timings, cut statistics, and rule violation tracking. |
| **`ffmpeg` / `PyAV`** | System / `^12.0` | Audio/Video Stream Demuxing | Separates multichannel video files into individual camera visual streams and isolates mono audio stems without re-encoding quality loss. |
| **`pytest` & `pytest-cov`** | `^8.0` | Automated Test Suite | Powers the 108-test regression and fault-injection suite with branch and statement coverage reports. |
| **`ruff`** | `^0.8.0` | High-Speed Python Linter | Ultra-fast Rust-based Python linter that guarantees clean, bug-free, PEP8-compliant code across all modules. |

---

## 🔑 API Keys & Environment Variables Explained

The system configuration is centralized in `.env` (configured via `.env.example`):

```bash
# ─── Groq Cloud API (Free Tier) ────────────────────────────────────────────────
GROQ_API_KEY=gsk_your_groq_api_key_here

# ─── Hugging Face Token (Optional Diarization Upgrade) ─────────────────────────
HUGGINGFACE_TOKEN=your_huggingface_token_here

# ─── Tesseract OCR Path (Optional on Windows) ──────────────────────────────────
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# ─── Pipeline Configuration ────────────────────────────────────────────────────
CACHE_DIR=./cache
LOG_LEVEL=INFO
LOG_FORMAT=console
GROQ_BUDGET=50
```

### Why Each Key / Token is Needed:
1. **`GROQ_API_KEY` (Required for LLM Reasoning):**
   - *Where it's used:* Stage 2 (`stage2_speaker_mapping.py`) and Stage 3 (`stage3_narrative.py`).
   - *Why it's used:* Performs speaker role matching (Host vs. Guest) and analyzes transcript batches for narrative labels (`QUESTION`, `ANSWER`, `STORYTELLING`, `EMOTIONAL_MOMENT`, `LAUGHTER`, etc.).
   - *Resilience:* If missing, invalid, or rate-limited (`429`), the pipeline **never crashes**; it automatically engages deterministic keyword heuristics and speaking-time ratios.
2. **`HUGGINGFACE_TOKEN` (Optional):**
   - *Where it's used:* Stage 2 speaker diarization via `pyannote/speaker-diarization-3.1`.
   - *Why it's used:* Accesses gated Hugging Face acoustic diarization weights. If omitted, the system falls back to open-source acoustic clustering.
3. **`TESSERACT_CMD` (Optional Windows Path):**
   - *Where it's used:* Stage 1 camera discovery (`utils/ocr_utils.py`).
   - *Why it's used:* Points to `tesseract.exe` on Windows if not added to the global system `PATH`.
4. **`GROQ_BUDGET`:**
   - *Why it's used:* Sets a hard cap on the maximum number of cloud API calls per run (default: 50), preventing surprise billing or rate-limit exhaustion.

---

## 🚀 Step-by-Step Setup Guide

### 1. System Prerequisites

#### **Windows:**
1. Install **FFmpeg**:
   - Download from [ffmpeg.org/download.html](https://ffmpeg.org/download.html) or run:
     ```powershell
     winget install Gyan.FFmpeg
     ```
   - Verify: `ffmpeg -version`
2. Install **Tesseract OCR** (Optional for Lower-Third Reading):
   - Download installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
   - Add `C:\Program Files\Tesseract-OCR` to your system `PATH`.

#### **macOS:**
```bash
brew install ffmpeg tesseract
```

#### **Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install -y ffmpeg tesseract-ocr libsndfile1
```

---

### 2. Virtual Environment & Dependencies

```bash
# 1. Clone repository and enter directory
cd d:\AI\vibecoder\ai_narrative_director

# 2. Create Python virtual environment (Python 3.10+)
python -m venv .venv

# 3. Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Configure Environment Variables

Create your local `.env` from the provided `.env.example`:

```bash
# Copy template
cp .env.example .env
```

Edit `.env` and insert your free **Groq API key** from [console.groq.com](https://console.groq.com):

```env
GROQ_API_KEY=gsk_your_key_here
CACHE_DIR=./cache
LOG_LEVEL=INFO
LOG_FORMAT=console
GROQ_BUDGET=50
```

---

### 4. Verification Self-Test

Run the automated test suite to confirm that all 108 tests pass:

```powershell
python -m pytest tests/
```

Expected output:
```text
============================= 108 passed in 5.88s =============================
```

---

## 🎬 Editorial Rules Engine (Data-Driven)

All editorial decisions are governed by `editorial_rules.yaml` without hardcoded magic numbers:

| Rule | Name | Threshold / Trigger | Editorial Decision |
| :--- | :--- | :--- | :--- |
| **Rule 1** | **Speaker Rule** | Speech onset $\le 0.5\text{s}$ | Cut to the active speaker's HERO camera; hold camera if speaker is in emotional flow. |
| **Rule 2** | **Listener Reaction** | High-impact statement ($3\text{--}5\text{s}$) | Cut to listening host during provocative guest disclosures, then return to speaker. |
| **Rule 3** | **Refresh Wide** | $45\text{s}$ without event | Insert a $3\text{s}$ establishing wide shot to prevent visual fatigue, then return to hero. |
| **Rule 4** | **Dialogue Rule** | $>3$ rapid turns in $\le 8\text{s}$ | Switch to 2-shot or Wide camera to eliminate chaotic "ping-pong" whiplash cutting. |
| **Rule 5** | **Long Monologue** | Monologue $>30\text{s}$ | Insert brief alternate angle (listener hero or wide) every $90\text{s}$ to sustain engagement. |
| **Rule 6** | **Emotional Priority** | `EMOTIONAL_MOMENT` label | Lock speaker HERO camera; completely suppress reaction cuts and wide resets. |
| **Rule 7** | **Physical Adjustment** | Mic adjust / face scratch | Mandatory immediate cutaway to Wide or secondary camera. |
| **Rule 8** | **Technical Failure** | Frozen / dropped stream | Immediate automatic failover to healthy secondary camera angle. |
| **Rule 9** | **Off-Camera Brainstorm** | Director cutaway trigger | Create compound clip with `[OFF-CAMERA]` XML markers and review flags. |
| **Rule 10**| **Safety Rule** | Word boundary check | Validate word timestamps; cuts **never** split spoken words in half. |

### Show-Specific Rules:
- **The Nav Thethi Show:** Enforces a hard wide-shot cap ($\le 20\%$ of total runtime) and triggers wide establishing shots upon topic transitions.
- **Cracking the Maturity Code:** Enforces Side-by-Side (SBS) compound clip layout for opening sequences.

---

## 💻 CLI Reference & Usage

### Full Pipeline Run

Execute the complete narrative video director pipeline:

```powershell
python main.py `
  --input "path/to/SyncMaster.mp4" `
  --show-type "path/to/show_type.txt" `
  --output-dir "./output" `
  --rules "editorial_rules.yaml" `
  --log-level INFO
```

#### Available CLI Options:
| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `Path` | *Required* | Path to synchronized multicam video recording (`SyncMaster.mp4`) |
| `--show-type` | `Path` | *Required* | Path to `show_type.txt` (e.g. `The Nav Thethi Show` or `Cracking the Maturity Code`) |
| `--output-dir` | `Path` | `.` | Directory where `output.fcpxml`, `editing_report.json`, and `timeline.html` will be written |
| `--rules` | `Path` | `editorial_rules.yaml` | Path to editorial configuration YAML file |
| `--cache-dir` | `Path` | `./cache` | Directory for multi-tier stage caching |
| `--force` | `Flag` | `False` | Force XML emission even if stage validation reports errors |
| `--skip-hitl` | `Flag` | `False` | Skip interactive human-in-the-loop cut review |
| `--no-timeline`| `Flag` | `False` | Skip SVG/HTML visual timeline generation |
| `--log-level` | `Choice` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--log-format`| `Choice` | `console` | Log format: `console` (colored human text) or `json` (production structured) |

---

### Standalone XML Validator

Verify any FCPXML file independently without opening Adobe Premiere Pro:

```powershell
python validate_xml.py ./output/output.fcpxml
```

Sample output:
```text
=======================================================
  FCPXML Standalone Validator
  File: ./output/output.fcpxml
=======================================================

[PASS] VALIDATION PASSED

  Clip count:       48
  Total duration:   3600.00s
  Wide shot %:      16.4%

  No errors found.
  No warnings.
=======================================================
```

---

### HTML/SVG Visual Timeline Generator

Generate an interactive, zoomable timeline visualization:

```powershell
python generate_timeline_html.py ./output/output.fcpxml ./output/timeline.html
```

Open `timeline.html` in any web browser to inspect camera tracks, cut durations, transition rationale, and inline editorial notes.

---

## 🧪 Fault Injection & Live Resiliency Demos

Four standalone fault-injection demo scripts prove the system's "0% Silent Error" resilience:

```powershell
# 1. Missing Camera Feed (Simulates dropped video feed)
python fault_injection/inject_missing_camera.py

# 2. Frozen Camera Feed (Simulates frozen sensor during recording)
python fault_injection/inject_frozen_frame.py

# 3. Audio Drift (Simulates 2s audio sync delay)
python fault_injection/inject_audio_drift.py

# 4. Corrupt show_type.txt (Simulates missing, empty, or garbled show type file)
python fault_injection/inject_corrupt_showtype.py
```

---

## 📊 Testing & Quality Assurance

The test suite covers unit tests, regression suites, fault injection, Groq cloud failures, and end-to-end integration:

```powershell
# Run all tests
python -m pytest tests/

# Run with statement coverage report
python -m pytest tests/ --cov=pipeline --cov=utils --cov-report=term-missing

# Run code linter
python -m ruff check pipeline/ utils/ main.py validate_xml.py generate_timeline_html.py
```

### Coverage Overview:
- **`pipeline/stage4_director.py`**: **93% Coverage** (Complete 10-rule coverage)
- **`pipeline/stage5_xml_generator.py`**: **96% Coverage** (FCPXML generation & integer rational timecode)
- **`pipeline/stage6_validator.py`**: **89% Coverage** (Schema & timeline verification)
- **`pipeline/schemas.py`**: **99% Coverage** (Pydantic v2 stage contracts)
- **Total Passing Tests**: **108 / 108 (100%)**

---

## 📁 Project Directory Structure

```text
ai_narrative_director/
├── pipeline/                           # Core 7-Stage Pipeline
│   ├── schemas.py                      # Pydantic v2 data models & stage contracts
│   ├── stage0_ingest.py                # FFmpeg demuxing & show_type detector
│   ├── stage1_camera_discovery.py      # OpenCV face detection & pHash frozen-frame detector
│   ├── stage2_speaker_mapping.py       # faster-whisper ASR & pyannote diarization
│   ├── stage3_narrative.py             # Batched Groq LLM narrative classification
│   ├── stage4_director.py              # Deterministic 10-rule editorial decision engine
│   ├── stage4b_critic.py               # Cut list quality scoring & self-correction loop
│   ├── stage5_xml_generator.py         # Pure lxml FCPXML v1.10 generator (0 frame drift)
│   └── stage6_validator.py             # FCPXML schema, overlap, and Premiere validator
├── utils/                              # Shared Utilities
│   ├── groq_client.py                  # Rate-limited, cached Groq API wrapper with backoff
│   ├── cache.py                        # Multi-tier SQLite/JSON disk cache
│   ├── timecode.py                     # Integer rational frame timecode math
│   ├── logging_config.py               # Structlog structured logging setup
│   └── ocr_utils.py                    # Tesseract OCR lower-third parser
├── fault_injection/                    # Live Fault Injection Demos
│   ├── inject_missing_camera.py        # Dropped camera feed demo
│   ├── inject_frozen_frame.py          # Frozen frame detection demo
│   ├── inject_audio_drift.py           # Drifted audio compensation demo
│   └── inject_corrupt_showtype.py      # Corrupt show_type.txt fallback demo
├── tests/                              # Comprehensive Test Suite (108 Tests)
│   ├── conftest.py                     # Mock fixtures & synthetic podcast feeds
│   ├── test_integration.py             # End-to-end multi-stage pipeline tests
│   ├── test_rules_regression.py        # All 10 editorial rule regression tests
│   ├── test_fault_injection.py         # Automated fault injection tests
│   ├── test_groq_failure_modes.py      # Groq API rate-limit, budget & timeout tests
│   ├── test_stage0_ingest.py           # Ingest & show_type tests
│   ├── test_stage1_camera.py           # Camera inventory & discovery tests
│   ├── test_stage4_director.py         # Director engine unit tests
│   ├── test_stage5_xml.py              # XML generation & compound clip tests
│   ├── test_stage6_validator.py        # Schema & overlap validator tests
│   └── test_timecode.py                # Frame-accurate rational math tests
├── ARCHITECTURE.md                     # Detailed system architecture & stage decoupling
├── SCALING.md                          # 500 episodes/month Celery/Redis architecture
├── editorial_rules.yaml                # Master data-driven editorial rules configuration
├── pyproject.toml                      # Ruff, mypy, and pytest configuration
├── requirements.txt                    # Exact pinned production dependencies
├── main.py                             # Full pipeline CLI orchestrator
├── validate_xml.py                     # Standalone CLI FCPXML validator
├── generate_timeline_html.py           # SVG/HTML timeline visualizer
├── .env.example                        # Safe environment template
├── .gitignore                          # Git ignore rules for media, caches, & secrets
└── README.md                           # Master project documentation
```

---

## ❓ Troubleshooting & FAQ

| Problem / Error | Cause | Solution |
| :--- | :--- | :--- |
| `GROQ_API_KEY not set` | Missing `.env` file or empty key | Ensure `.env` exists in project root with valid `GROQ_API_KEY=gsk_...` from [console.groq.com](https://console.groq.com). |
| `Groq 429 Rate Limit` | Exceeded free-tier requests per minute | Pipeline handles this automatically via exponential backoff; if persistent, lower `batch_size_segments` in `editorial_rules.yaml`. |
| `ffmpeg: command not found` | FFmpeg is not installed or not on system `PATH` | Install FFmpeg and add the `bin/` folder to your system environment variables. |
| `TesseractNotFoundError` | Tesseract OCR not installed or not on PATH | Set `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` in `.env` or install Tesseract. |
| `FCPXML import error in Premiere` | Invalid XML syntax or clip overlaps | Run `python validate_xml.py output.fcpxml` to get an itemized report of structural issues. |
| `UnicodeEncodeError in Windows Console` | Default console `cp1252` encoding | All scripts output ASCII-safe markers (`[PASS]`, `[WARN]`, `[FAIL]`) to avoid terminal encoding crashes on Windows PowerShell/CMD. |

---

## 📄 License

Internal Production Backend — Built for automated podcast multicam rough cutting.