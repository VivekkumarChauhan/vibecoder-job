# ARCHITECTURE.md — AI Narrative Video Director

## Overview

The AI Narrative Video Director is a **production-grade, stage-isolated pipeline** that transforms a synchronized multicam podcast recording into a professionally edited FCPXML rough cut.

Architecture guiding principles:
1. **Never crash** — every stage returns `(result, warnings, errors)`, never raises
2. **Never produce invalid XML** — validated at generation and verified by a standalone validator
3. **AI is isolated** — Groq LLM calls are confined to Stages 2–3, cached to disk, never in the editorial engine
4. **Rules are data** — all thresholds in `editorial_rules.yaml`, never hardcoded
5. **0% silent errors** — every anomaly appears in `editing_report.json["warnings"]`

---

## Stage Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AI NARRATIVE VIDEO DIRECTOR                            │
│                                                                             │
│  SyncMaster.mp4 ──► [Stage 0: Ingest] ──► ingest.json                     │
│  show_type.txt  ──► │  FFmpeg/PyAV        (stream metadata, fps, duration) │
│                     ▼                                                       │
│                [Stage 1: Camera Discovery] ──► camera_inventory.json        │
│                │  OpenCV DNN face detect                                   │
│                │  Frame-diff + perceptual hash (frozen detection)           │
│                │  Tesseract OCR (supplementary: tablet/monitor detection)   │
│                │  Groq vision (optional confirmation)                       │
│                ▼                                                            │
│                [Stage 2: Speaker Mapping] ──► speaker_map.json             │
│                │  faster-whisper (ASR, word-level timestamps)              │
│                │  pyannote.audio (speaker diarization)                     │
│                │  Groq LLM (host/guest role assignment, ONE-TIME)          │
│                ▼                                                            │
│                [Stage 3: Narrative Understanding] ──► narrative.json        │
│                │  Groq LLM (JSON-only structured output, batched)          │
│                │  Pydantic validation at every Groq response               │
│                │  Retry up to N times → safe default label "unknown"       │
│                ▼                                                            │
│            ┌─[Stage 4: Director] ──► cut_list.json  ──┐                   │
│            │  │  Pure deterministic Python             │                   │
│            │  │  editorial_rules.yaml thresholds       │                   │
│            │  │  All 10 editorial rules               │                   │
│            │  │  Show-specific post-processing        │                   │
│            │  ▼                                       │                   │
│            │ [Stage 4b: Critic] ──► critic_report.json│                   │
│            │  │  Second-pass deterministic validator   │                   │
│            │  │  Rule violation check                  │                   │
│            │  │  Quality score computation            │                   │
│            │  ▼ violations? ──────────────────────────┘ (retry, max N)    │
│            │                                                               │
│                [Stage 5: XML Generator] ──► output.fcpxml                  │
│                │  Pure lxml — zero API calls                               │
│                │  FCP ticks (integer), never floats in XML                 │
│                │  Inline XML comments per cut                              │
│                ▼                                                            │
│                [Stage 6: Validator] ──► validation_report.json             │
│                │  Schema, overlaps, asset refs, timecodes                  │
│                │  Pipeline refuses to emit on failure (unless --force)     │
│                ▼                                                            │
│           output.fcpxml + editing_report.json + timeline.html              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why Each Stage is Separate

### Stage 0 — Ingest
**Responsibility**: source media → structured metadata.
**Why separate**: All subsequent stages depend on stream metadata. If the source is unreadable, we fail fast with a clear error before any expensive processing.

### Stage 1 — Camera Discovery
**Responsibility**: raw video → `{CAM_HOST_HERO, CAM_GUEST_HERO, CAM_WIDE}` mapping.
**Why separate**: Camera discovery is entirely vision-based and can be cached. Re-running editorial stages against the same SyncMaster makes zero vision API calls.
**Why no hardcoded coordinates**: Different shows, different setups. The system must work on any SyncMaster format.

### Stage 2 — Speaker Mapping
**Responsibility**: audio → word-timestamped transcript + speaker identity.
**Why separate**: ASR (faster-whisper) and diarization (pyannote) are the heaviest compute steps. Caching means re-runs never re-transcribe.
**Key design**: Identity established **once** from the first ~5 minutes of transcript; persisted for the entire episode. Never re-derived per segment (which would drift).

### Stage 3 — Narrative Understanding
**Responsibility**: transcript + diarization → narrative labels per segment.
**Why Groq here**: This is the one step requiring language understanding that goes beyond keyword detection (questions vs. storytelling vs. emotional moments). Groq's free-tier LLaMA model is well-suited for this classification task.
**Why not Groq in Stage 4**: Editorial decisions must be **deterministic and auditable**. Mixing LLM outputs into the cut engine would make the system non-reproducible and hard to debug.

### Stage 4 — Director (Editorial Decision Engine)
**Responsibility**: narrative segments + camera inventory → cut list.
**Why pure Python**: Every cut must be explainable and reproducible. The same intermediate JSON must always produce the same FCPXML. No randomness, no model drift.
**Why rules-as-data**: An editor should be able to change `refresh_interval_s: 45.0 → 30.0` and re-run only stages 4+5 in seconds, without touching code.

### Stage 4b — Critic Agent
**Responsibility**: cut list → violation report + quality score.
**Why separate**: The Director is optimistic (makes decisions). The Critic is skeptical (checks them). Separating these concerns enables the self-correction loop: Director → Critic → (if violations) → Director with adjusted params → Critic → ... (up to N iterations).

### Stage 5 — XML Generator
**Responsibility**: cut list → FCPXML v1.10.
**Why pure lxml**: Zero API calls, zero randomness. The same cut_list.json always produces the same XML byte-for-byte (except for UUIDs in cut IDs). `lxml` provides correct XML serialization with proper encoding/escaping.
**Timecode design**: All math uses `Fraction` arithmetic → integer FCP ticks. Never floating-point for timecodes. This prevents frame-boundary rounding errors that would break Premiere imports.

### Stage 6 — Validator
**Responsibility**: FCPXML file → pass/fail + itemized error list.
**Why mandatory**: Generated XML is treated as untrusted until validated. The Validator catches structural issues, missing asset refs, overlapping clips, and timecode bounds violations before the file is considered "done".

---

## Inter-Stage Communication: The Shared JSON Contract

Every stage reads its inputs and writes its outputs as **Pydantic-validated JSON files** cached to disk:

```
ingest.json ─────────── IngestResult (Pydantic)
camera_inventory.json ── CameraInventory (Pydantic)  
speaker_map.json ──────── SpeakerMapResult (Pydantic)
narrative.json ─────────── NarrativeResult (Pydantic)
cut_list.json ─────────────── CutList (Pydantic)
critic_report.json ──────────── CriticReport (Pydantic)
output.fcpxml ──────────────────── (lxml validated)
validation_report.json ───────────── ValidationReport (Pydantic)
```

This design means:
- Any stage can be run independently with a pre-existing JSON file
- A failure in Stage 3 never corrupts Stage 4's input (Pydantic validation at the boundary)
- Debugging: you can inspect exactly what each stage saw and produced

---

## Groq Usage: Why, Where, and How It's Bounded

### Where Groq is used
| Stage | Purpose | Calls per episode |
|-------|---------|------------------|
| Stage 1 | Vision: confirm camera roles | 1 (optional, cached) |
| Stage 2 | Text: map speaker labels to host/guest | 1 (cached) |
| Stage 3 | Text: classify narrative segments | N/20 batches (cached) |

### Why Groq for these and not others
- **Camera discovery**: Heuristics (face area ratio, frame-diff) are the primary method. Groq vision is an optional confirmation layer.
- **Speaker mapping**: Requires understanding conversational context to distinguish host from guest. Rule-based approaches fail on ambiguous openings.
- **Narrative labeling**: The difference between "storytelling" and "monologue" or "question" and "answer" requires language understanding beyond keyword matching.
- **Director/XML**: Pure logic. No benefit from LLM, only risk of non-determinism.

### How Groq calls are bounded
1. **Budget**: `budget_per_run: 50` in `editorial_rules.yaml`. Configurable per run.
2. **Cache**: Every response keyed by SHA-256(messages+model). Cache hit = zero API call.
3. **Retry**: Exponential backoff (base: 2s, max: 30s) on rate limit/errors.
4. **Schema enforcement**: `response_format={"type":"json_object"}` + Pydantic validation on every response. Bad JSON → retry → safe default (never raise).

---

## Error Handling Philosophy

### "0% Silent Error" Rule
```python
# Every stage returns this pattern:
StageResult(
    stage="director",
    success=True,        # or False
    result=CutList(...), # or None on fatal failure
    warnings=["..."],    # logged to editing_report.json
    errors=["..."],      # logged to editing_report.json
)
```

Never: `raise Exception("something went wrong")` out of a stage boundary.
Always: log it, pick a safe fallback, continue.

### Safe Fallback Hierarchy
1. Camera frozen → switch to next valid camera
2. Camera missing → use HOST_HERO (first active camera)
3. Groq failure → heuristic classification
4. pyannote unavailable → single-speaker transcript
5. faster-whisper unavailable → empty transcript with fallback segment
6. Tesseract unavailable → warning, OCR features disabled (non-blocking)
7. Validation failure → refuse to emit (or emit with `--force`, warning in report)

---

## The Self-Correction Loop

```
Director (iteration 0)
    │
    ▼
Critic → violations found?
    │ YES
    ├──► adjust rules (tighten wide cap, increase safety hold)
    │
    ▼
Director (iteration 1)
    │
    ▼
Critic → violations found?
    │ YES (iteration < max)
    ├──► adjust rules again
    │
    ▼
Director (iteration 2)
    │
    ▼
Critic → still violations? → emit best available + warning in report
```

`max_self_correction_iterations: 3` in `editorial_rules.yaml`.

---

## Human-in-the-Loop Hook

Cuts with `confidence < 0.6` (configurable) are written to `needs_review.json`:

```json
{
  "description": "Edit camera_id or approve before re-running XML generator",
  "cuts": [
    {"cut_id": "cut_abc123", "camera_id": "cam_2", "needs_review": true, ...}
  ]
}
```

The editor can approve or modify cuts, then re-run only Stage 5 (XML generator) + Stage 6 (validator) against the modified JSON — no AI calls needed.
