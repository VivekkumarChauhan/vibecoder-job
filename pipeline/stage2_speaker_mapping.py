"""pipeline/stage2_speaker_mapping.py — ASR + diarization + speaker role mapping.

Uses:
  - faster-whisper for word-level ASR transcription
  - pyannote.audio for speaker diarization
  - Groq LLM to map raw speaker labels (SPEAKER_00) to host/guest roles

Identity is established ONCE using the first N seconds of transcript and persisted
for the entire episode — never re-derived per segment.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from pipeline.schemas import (
    CameraInventory,
    IngestResult,
    SpeakerMapping,
    SpeakerMapResult,
    SpeakerRole,
    StageResult,
    TranscriptSegment,
    WordToken,
)
from utils.cache import StageCache, hash_payload
from utils.groq_client import GroqClient, build_json_system_prompt
from utils.logging_config import get_logger

logger = get_logger(__name__)


# ─── ASR: faster-whisper ──────────────────────────────────────────────────────

def _transcribe_with_whisper(
    audio_path: str,
    model_size: str = "small",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Transcribe audio with faster-whisper. Returns (segments, warnings)."""
    warnings: list[str] = []
    try:
        from faster_whisper import WhisperModel  # type: ignore[import]

        logger.info("whisper_loading", model_size=model_size)
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments_iter, info = model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            language=None,  # auto-detect
        )

        logger.info(
            "whisper_transcribing",
            detected_language=info.language,
            probability=round(info.language_probability, 3),
        )

        segments = []
        for seg in segments_iter:
            words = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        {
                            "word": w.word.strip(),
                            "start": float(w.start),
                            "end": float(w.end),
                            "confidence": float(w.probability),
                        }
                    )
            segments.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text.strip(),
                    "words": words,
                }
            )

        logger.info("whisper_complete", segment_count=len(segments))
        return segments, warnings

    except ImportError:
        warnings.append("faster-whisper not installed; transcription unavailable")
        return [], warnings
    except Exception as e:
        warnings.append(f"Whisper transcription failed: {e}")
        return [], warnings


# ─── Diarization: pyannote ────────────────────────────────────────────────────

def _diarize_with_pyannote(
    audio_path: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run pyannote diarization. Returns (speaker_segments, warnings)."""
    warnings: list[str] = []
    hf_token = os.getenv("HUGGINGFACE_TOKEN", "")

    if not hf_token:
        warnings.append(
            "HUGGINGFACE_TOKEN not set; diarization unavailable. "
            "Speaker mapping will use transcript-only heuristics."
        )
        return [], warnings

    try:
        from pyannote.audio import Pipeline  # type: ignore[import]

        logger.info("pyannote_loading")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )

        logger.info("pyannote_diarizing", audio=audio_path)
        diarization = pipeline(audio_path)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "speaker": speaker,
                    "start": float(turn.start),
                    "end": float(turn.end),
                }
            )

        logger.info("pyannote_complete", speaker_count=len({s["speaker"] for s in segments}))
        return segments, warnings

    except ImportError:
        warnings.append("pyannote.audio not installed; diarization unavailable")
        return [], warnings
    except Exception as e:
        warnings.append(f"Pyannote diarization failed: {e}")
        return [], warnings


# ─── Merge ASR + diarization ─────────────────────────────────────────────────

def _merge_asr_diarization(
    asr_segments: list[dict[str, Any]],
    diar_segments: list[dict[str, Any]],
    min_segment_s: float = 0.5,
) -> list[dict[str, Any]]:
    """Merge whisper segments with pyannote speaker labels by time overlap."""
    if not diar_segments:
        # No diarization — create synthetic single-speaker segments
        merged = []
        for i, seg in enumerate(asr_segments):
            if seg["end"] - seg["start"] >= min_segment_s:
                merged.append(
                    {
                        "segment_id": f"seg_{i:04d}",
                        "speaker_label": "SPEAKER_00",
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"],
                        "words": seg.get("words", []),
                    }
                )
        return merged

    merged = []
    seg_id = 0

    for asr_seg in asr_segments:
        if asr_seg["end"] - asr_seg["start"] < min_segment_s:
            continue

        # Find diarization segment that overlaps most with this ASR segment
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0

        for diar in diar_segments:
            overlap_start = max(asr_seg["start"], diar["start"])
            overlap_end = min(asr_seg["end"], diar["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar["speaker"]

        merged.append(
            {
                "segment_id": f"seg_{seg_id:04d}",
                "speaker_label": best_speaker,
                "start": asr_seg["start"],
                "end": asr_seg["end"],
                "text": asr_seg["text"],
                "words": asr_seg.get("words", []),
            }
        )
        seg_id += 1

    return merged


# ─── Speaker role mapping via Groq ───────────────────────────────────────────

def _map_speaker_roles_groq(
    merged_segments: list[dict[str, Any]],
    context_window_s: float,
    groq_client: GroqClient | None,
) -> tuple[SpeakerMapping, list[str]]:
    """Map raw speaker labels to host/guest roles using Groq LLM.

    Uses first `context_window_s` seconds of transcript.
    Identity is established ONCE and persisted — never re-derived.
    """
    warnings: list[str] = []

    # Collect unique speaker labels
    all_labels = sorted({s["speaker_label"] for s in merged_segments})

    if len(all_labels) == 0:
        warnings.append("No speakers detected; using default mapping")
        return SpeakerMapping(host_label="SPEAKER_00", all_labels=["SPEAKER_00"]), warnings

    if len(all_labels) == 1:
        return SpeakerMapping(host_label=all_labels[0], all_labels=all_labels, confidence=0.7), warnings

    # Build context from first N seconds
    context_segs = [s for s in merged_segments if s["start"] <= context_window_s]
    context_text = "\n".join(
        f"[{s['speaker_label']} @ {s['start']:.1f}s]: {s['text']}"
        for s in context_segs[:40]  # cap at 40 segments for prompt size
    )

    if groq_client is None:
        # Heuristic: speaker with most speaking time → host
        speaker_time: dict[str, float] = {}
        for s in merged_segments:
            label = s["speaker_label"]
            speaker_time[label] = speaker_time.get(label, 0.0) + (s["end"] - s["start"])
        host = max(speaker_time, key=lambda k: speaker_time[k])
        guest = next((lbl for lbl in all_labels if lbl != host), None)
        warnings.append("Groq unavailable; speaker mapping uses speaking-time heuristic")
        return SpeakerMapping(
            host_label=host,
            guest_label=guest,
            all_labels=all_labels,
            confidence=0.5,
            warnings=warnings,
        ), warnings

    schema_desc = (
        "{\n"
        '  "host_label": "SPEAKER_XX",\n'
        '  "guest_label": "SPEAKER_XX or null",\n'
        '  "reasoning": "brief explanation"\n'
        "}"
    )
    system_prompt = build_json_system_prompt(
        schema_desc,
        {"host_label": "SPEAKER_00", "guest_label": "SPEAKER_01", "reasoning": "SPEAKER_00 opens the show"},
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"This is the opening transcript of a podcast. "
                f"Speaker labels: {all_labels}\n\n"
                f"Transcript:\n{context_text}\n\n"
                "Identify who is the HOST (runs the show, asks questions, introduces topics) "
                "and who is the GUEST (interviewee, answers questions). "
                "The host typically speaks first, introduces themselves, and asks questions. "
                "Return JSON only."
            ),
        },
    ]

    result = groq_client.call(messages=messages, safe_default=None)

    if result and isinstance(result, dict):
        host_label = result.get("host_label", all_labels[0])
        guest_label = result.get("guest_label")
        if host_label not in all_labels:
            warnings.append(f"Groq returned unknown host label '{host_label}'; using first speaker")
            host_label = all_labels[0]
        if guest_label and guest_label not in all_labels:
            warnings.append(f"Groq returned unknown guest label '{guest_label}'; setting to None")
            guest_label = None
        mapping = SpeakerMapping(
            host_label=host_label,
            guest_label=guest_label,
            all_labels=all_labels,
            confidence=0.85,
        )
        logger.info(
            "speaker_roles_mapped",
            host=host_label,
            guest=guest_label,
            reasoning=str(result.get("reasoning", ""))[:100],
        )
        return mapping, warnings

    # Fallback
    warnings.append("Groq speaker mapping failed; using speaking-time heuristic")
    speaker_time = {label: 0.0 for label in all_labels}
    for s in merged_segments:
        speaker_time[s["speaker_label"]] = speaker_time.get(s["speaker_label"], 0.0) + (s["end"] - s["start"])
    host = max(speaker_time, key=lambda k: speaker_time[k])
    guest = next((lbl for lbl in all_labels if lbl != host), None)
    return SpeakerMapping(host_label=host, guest_label=guest, all_labels=all_labels, confidence=0.4), warnings


# ─── Main stage entry point ──────────────────────────────────────────────────

def extract_audio_from_video(source_path: str, output_path: str) -> bool:
    """Extract mono 16kHz audio from video for ASR/diarization."""
    import subprocess

    cmd = [
        "ffmpeg", "-y", "-i", source_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def map_speakers(
    ingest_result: IngestResult,
    source_path: str,
    camera_inventory: CameraInventory,
    rules: dict,
    cache: StageCache | None = None,
    groq_client: GroqClient | None = None,
) -> StageResult:
    """Stage 2: Transcribe + diarize + map speakers to host/guest roles."""
    start_time = time.monotonic()
    warnings: list[str] = []
    errors: list[str] = []

    # Cache check
    cache_key = hash_payload({"source": source_path, "rules_sm": rules.get("speaker_mapping", {})})
    if cache:
        cached = cache.get_stage("speaker_mapping", cache_key)
        if cached:
            logger.info("stage2_cache_hit")
            return StageResult(
                stage="speaker_mapping",
                success=True,
                result=SpeakerMapResult.model_validate(cached),
                duration_s=time.monotonic() - start_time,
            )

    # Extract audio to temp WAV
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    audio_ok = extract_audio_from_video(source_path, audio_path)
    if not audio_ok:
        warnings.append(f"Audio extraction failed for {source_path}; using empty transcript")
        audio_path_to_use = source_path  # try direct
    else:
        audio_path_to_use = audio_path

    # ASR
    asr_segments, asr_warnings = _transcribe_with_whisper(audio_path_to_use)
    warnings.extend(asr_warnings)

    # Diarization
    diar_segments, diar_warnings = _diarize_with_pyannote(audio_path_to_use)
    warnings.extend(diar_warnings)

    # Clean up temp audio
    Path(audio_path).unlink(missing_ok=True)

    # Merge ASR + diarization
    min_seg_s = rules.get("speaker_mapping", {}).get("min_segment_seconds", 0.5)
    merged = _merge_asr_diarization(asr_segments, diar_segments, min_segment_s=min_seg_s)

    if not merged:
        warnings.append("No transcript segments produced; pipeline will use empty transcript")
        # Generate a single fallback segment
        merged = [
            {
                "segment_id": "seg_0000",
                "speaker_label": "SPEAKER_00",
                "start": 0.0,
                "end": ingest_result.duration_s,
                "text": "[No transcript available]",
                "words": [],
            }
        ]

    # Map speaker roles
    context_window_s = rules.get("speaker_mapping", {}).get("context_window_seconds", 300)
    mapping, map_warnings = _map_speaker_roles_groq(merged, context_window_s, groq_client)
    warnings.extend(map_warnings)
    warnings.extend(mapping.warnings)

    # Build TranscriptSegment objects — enrich with speaker role
    transcript: list[TranscriptSegment] = []
    for m in merged:
        speaker_label = m["speaker_label"]
        role = SpeakerRole.HOST if speaker_label == mapping.host_label else (
            SpeakerRole.GUEST if speaker_label == mapping.guest_label else SpeakerRole.UNKNOWN
        )

        # Assign camera: host → host_hero, guest → guest_hero
        cam_id = None
        if role == SpeakerRole.HOST and mapping.host_camera_id:
            cam_id = mapping.host_camera_id
        elif role == SpeakerRole.GUEST and mapping.guest_camera_id:
            cam_id = mapping.guest_camera_id

        words = [WordToken(**w) for w in m.get("words", [])]

        seg = TranscriptSegment(
            segment_id=m["segment_id"],
            speaker_label=speaker_label,
            start=m["start"],
            end=m["end"],
            text=m["text"],
            words=words,
            camera_id=cam_id,
        )
        transcript.append(seg)

    result = SpeakerMapResult(
        mapping=mapping,
        transcript=transcript,
        warnings=warnings,
        errors=errors,
    )

    if cache:
        cache.set_stage("speaker_mapping", cache_key, result.model_dump())

    logger.info(
        "stage2_complete",
        segments=len(transcript),
        host=mapping.host_label,
        guest=mapping.guest_label,
        warnings=len(warnings),
    )

    return StageResult(
        stage="speaker_mapping",
        success=True,
        result=result,
        warnings=warnings,
        errors=errors,
        duration_s=time.monotonic() - start_time,
    )
