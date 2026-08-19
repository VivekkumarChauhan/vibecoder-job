"""pipeline/stage3_narrative.py — Narrative understanding via Groq LLM.

Classifies each diarized segment with a narrative label:
question / answer / storytelling / emotional_moment / laughter / interruption /
silence / transition / topic_change / monologue / framework_discussion /
shared_laughter / intro / outro / off_camera / unknown

Also detects:
- Off-camera triggers ("stop rolling", "stop the rolling")
- Resume triggers ("restart rolling", "restart the rolling")
- Physical events (mic adjust, face scratch, etc.) — from transcript keywords

All Groq calls use JSON-only mode and are validated against NarrativeSegment schema.
On validation failure: retry up to max_retries, then fall back to safe_default label "unknown".
"""
from __future__ import annotations

import contextlib
import re
import time
from typing import Any

from pipeline.schemas import (
    NarrativeLabel,
    NarrativeResult,
    NarrativeSegment,
    ShowType,
    SpeakerMapResult,
    SpeakerRole,
    StageResult,
)
from utils.cache import StageCache, hash_payload
from utils.groq_client import GroqClient, build_json_system_prompt
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ─── Off-camera trigger patterns ─────────────────────────────────────────────
OFF_CAMERA_PATTERNS = [
    r"\bstop\s+rolling\b",
    r"\bstop\s+the\s+rolling\b",
    r"\bcut\s+the\s+camera\b",
    r"\bstop\s+recording\b",
]
RESUME_PATTERNS = [
    r"\brestart\s+rolling\b",
    r"\brestart\s+the\s+rolling\b",
    r"\bstart\s+rolling\b",
    r"\bstart\s+recording\b",
    r"\bwe.re\s+rolling\b",
]

# Physical adjustment keywords in transcript text
PHYSICAL_EVENT_PATTERNS = {
    "mic_adjust": [r"\bmic\b", r"\bmicrophone\b", r"\bheadphone\b"],
    "posture_change": [r"\bhold\s+on\b", r"\bjust\s+a\s+second\b"],
}


def _detect_off_camera(text: str) -> tuple[bool, bool]:
    """Detect off-camera (stop) and resume (restart) triggers in text.

    Returns (off_camera_trigger, resume_trigger).
    """
    text_lower = text.lower()
    off_cam = any(re.search(p, text_lower) for p in OFF_CAMERA_PATTERNS)
    resume = any(re.search(p, text_lower) for p in RESUME_PATTERNS)
    return off_cam, resume


def _detect_physical_event(text: str) -> str | None:
    """Detect physical event keywords in transcript text."""
    text_lower = text.lower()
    for event, patterns in PHYSICAL_EVENT_PATTERNS.items():
        if any(re.search(p, text_lower) for p in patterns):
            return event
    return None


def _build_narrative_prompt(
    segments_batch: list[dict[str, Any]],
    show_type: ShowType,
    speaker_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    """Build Groq messages for narrative classification of a batch of segments."""
    schema_desc = """
{
  "segments": [
    {
      "segment_id": "seg_0001",
      "label": "question|answer|storytelling|emotional_moment|laughter|interruption|silence|transition|topic_change|monologue|framework_discussion|shared_laughter|intro|outro|off_camera|unknown",
      "sub_labels": ["optional additional labels from the same list"],
      "confidence": 0.0,
      "has_laughter": false,
      "has_emotion": false,
      "has_interruption": false
    }
  ]
}
"""
    example = {
        "segments": [
            {
                "segment_id": "seg_0001",
                "label": "question",
                "sub_labels": [],
                "confidence": 0.9,
                "has_laughter": False,
                "has_emotion": False,
                "has_interruption": False,
            }
        ]
    }

    system_prompt = build_json_system_prompt(schema_desc, example)

    batch_text = "\n".join(
        f"[{s['segment_id']} | {s['speaker_label']} ({speaker_mapping.get(s['speaker_label'], 'unknown')}) "
        f"| {s['start']:.1f}s–{s['end']:.1f}s]: {s['text']}"
        for s in segments_batch
    )

    user_content = (
        f"Show type: {show_type.value}\n\n"
        f"Speaker roles: {speaker_mapping}\n\n"
        f"Classify each of these podcast transcript segments with the most appropriate narrative label.\n\n"
        f"Segments:\n{batch_text}\n\n"
        "Rules:\n"
        "- 'question': host/guest asks a direct question\n"
        "- 'answer': responding to a question\n"
        "- 'storytelling': extended narrative, personal story\n"
        "- 'emotional_moment': vulnerable, heartfelt, emotional delivery\n"
        "- 'laughter': laughter or humorous exchange\n"
        "- 'interruption': speaker cuts in mid-sentence\n"
        "- 'silence': extended silence or pause\n"
        "- 'transition': topic transition or intro/outro of a section\n"
        "- 'topic_change': clear shift to a new subject\n"
        "- 'monologue': extended uninterrupted speech by one person\n"
        "- 'framework_discussion': (Cracking the Maturity Code) discussing a concept/framework\n"
        "- 'shared_laughter': both speakers laughing together\n"
        "- 'intro': show/segment introduction\n"
        "- 'outro': show/segment wrap-up\n"
        "- 'off_camera': off-camera / stop rolling\n"
        "Return JSON only with the 'segments' array."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _classify_batch_groq(
    batch: list[dict[str, Any]],
    show_type: ShowType,
    speaker_mapping: dict[str, str],
    groq_client: GroqClient,
    max_retries: int,
    safe_default: str,
) -> list[dict[str, Any]]:
    """Classify a batch of segments using Groq LLM. Returns enriched dicts."""
    messages = _build_narrative_prompt(batch, show_type, speaker_mapping)

    for attempt in range(max_retries):
        result = groq_client.call(
            messages=messages,
            safe_default=None,
        )

        if result is None:
            break

        classified = result.get("segments", [])
        if not classified:
            logger.warning("groq_empty_segments_response", attempt=attempt)
            continue

        # Build lookup by segment_id
        classified_map = {s["segment_id"]: s for s in classified}

        # Merge back into batch
        output = []
        for seg in batch:
            cl = classified_map.get(seg["segment_id"])
            if cl:
                output.append({**seg, **cl})
            else:
                output.append({**seg, "label": safe_default, "confidence": 0.0})
        return output

    # All retries failed → safe default
    logger.warning("groq_narrative_fallback", batch_size=len(batch), default=safe_default)
    return [{**s, "label": safe_default, "confidence": 0.0} for s in batch]


def _classify_heuristic(segment: dict[str, Any]) -> dict[str, Any]:
    """Heuristic narrative classification when Groq is unavailable."""
    text = segment.get("text", "").strip()
    text_lower = text.lower()
    duration = segment.get("end", 0) - segment.get("start", 0)

    label = "unknown"
    has_laughter = any(w in text_lower for w in ["haha", "hehe", "lol", "laugh", "funny"])
    has_emotion = any(w in text_lower for w in ["feel", "heart", "love", "miss", "cry", "amazing"])

    if text.endswith("?") or text_lower.startswith(("what", "how", "why", "when", "where", "who", "do you", "can you", "tell me")):
        label = "question"
    elif has_laughter:
        label = "laughter"
    elif duration > 30:
        label = "monologue"
    elif not text or duration < 1:
        label = "silence"
    else:
        label = "answer"

    return {
        **segment,
        "label": label,
        "sub_labels": [],
        "confidence": 0.4,
        "has_laughter": has_laughter,
        "has_emotion": has_emotion,
        "has_interruption": False,
    }


def understand_narrative(
    speaker_result: SpeakerMapResult,
    show_type: ShowType,
    rules: dict,
    cache: StageCache | None = None,
    groq_client: GroqClient | None = None,
) -> StageResult:
    """Stage 3: Classify all segments with narrative labels."""
    start_time = time.monotonic()
    warnings: list[str] = []
    errors: list[str] = []

    transcript = speaker_result.transcript
    mapping = speaker_result.mapping

    # Cache key
    transcript_hash = hash_payload([s.model_dump() for s in transcript])
    cache_key = hash_payload({"transcript": transcript_hash, "show": show_type.value})
    if cache:
        cached = cache.get_stage("narrative", cache_key)
        if cached:
            logger.info("stage3_cache_hit")
            return StageResult(
                stage="narrative",
                success=True,
                result=NarrativeResult.model_validate(cached),
                duration_s=time.monotonic() - start_time,
            )

    narrative_rules = rules.get("narrative", {})
    max_retries = narrative_rules.get("max_retries_on_bad_json", 3)
    batch_size = narrative_rules.get("batch_size_segments", 20)
    safe_default = narrative_rules.get("safe_default_label", "unknown")

    # Build speaker role mapping for prompt context
    speaker_role_map: dict[str, str] = {}
    if mapping.host_label:
        speaker_role_map[mapping.host_label] = "host"
    if mapping.guest_label:
        speaker_role_map[mapping.guest_label] = "guest"

    # Convert transcript to raw dicts for processing
    raw_segments = [s.model_dump() for s in transcript]

    # First pass: detect off-camera triggers and physical events from text
    for seg in raw_segments:
        text = seg.get("text", "")
        off_cam, resume = _detect_off_camera(text)
        seg["off_camera_trigger"] = off_cam
        seg["resume_trigger"] = resume
        seg["physical_event"] = _detect_physical_event(text)
        if off_cam:
            seg["label"] = NarrativeLabel.OFF_CAMERA.value
            seg["confidence"] = 1.0
            seg["has_laughter"] = False
            seg["has_emotion"] = False
            seg["has_interruption"] = False

    # Classify non-off-camera segments
    segments_to_classify = [s for s in raw_segments if not s.get("off_camera_trigger")]

    if groq_client is not None:
        # Process in batches
        classified_segments: list[dict] = []
        for i in range(0, len(segments_to_classify), batch_size):
            batch = segments_to_classify[i : i + batch_size]
            classified = _classify_batch_groq(
                batch, show_type, speaker_role_map, groq_client, max_retries, safe_default
            )
            classified_segments.extend(classified)
    else:
        warnings.append("Groq unavailable; using heuristic narrative classification")
        classified_segments = [_classify_heuristic(s) for s in segments_to_classify]

    # Merge back off-camera segments
    off_cam_segs = {s["segment_id"]: s for s in raw_segments if s.get("off_camera_trigger")}
    classified_map = {s["segment_id"]: s for s in classified_segments}

    # Rebuild in original order
    final_raw: list[dict] = []
    for seg in raw_segments:
        sid = seg["segment_id"]
        if sid in off_cam_segs:
            final_raw.append(off_cam_segs[sid])
        elif sid in classified_map:
            final_raw.append(classified_map[sid])
        else:
            final_raw.append({**seg, "label": safe_default, "confidence": 0.0})

    # Build NarrativeSegment objects
    narrative_segments: list[NarrativeSegment] = []
    for raw in final_raw:
        try:
            label_str = raw.get("label", safe_default)
            try:
                label = NarrativeLabel(label_str)
            except ValueError:
                label = NarrativeLabel.UNKNOWN
                warnings.append(f"Unknown narrative label '{label_str}' for {raw['segment_id']}; using UNKNOWN")

            sub_labels = []
            for sl in raw.get("sub_labels", []):
                with contextlib.suppress(ValueError):
                    sub_labels.append(NarrativeLabel(sl))

            speaker_label = raw.get("speaker_label", "SPEAKER_00")
            speaker_role = SpeakerRole.HOST if speaker_label == mapping.host_label else (
                SpeakerRole.GUEST if speaker_label == mapping.guest_label else SpeakerRole.UNKNOWN
            )

            seg = NarrativeSegment(
                segment_id=raw["segment_id"],
                speaker_label=speaker_label,
                speaker_role=speaker_role,
                start=float(raw.get("start", 0.0)),
                end=float(raw.get("end", 0.0)),
                text=raw.get("text", ""),
                label=label,
                sub_labels=sub_labels,
                confidence=float(raw.get("confidence", 0.0)),
                has_laughter=bool(raw.get("has_laughter", False)),
                has_emotion=bool(raw.get("has_emotion", False)),
                has_interruption=bool(raw.get("has_interruption", False)),
                physical_event=raw.get("physical_event"),
                off_camera_trigger=bool(raw.get("off_camera_trigger", False)),
                resume_trigger=bool(raw.get("resume_trigger", False)),
            )
            narrative_segments.append(seg)
        except Exception as e:
            warnings.append(f"Failed to build NarrativeSegment for {raw.get('segment_id')}: {e}")
            # Add safe fallback segment
            narrative_segments.append(
                NarrativeSegment(
                    segment_id=raw.get("segment_id", "seg_unknown"),
                    speaker_label=raw.get("speaker_label", "SPEAKER_00"),
                    start=float(raw.get("start", 0.0)),
                    end=float(raw.get("end", 0.0)),
                    text=raw.get("text", ""),
                    label=NarrativeLabel.UNKNOWN,
                    confidence=0.0,
                )
            )

    result = NarrativeResult(
        segments=narrative_segments,
        show_type=show_type,
        warnings=warnings,
        errors=errors,
    )

    if cache:
        cache.set_stage("narrative", cache_key, result.model_dump())

    label_counts: dict[str, int] = {}
    for seg in narrative_segments:
        key = seg.label.value
        label_counts[key] = label_counts.get(key, 0) + 1

    logger.info(
        "stage3_complete",
        total_segments=len(narrative_segments),
        label_distribution=label_counts,
        show_type=show_type.value,
    )

    return StageResult(
        stage="narrative",
        success=True,
        result=result,
        warnings=warnings,
        errors=errors,
        duration_s=time.monotonic() - start_time,
    )
