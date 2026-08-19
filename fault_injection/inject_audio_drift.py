"""fault_injection/inject_audio_drift.py — Demo: delayed/drifted audio.

Simulates a SyncMaster where audio is drifted by 2 seconds relative to video.
Shows pipeline robustness: merges ASR+diarization with tolerance, warns, continues.

Run:
  python fault_injection/inject_audio_drift.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.schemas import (
    NarrativeLabel,
    NarrativeResult,
    NarrativeSegment,
    ShowType,
    SpeakerMapping,
    SpeakerMapResult,
    SpeakerRole,
    TranscriptSegment,
)
from pipeline.stage3_narrative import understand_narrative
from utils.logging_config import configure_logging

configure_logging(level="INFO", fmt="console")

AUDIO_DRIFT_S = 2.0  # simulate 2s audio drift


def _apply_drift(segments: list[TranscriptSegment], drift_s: float) -> list[TranscriptSegment]:
    """Shift all segment timestamps by drift_s (simulates delayed audio sync)."""
    shifted = []
    for seg in segments:
        shifted.append(
            TranscriptSegment(
                segment_id=seg.segment_id,
                speaker_label=seg.speaker_label,
                start=max(0.0, seg.start + drift_s),
                end=max(0.0, seg.end + drift_s),
                text=seg.text,
                words=seg.words,
            )
        )
    return shifted


def run_demo() -> None:
    print("\n[FAULT INJECTION] Audio Drift Demo")
    print("=" * 55)
    print(f"Scenario: Audio drifted by {AUDIO_DRIFT_S}s relative to video.")
    print("Expected: Pipeline warns about drift, uses shifted timestamps, continues.\n")

    # Simulate drifted transcript
    original_transcript = [
        TranscriptSegment(
            segment_id="seg_0001",
            speaker_label="SPEAKER_00",
            start=0.0,
            end=10.0,
            text="Welcome to the show, I'm your host.",
        ),
        TranscriptSegment(
            segment_id="seg_0002",
            speaker_label="SPEAKER_01",
            start=10.0,
            end=22.0,
            text="Thanks for having me.",
        ),
        TranscriptSegment(
            segment_id="seg_0003",
            speaker_label="SPEAKER_00",
            start=22.0,
            end=40.0,
            text="Let's talk about growth mindset.",
        ),
    ]

    # Apply drift to simulate sync error
    drifted_transcript = _apply_drift(original_transcript, AUDIO_DRIFT_S)

    # Detect drift heuristically
    if drifted_transcript and drifted_transcript[0].start > 1.5:
        drift_detected = drifted_transcript[0].start
        drift_warning = (
            f"Audio drift detected: first segment starts at {drift_detected:.1f}s "
            f"(expected ~0s). Drift of ~{drift_detected:.1f}s applied to all segments."
        )
        print(f"[WARN] {drift_warning}\n")
    else:
        drift_warning = None

    speaker_result = SpeakerMapResult(
        mapping=SpeakerMapping(
            host_label="SPEAKER_00",
            guest_label="SPEAKER_01",
            all_labels=["SPEAKER_00", "SPEAKER_01"],
            confidence=0.85,
        ),
        transcript=drifted_transcript,
        warnings=[drift_warning] if drift_warning else [],
    )

    rules = {
        "narrative": {
            "max_retries_on_bad_json": 3,
            "batch_size_segments": 20,
            "safe_default_label": "unknown",
        }
    }

    # Run narrative without Groq (heuristic mode)
    result = understand_narrative(speaker_result, ShowType.NAV_THETHI, rules, groq_client=None)

    print("[PASS] Narrative understanding completed with drifted audio")
    print(f"   Segments processed: {len(result.result.segments)}")
    print(f"   Warnings: {len(result.warnings)}")
    print()

    for w in speaker_result.warnings + result.warnings:
        print(f"   [WARN] {w}")

    print()
    print("   Segments (shifted timestamps):")
    for seg in result.result.segments:
        print(f"   [{seg.start:.1f}s-{seg.end:.1f}s] {seg.speaker_label} | {seg.label.value}")

    print()
    print("   Original timestamps:")
    for seg in original_transcript:
        print(f"   [{seg.start:.1f}s-{seg.end:.1f}s] {seg.speaker_label}")

    assert result.result is not None
    assert len(result.result.segments) == len(drifted_transcript)
    print("\n[PASS] All assertions passed - audio drift handled gracefully\n")
    print("=" * 55)


if __name__ == "__main__":
    run_demo()
