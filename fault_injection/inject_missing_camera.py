"""fault_injection/inject_missing_camera.py — Demo: missing camera feed.

Simulates a SyncMaster where one camera feed is completely missing.
Shows graceful degradation: warning emitted, pipeline continues with available cameras.

Run:
  python fault_injection/inject_missing_camera.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.schemas import (
    CameraInfo,
    CameraInventory,
    CameraRole,
    CutList,
    NarrativeLabel,
    NarrativeResult,
    NarrativeSegment,
    ShowType,
    SpeakerRole,
)
from pipeline.stage4_director import direct
from utils.logging_config import configure_logging

configure_logging(level="INFO", fmt="console")


def run_demo() -> None:
    print("\n[FAULT INJECTION] Missing Camera Feed Demo")
    print("=" * 55)
    print("Scenario: Camera 2 (GUEST_HERO) is completely missing from SyncMaster.")
    print("Expected: Warning emitted, pipeline continues using remaining cameras.\n")

    # Inventory with only 2 cameras (cam_2 / GUEST_HERO is missing)
    inventory = CameraInventory(
        cameras=[
            CameraInfo(
                camera_id="cam_1",
                stream_index=0,
                role=CameraRole.HOST_HERO,
                is_active=True,
                face_detected=True,
                face_area_ratio=0.15,
            ),
            # cam_2 (GUEST_HERO) intentionally absent
            CameraInfo(
                camera_id="cam_3",
                stream_index=2,
                role=CameraRole.WIDE,
                is_active=True,
                face_detected=False,
                face_area_ratio=0.02,
                is_wide_shot=True,
            ),
        ],
        role_map={
            CameraRole.HOST_HERO.value: "cam_1",
            # No GUEST_HERO mapped!
            CameraRole.WIDE.value: "cam_3",
        },
        total_cameras=2,
        active_cameras=2,
        empty_cameras=0,
        warnings=["Camera cam_2 (GUEST_HERO) stream not found — feed missing"],
    )

    # Narrative with speaker alternation
    narrative = NarrativeResult(
        segments=[
            NarrativeSegment(
                segment_id="seg_0001",
                speaker_label="SPEAKER_00",
                speaker_role=SpeakerRole.HOST,
                start=0.0,
                end=10.0,
                text="Welcome to the show.",
                label=NarrativeLabel.INTRO,
                confidence=0.9,
            ),
            NarrativeSegment(
                segment_id="seg_0002",
                speaker_label="SPEAKER_01",
                speaker_role=SpeakerRole.GUEST,
                start=10.0,
                end=25.0,
                text="Great to be here.",
                label=NarrativeLabel.ANSWER,
                confidence=0.9,
            ),
            NarrativeSegment(
                segment_id="seg_0003",
                speaker_label="SPEAKER_00",
                speaker_role=SpeakerRole.HOST,
                start=25.0,
                end=60.0,
                text="Let's talk about leadership.",
                label=NarrativeLabel.QUESTION,
                confidence=0.9,
            ),
        ],
        show_type=ShowType.NAV_THETHI,
    )

    rules = {
        "director": {
            "listener_reaction_min_s": 3.0,
            "listener_reaction_max_s": 5.0,
            "refresh_interval_s": 45.0,
            "refresh_wide_duration_s": 3.0,
            "monologue_threshold_s": 30.0,
            "monologue_alternate_interval_s": 90.0,
            "physical_adjustment_gaze_threshold_s": 3.0,
            "safety_min_hold_s": 1.0,
            "rapid_exchange_max_turn_s": 8.0,
            "rapid_exchange_min_turns": 3,
        },
        "show_types": {
            "The Nav Thethi Show": {"wide_shot_max_pct": 20.0, "pacing": "cinematic"},
        },
    }

    result = direct(narrative, inventory, rules)

    print("[PASS] Director ran successfully despite missing camera")
    print(f"   Total cuts generated: {len(result.result.cuts)}")
    print(f"   Warnings: {len(result.warnings)}")
    print()
    for w in inventory.warnings + result.warnings:
        print(f"   [WARN] {w}")
    print()
    print("   Cut list:")
    for cut in result.result.cuts:
        print(f"   [{cut.start_s:.1f}s-{cut.end_s:.1f}s] {cut.camera_id} | {cut.rule_tag}")

    print()
    print("[INFO] Recovery: Guest speaker cuts fell back to HOST_HERO (cam_1)")
    print("=" * 55)

    # Verify no crash and cuts are valid
    assert result.result is not None
    assert len(result.result.cuts) > 0
    for cut in result.result.cuts:
        assert cut.camera_id in ["cam_1", "cam_3"], f"Unexpected camera: {cut.camera_id}"
    print("[PASS] All assertions passed - graceful degradation confirmed\n")


if __name__ == "__main__":
    run_demo()
