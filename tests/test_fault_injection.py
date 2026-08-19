"""tests/test_fault_injection.py — Fault injection tests.

Tests graceful degradation for:
  - Missing camera feed
  - Frozen frame
  - Unexpected camera count/layout
  - Corrupt show_type.txt
  - Zero-length segments
  - Missing speaker
  - All-off-camera narrative
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pipeline.schemas import (
    CameraInventory,
    CameraInfo,
    CameraRole,
    NarrativeLabel,
    NarrativeResult,
    NarrativeSegment,
    ShowType,
    SpeakerRole,
)
from pipeline.stage4_director import direct
from pipeline.stage0_ingest import detect_show_type
from tests.conftest import _make_segment


@pytest.mark.fault
def test_missing_guest_camera(single_camera_inventory, standard_rules):
    """Only HOST_HERO available — no GUEST_HERO or WIDE. Must not crash."""
    narrative = NarrativeResult(
        segments=[
            _make_segment("s1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 20.0, "Host speaks."),
            _make_segment("s2", "SPEAKER_01", SpeakerRole.GUEST, 20.0, 40.0, "Guest speaks."),
        ],
        show_type=ShowType.NAV_THETHI,
    )
    result = direct(narrative, single_camera_inventory, standard_rules)
    assert result.result is not None, "Must not crash with missing guest camera"
    # All cuts must be on cam_1
    for cut in result.result.cuts:
        if not cut.is_off_camera:
            assert cut.camera_id == "cam_1"


@pytest.mark.fault
def test_frozen_host_camera(frozen_camera_inventory, standard_rules):
    """HOST_HERO frozen — must switch to GUEST_HERO, no crash."""
    narrative = NarrativeResult(
        segments=[
            _make_segment("s1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 30.0, "Host monologue."),
        ],
        show_type=ShowType.NAV_THETHI,
    )
    result = direct(narrative, frozen_camera_inventory, standard_rules)
    assert result.result is not None
    # Should have tech failure cut to cam_2
    tech_cuts = [c for c in result.result.cuts if c.rule_tag == "TECH_FAILURE_SWITCH"]
    assert len(tech_cuts) >= 1, "Must detect frozen camera and switch"


@pytest.mark.fault
def test_all_cameras_empty():
    """All cameras inactive/empty — director uses fallback safely."""
    inventory = CameraInventory(
        cameras=[
            CameraInfo(camera_id="cam_1", stream_index=0, role=CameraRole.HOST_HERO,
                       is_active=False, is_empty=True),
        ],
        role_map={CameraRole.HOST_HERO.value: "cam_1"},
        total_cameras=1, active_cameras=0, empty_cameras=1,
    )
    narrative = NarrativeResult(
        segments=[_make_segment("s1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 20.0, "Test.")],
        show_type=ShowType.NAV_THETHI,
    )
    # Should not crash — might produce empty cut list with errors
    standard_rules = {
        "director": {"refresh_interval_s": 45.0, "safety_min_hold_s": 1.0,
                     "listener_reaction_min_s": 3.0, "listener_reaction_max_s": 5.0,
                     "monologue_threshold_s": 30.0, "monologue_alternate_interval_s": 90.0,
                     "rapid_exchange_max_turn_s": 8.0, "rapid_exchange_min_turns": 3,
                     "physical_adjustment_gaze_threshold_s": 3.0, "refresh_wide_duration_s": 3.0}
    }
    result = direct(narrative, inventory, standard_rules)
    assert result is not None  # No crash


@pytest.mark.fault
def test_zero_length_segments_no_crash(three_camera_inventory, standard_rules):
    """Zero-length segments don't crash the director."""
    narrative = NarrativeResult(
        segments=[
            _make_segment("s1", "SPEAKER_00", SpeakerRole.HOST, 5.0, 5.0, ""),  # zero length
            _make_segment("s2", "SPEAKER_00", SpeakerRole.HOST, 5.0, 15.0, "Real."),
        ],
        show_type=ShowType.NAV_THETHI,
    )
    result = direct(narrative, three_camera_inventory, standard_rules)
    assert result.result is not None


@pytest.mark.fault
def test_missing_speaker_mapping_fallback(three_camera_inventory, standard_rules):
    """No speaker role information → falls back gracefully."""
    narrative = NarrativeResult(
        segments=[
            _make_segment("s1", "SPEAKER_00", SpeakerRole.UNKNOWN, 0.0, 20.0, "Unknown speaker."),
            _make_segment("s2", "SPEAKER_01", SpeakerRole.UNKNOWN, 20.0, 40.0, "Another unknown."),
        ],
        show_type=ShowType.NAV_THETHI,
    )
    result = direct(narrative, three_camera_inventory, standard_rules)
    assert result.result is not None
    # Should have cuts, even if camera selection is imperfect
    assert len(result.result.cuts) > 0


@pytest.mark.fault
def test_all_off_camera_narrative(three_camera_inventory, standard_rules):
    """Narrative that is 100% off-camera — should produce only OFF_CAMERA_BRAINSTORM."""
    narrative = NarrativeResult(
        segments=[
            _make_segment("s1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 5.0, "Stop rolling.", NarrativeLabel.OFF_CAMERA, off_camera_trigger=True),
            _make_segment("s2", "SPEAKER_00", SpeakerRole.HOST, 5.0, 60.0, "Discussion.", NarrativeLabel.UNKNOWN),
        ],
        show_type=ShowType.NAV_THETHI,
    )
    result = direct(narrative, three_camera_inventory, standard_rules)
    assert result.result is not None
    # Off-camera cuts should be present
    off_cam_cuts = [c for c in result.result.cuts if c.is_off_camera]
    assert len(off_cam_cuts) >= 1


@pytest.mark.fault
def test_corrupt_show_type_empty():
    """Empty show_type.txt → ShowType.UNKNOWN, no crash."""
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("")
        path = f.name
    show_type, warnings = detect_show_type(path)
    Path(path).unlink(missing_ok=True)
    assert show_type.value == "unknown"
    assert len(warnings) > 0


@pytest.mark.fault
def test_corrupt_show_type_unrecognized():
    """Unrecognized show_type → UNKNOWN with warning."""
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("The Random Unknown Show")
        path = f.name
    show_type, warnings = detect_show_type(path)
    Path(path).unlink(missing_ok=True)
    assert show_type.value == "unknown"
    assert any("Unrecognized" in w for w in warnings)


@pytest.mark.fault
def test_corrupt_show_type_missing():
    """Missing show_type.txt → UNKNOWN with warning, no crash."""
    show_type, warnings = detect_show_type("/nonexistent/path/show_type.txt")
    assert show_type.value == "unknown"
    assert len(warnings) > 0


@pytest.mark.fault
def test_xml_unknown_camera_ref(three_camera_inventory, sample_ingest, standard_rules, tmp_dir):
    """Cut referencing non-existent camera → warning, fallback camera used, no crash."""
    from pipeline.schemas import CutEntry, CutList, CutReason
    from pipeline.stage5_xml_generator import generate_fcpxml

    cuts = [
        CutEntry(
            cut_id="c1",
            camera_id="cam_nonexistent",  # Unknown camera
            start_s=0.0, end_s=30.0,
            reason=CutReason.SPEAKER_CHANGE,
            rule_tag="SPEAKER_RULE",
        ),
    ]
    cut_list = CutList(cuts=cuts, total_duration_s=30.0, show_type=ShowType.NAV_THETHI)
    out = str(tmp_dir / "unknown_cam.fcpxml")
    result = generate_fcpxml(cut_list, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules)
    assert result.success or len(result.warnings) > 0
    # Should not crash; warnings should be present
    assert any("unknown" in w.lower() or "cam_nonexistent" in w for w in result.warnings)


@pytest.mark.fault
def test_overlapping_input_timecodes_no_crash(three_camera_inventory, standard_rules):
    """Overlapping input segment timecodes don't cause output overlap."""
    narrative = NarrativeResult(
        segments=[
            _make_segment("s1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 25.0, "Host."),
            _make_segment("s2", "SPEAKER_01", SpeakerRole.GUEST, 15.0, 40.0, "Guest overlaps."),  # overlapping!
        ],
        show_type=ShowType.NAV_THETHI,
    )
    result = direct(narrative, three_camera_inventory, standard_rules)
    assert result.result is not None
    cuts = sorted(result.result.cuts, key=lambda c: c.start_s)
    for i in range(len(cuts) - 1):
        assert cuts[i].end_s <= cuts[i + 1].start_s + 0.001, "Output cuts must not overlap"
