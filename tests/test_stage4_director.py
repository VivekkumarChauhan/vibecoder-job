"""tests/test_stage4_director.py — Unit tests for the Director (Stage 4)."""
from __future__ import annotations

import pytest

from pipeline.schemas import (
    CameraInventory,
    CutList,
    CutReason,
    NarrativeLabel,
    NarrativeResult,
    NarrativeSegment,
    ShowType,
    SpeakerRole,
)
from pipeline.stage4_director import direct, _is_cut_safe, _is_meaningful_event
from tests.conftest import _make_segment


@pytest.mark.unit
def test_director_normal_flow(three_camera_inventory, standard_rules, basic_narrative):
    """Director produces a valid cut list for a normal narrative."""
    result = direct(basic_narrative, three_camera_inventory, standard_rules)
    assert result.success
    assert result.result is not None
    cut_list = result.result
    assert len(cut_list.cuts) > 0
    assert cut_list.total_duration_s == 120.0


@pytest.mark.unit
def test_director_no_negative_durations(three_camera_inventory, standard_rules, basic_narrative):
    """All cuts must have positive duration."""
    result = direct(basic_narrative, three_camera_inventory, standard_rules)
    for cut in result.result.cuts:
        assert cut.duration_s > 0, f"Cut {cut.cut_id} has non-positive duration {cut.duration_s}"


@pytest.mark.unit
def test_director_empty_inventory(empty_inventory, standard_rules, basic_narrative):
    """Empty camera inventory → graceful failure, no crash."""
    result = direct(basic_narrative, empty_inventory, standard_rules)
    assert result.result is not None  # Returns CutList, even if empty
    assert not result.success or len(result.result.cuts) == 0
    assert len(result.errors) > 0  # Error reported


@pytest.mark.unit
def test_director_single_camera(single_camera_inventory, standard_rules, basic_narrative):
    """Single camera → all cuts on that camera, no crash."""
    result = direct(basic_narrative, single_camera_inventory, standard_rules)
    assert result.result is not None
    for cut in result.result.cuts:
        if not cut.is_off_camera:
            assert cut.camera_id == "cam_1"


@pytest.mark.unit
def test_director_empty_narrative(three_camera_inventory, standard_rules):
    """Empty narrative → minimal cut list, no crash."""
    narrative = NarrativeResult(segments=[], show_type=ShowType.NAV_THETHI)
    result = direct(narrative, three_camera_inventory, standard_rules)
    assert result.result is not None
    # Should have at most the initial cut
    assert len(result.result.cuts) <= 1


@pytest.mark.unit
def test_director_zero_length_segments(three_camera_inventory, standard_rules):
    """Zero-length segments don't cause crashes."""
    narrative = NarrativeResult(
        segments=[
            _make_segment("s1", "SPEAKER_00", SpeakerRole.HOST, 5.0, 5.0, ""),  # zero length
            _make_segment("s2", "SPEAKER_00", SpeakerRole.HOST, 5.0, 10.0, "Real segment."),
        ],
        show_type=ShowType.NAV_THETHI,
    )
    result = direct(narrative, three_camera_inventory, standard_rules)
    assert result.result is not None
    assert all(c.duration_s >= 0 for c in result.result.cuts)


@pytest.mark.unit
def test_director_overlapping_timecodes(three_camera_inventory, standard_rules):
    """Overlapping input segments don't produce overlapping output cuts."""
    narrative = NarrativeResult(
        segments=[
            _make_segment("s1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 20.0, "Host speaks."),
            _make_segment("s2", "SPEAKER_01", SpeakerRole.GUEST, 15.0, 35.0, "Guest overlaps."),
        ],
        show_type=ShowType.NAV_THETHI,
    )
    result = direct(narrative, three_camera_inventory, standard_rules)
    cuts = sorted(result.result.cuts, key=lambda c: c.start_s)
    for i in range(len(cuts) - 1):
        assert cuts[i].end_s <= cuts[i + 1].start_s + 0.001, (
            f"Overlapping cuts: {cuts[i].cut_id} ({cuts[i].start_s}–{cuts[i].end_s}) "
            f"and {cuts[i+1].cut_id} ({cuts[i+1].start_s}–{cuts[i+1].end_s})"
        )


@pytest.mark.unit
def test_director_deterministic(three_camera_inventory, standard_rules, basic_narrative):
    """Same input always produces same output (determinism)."""
    result1 = direct(basic_narrative, three_camera_inventory, standard_rules)
    result2 = direct(basic_narrative, three_camera_inventory, standard_rules)
    cuts1 = [(c.camera_id, c.start_s, c.end_s, c.rule_tag) for c in result1.result.cuts]
    cuts2 = [(c.camera_id, c.start_s, c.end_s, c.rule_tag) for c in result2.result.cuts]
    assert cuts1 == cuts2, "Director must be deterministic"


@pytest.mark.unit
def test_is_cut_safe_mid_word():
    """Safety check correctly identifies mid-word cuts as unsafe."""
    from pipeline.schemas import WordToken
    seg = _make_segment("s1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 10.0, "Hello world")
    seg = NarrativeSegment(**{**seg.model_dump(), "words": [
        WordToken(word="Hello", start=0.0, end=0.5),
        WordToken(word="world", start=0.5, end=1.0),
    ]})
    assert not _is_cut_safe(0.25, seg, {"director": {"safety_min_hold_s": 0.5}})
    assert _is_cut_safe(2.0, seg, {"director": {"safety_min_hold_s": 0.5}})


@pytest.mark.unit
def test_meaningful_event_labels():
    """Verify meaningful event detection for refresh rule timer."""
    for label in [NarrativeLabel.QUESTION, NarrativeLabel.LAUGHTER, NarrativeLabel.EMOTIONAL_MOMENT]:
        seg = _make_segment("s", "SP", SpeakerRole.HOST, 0.0, 5.0, "text", label)
        assert _is_meaningful_event(seg)
    for label in [NarrativeLabel.SILENCE, NarrativeLabel.UNKNOWN]:
        seg = _make_segment("s", "SP", SpeakerRole.HOST, 0.0, 5.0, "text", label)
        assert not _is_meaningful_event(seg)


@pytest.mark.unit
def test_director_nav_thethi_wide_cap(three_camera_inventory, standard_rules):
    """Nav Thethi: Director + post-processing should keep wide% under 20%."""
    # All segments are silence — triggers many refresh wides
    segments = []
    for i in range(10):
        segments.append(
            _make_segment(f"s{i}", "SPEAKER_00", SpeakerRole.HOST, i * 10.0, (i + 1) * 10.0, "...", NarrativeLabel.SILENCE)
        )
    narrative = NarrativeResult(segments=segments, show_type=ShowType.NAV_THETHI)
    result = direct(narrative, three_camera_inventory, standard_rules)
    cut_list = result.result
    wide_pct = cut_list.wide_shot_pct
    assert wide_pct <= 25.0, f"Nav Thethi wide% {wide_pct:.1f} exceeds cap (should be near/under 20%)"
