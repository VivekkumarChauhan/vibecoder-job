"""tests/test_stage0_ingest.py — Unit tests for Stage 0: Ingest."""
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.mark.unit
def test_detect_show_type_nav(show_type_nav):
    from pipeline.schemas import ShowType
    from pipeline.stage0_ingest import detect_show_type
    show_type, warnings = detect_show_type(show_type_nav)
    assert show_type == ShowType.NAV_THETHI
    assert len(warnings) == 0


@pytest.mark.unit
def test_detect_show_type_maturity(show_type_maturity):
    from pipeline.schemas import ShowType
    from pipeline.stage0_ingest import detect_show_type
    show_type, warnings = detect_show_type(show_type_maturity)
    assert show_type == ShowType.MATURITY_CODE
    assert len(warnings) == 0


@pytest.mark.unit
def test_detect_show_type_empty(show_type_empty):
    from pipeline.schemas import ShowType
    from pipeline.stage0_ingest import detect_show_type
    show_type, warnings = detect_show_type(show_type_empty)
    assert show_type == ShowType.UNKNOWN
    assert len(warnings) > 0


@pytest.mark.unit
def test_detect_show_type_corrupt(show_type_corrupt):
    from pipeline.schemas import ShowType
    from pipeline.stage0_ingest import detect_show_type
    show_type, warnings = detect_show_type(show_type_corrupt)
    assert show_type == ShowType.UNKNOWN
    assert len(warnings) > 0


@pytest.mark.unit
def test_detect_show_type_missing_file():
    from pipeline.schemas import ShowType
    from pipeline.stage0_ingest import detect_show_type
    show_type, warnings = detect_show_type("/nonexistent/path.txt")
    assert show_type == ShowType.UNKNOWN
    assert len(warnings) > 0


@pytest.mark.unit
def test_ingest_missing_source(tmp_path):
    from utils.cache import StageCache
    from pipeline.stage0_ingest import ingest_syncmaster
    cache = StageCache(str(tmp_path / "cache"))
    result = ingest_syncmaster("/nonexistent.mp4", "/fake/show_type.txt", cache=cache)
    assert not result.success
    assert len(result.errors) > 0


"""tests/test_stage1_camera.py — Unit tests for Stage 1: Camera Discovery."""


@pytest.mark.unit
def test_camera_inventory_get_by_role(three_camera_inventory):
    from pipeline.schemas import CameraRole
    host_cam = three_camera_inventory.get_camera_by_role(CameraRole.HOST_HERO)
    assert host_cam is not None
    assert host_cam.camera_id == "cam_1"

    wide_cam = three_camera_inventory.get_camera_by_role(CameraRole.WIDE)
    assert wide_cam is not None
    assert wide_cam.camera_id == "cam_3"

    guest_cam = three_camera_inventory.get_camera_by_role(CameraRole.GUEST_HERO)
    assert guest_cam is not None
    assert guest_cam.camera_id == "cam_2"


@pytest.mark.unit
def test_camera_inventory_get_valid_cameras(frozen_camera_inventory):
    valid = frozen_camera_inventory.get_valid_cameras()
    assert len(valid) == 1
    assert valid[0].camera_id == "cam_2"


@pytest.mark.unit
def test_single_camera_inventory_valid(single_camera_inventory):
    assert single_camera_inventory.total_cameras == 1
    assert len(single_camera_inventory.get_valid_cameras()) == 1


"""tests/test_stage2_speaker.py — Unit tests for Stage 2: Speaker Mapping."""


@pytest.mark.unit
def test_speaker_mapping_host_guest(basic_speaker_result):
    mapping = basic_speaker_result.mapping
    assert mapping.host_label == "SPEAKER_00"
    assert mapping.guest_label == "SPEAKER_01"


@pytest.mark.unit
def test_transcript_speaker_roles_assigned(basic_speaker_result):
    from pipeline.schemas import SpeakerRole
    for seg in basic_speaker_result.transcript:
        # All speakers should map to host or guest
        assert seg.speaker_label in ["SPEAKER_00", "SPEAKER_01"]


@pytest.mark.unit
def test_groq_unavailable_uses_heuristic(basic_speaker_result, standard_rules):
    """When Groq is None, speaker mapping uses speaking-time heuristic."""
    from pipeline.stage2_speaker_mapping import _map_speaker_roles_groq
    merged = [s.model_dump() for s in basic_speaker_result.transcript]
    mapping, warnings = _map_speaker_roles_groq(merged, 300, groq_client=None)
    assert mapping.host_label is not None
    assert any("heuristic" in w.lower() or "groq" in w.lower() for w in warnings)


"""tests/test_stage3_narrative.py — Unit tests for Stage 3: Narrative Understanding."""


@pytest.mark.unit
def test_narrative_off_camera_detection(off_camera_narrative):
    off_cam_segs = [s for s in off_camera_narrative.segments if s.off_camera_trigger]
    assert len(off_cam_segs) >= 1


@pytest.mark.unit
def test_narrative_resume_detection(off_camera_narrative):
    resume_segs = [s for s in off_camera_narrative.segments if s.resume_trigger]
    assert len(resume_segs) >= 1


@pytest.mark.unit
def test_narrative_physical_event_detection(physical_adjustment_narrative):
    phy_segs = [s for s in physical_adjustment_narrative.segments if s.physical_event]
    assert len(phy_segs) >= 1
    assert phy_segs[0].physical_event == "mic_adjust"


@pytest.mark.unit
def test_narrative_heuristic_labels(basic_speaker_result, standard_rules):
    from pipeline.schemas import ShowType
    from pipeline.stage3_narrative import understand_narrative
    result = understand_narrative(basic_speaker_result, ShowType.NAV_THETHI, standard_rules, groq_client=None)
    assert result.result is not None
    assert len(result.result.segments) == len(basic_speaker_result.transcript)
    for seg in result.result.segments:
        assert seg.label is not None  # Every segment gets a label


@pytest.mark.unit
def test_narrative_handles_empty_transcript(standard_rules):
    from pipeline.schemas import ShowType, SpeakerMapping, SpeakerMapResult
    from pipeline.stage3_narrative import understand_narrative
    empty_result = SpeakerMapResult(
        mapping=SpeakerMapping(host_label="SPEAKER_00", all_labels=["SPEAKER_00"]),
        transcript=[],
    )
    result = understand_narrative(empty_result, ShowType.NAV_THETHI, standard_rules, groq_client=None)
    assert result.result is not None
    assert len(result.result.segments) == 0


"""tests/test_stage4b_critic.py — Unit tests for Stage 4b: Critic."""


@pytest.mark.unit
def test_critic_passes_clean_cut_list(three_camera_inventory, basic_narrative, standard_rules):
    from pipeline.stage4_director import direct
    from pipeline.stage4b_critic import critique
    dir_result = direct(basic_narrative, three_camera_inventory, standard_rules)
    crit_result = critique(dir_result.result, basic_narrative, three_camera_inventory, standard_rules)
    assert crit_result.result is not None
    errors = [v for v in crit_result.result.violations if v.severity == "error"]
    # A clean nav thethi narrative should have no errors
    assert len(errors) == 0, f"Unexpected errors: {[v.description for v in errors]}"


@pytest.mark.unit
def test_critic_detects_overlap():
    from pipeline.schemas import CutEntry, CutList, NarrativeResult, ShowType
    from pipeline.stage4b_critic import critique, _check_overlapping_cuts

    cuts = [
        CutEntry(cut_id="c1", camera_id="cam_1", start_s=0.0, end_s=15.0, reason="speaker_change", rule_tag="SPEAKER_RULE"),
        CutEntry(cut_id="c2", camera_id="cam_1", start_s=10.0, end_s=25.0, reason="speaker_change", rule_tag="SPEAKER_RULE"),
    ]
    violations = _check_overlapping_cuts(cuts)
    assert len(violations) >= 1
    assert any("CLIP_OVERLAP" in v.rule or "TIMELINE_OVERLAP" in v.rule for v in violations)


@pytest.mark.unit
def test_critic_quality_score_range(three_camera_inventory, basic_narrative, standard_rules):
    from pipeline.stage4_director import direct
    from pipeline.stage4b_critic import critique
    dir_result = direct(basic_narrative, three_camera_inventory, standard_rules)
    crit_result = critique(dir_result.result, basic_narrative, three_camera_inventory, standard_rules)
    score = crit_result.result.quality_score
    assert 0.0 <= score <= 1.0, f"Quality score {score} out of range [0,1]"
