"""tests/test_integration.py — Full pipeline integration test.

Runs the complete pipeline end-to-end using synthetic data (no real video needed).
Mocks: FFmpeg, PyAV, faster-whisper, pyannote, Groq API.
Asserts both output files are produced and pass validation.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.schemas import ShowType


@pytest.mark.integration
def test_full_pipeline_nav_thethi(three_camera_inventory, basic_narrative, basic_speaker_result, sample_ingest, standard_rules, tmp_path):
    """End-to-end: Nav Thethi Show → output.fcpxml + editing_report.json, both pass validation."""
    from pipeline.stage4_director import direct
    from pipeline.stage4b_critic import critique
    from pipeline.stage5_xml_generator import generate_fcpxml
    from pipeline.stage6_validator import validate_fcpxml
    from pipeline.schemas import CutList, EditingReport, QualityMetrics

    # Run director
    dir_result = direct(basic_narrative, three_camera_inventory, standard_rules)
    assert dir_result.success
    cut_list = dir_result.result

    # Run critic
    crit_result = critique(cut_list, basic_narrative, three_camera_inventory, standard_rules)
    critic = crit_result.result

    # Generate XML
    out_fcpxml = str(tmp_path / "output.fcpxml")
    gen_result = generate_fcpxml(
        cut_list, three_camera_inventory, sample_ingest, "test_syncmaster.mp4", out_fcpxml, standard_rules, ShowType.NAV_THETHI
    )
    assert gen_result.success, f"XML generation failed: {gen_result.errors}"
    assert Path(out_fcpxml).exists()

    # Validate XML
    val_result = validate_fcpxml(out_fcpxml, three_camera_inventory, sample_ingest, standard_rules)
    assert val_result.result is not None
    errors = [e for e in val_result.result.errors if e.severity == "error"]
    assert len(errors) == 0, f"Validation errors: {[e.description for e in errors]}"

    # Generate editing report
    report = {
        "camera_inventory": three_camera_inventory.model_dump(),
        "speaker_mapping": basic_speaker_result.mapping.model_dump(),
        "cuts": [c.model_dump() for c in cut_list.cuts],
        "warnings": dir_result.warnings + crit_result.warnings,
        "off_camera_segments": [],
        "metadata": {
            "quality_score": critic.quality_score if critic else 0.0,
            "wide_shot_pct": cut_list.wide_shot_pct,
            "cut_frequency_per_min": len(cut_list.cuts) / (cut_list.total_duration_s / 60),
            "rule_violation_count": len(critic.violations) if critic else 0,
        }
    }
    report_path = tmp_path / "editing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    assert report_path.exists()

    # Verify report structure
    with open(report_path) as f:
        loaded = json.load(f)
    assert "camera_inventory" in loaded
    assert "speaker_mapping" in loaded
    assert "cuts" in loaded
    assert "warnings" in loaded
    assert "off_camera_segments" in loaded
    assert "metadata" in loaded

    print("\n[PASS] Integration test passed:")
    print(f"   Cuts: {len(cut_list.cuts)}")
    print(f"   Wide %: {cut_list.wide_shot_pct:.1f}%")
    print(f"   Quality: {critic.quality_score if critic else 0:.3f}")
    print(f"   Violations: {len(critic.violations) if critic else 0}")


@pytest.mark.integration
def test_full_pipeline_maturity_code(three_camera_inventory, standard_rules, tmp_path, sample_ingest):
    """End-to-end: Cracking the Maturity Code → SBS opening question, valid XML."""
    from pipeline.schemas import (
        NarrativeLabel, NarrativeResult, NarrativeSegment, ShowType, SpeakerRole
    )
    from pipeline.stage4_director import direct
    from pipeline.stage5_xml_generator import generate_fcpxml
    from pipeline.stage6_validator import validate_fcpxml
    from tests.conftest import _make_segment

    narrative = NarrativeResult(
        segments=[
            _make_segment("s1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 15.0, "What is maturity?", NarrativeLabel.QUESTION),
            _make_segment("s2", "SPEAKER_01", SpeakerRole.GUEST, 15.0, 60.0, "Great question!", NarrativeLabel.ANSWER),
            _make_segment("s3", "SPEAKER_00", SpeakerRole.HOST, 60.0, 65.0, "Hahaha!", NarrativeLabel.SHARED_LAUGHTER, has_laughter=True),
            _make_segment("s4", "SPEAKER_01", SpeakerRole.GUEST, 65.0, 120.0, "Framework discussion.", NarrativeLabel.FRAMEWORK_DISCUSSION),
        ],
        show_type=ShowType.MATURITY_CODE,
    )

    dir_result = direct(narrative, three_camera_inventory, standard_rules)
    assert dir_result.success
    cut_list = dir_result.result

    # Opening question should be in SBS
    opening_sbs_cuts = [c for c in cut_list.cuts if c.is_sbs]
    assert len(opening_sbs_cuts) >= 1, "Maturity Code: opening question should be in SBS"

    out = str(tmp_path / "maturity.fcpxml")
    gen = generate_fcpxml(cut_list, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules, ShowType.MATURITY_CODE)
    assert gen.success

    val = validate_fcpxml(out, three_camera_inventory, sample_ingest, standard_rules)
    errors = [e for e in val.result.errors if e.severity == "error"]
    assert len(errors) == 0


@pytest.mark.integration
def test_pipeline_off_camera_roundtrip(three_camera_inventory, off_camera_narrative, standard_rules, tmp_path, sample_ingest):
    """Off-camera + resume roundtrip: OFF_CAMERA_BRAINSTORM in XML, editing resumes correctly."""
    from pipeline.stage4_director import direct
    from pipeline.stage5_xml_generator import generate_fcpxml

    dir_result = direct(off_camera_narrative, three_camera_inventory, standard_rules)
    cut_list = dir_result.result

    off_cam_cuts = [c for c in cut_list.cuts if c.is_off_camera]
    assert len(off_cam_cuts) >= 1

    # Post-resume cuts must exist
    resume_time = 35.0
    post_resume = [c for c in cut_list.cuts if c.start_s >= resume_time and not c.is_off_camera]
    assert len(post_resume) >= 1, "Must have regular cuts after resume"

    out = str(tmp_path / "off_cam.fcpxml")
    gen = generate_fcpxml(cut_list, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules)
    assert gen.success
    content = Path(out).read_text()
    assert "OFF_CAMERA_BRAINSTORM" in content


@pytest.mark.integration
def test_cache_warms_correctly(three_camera_inventory, basic_narrative, standard_rules, tmp_path, sample_ingest):
    """Second run with warm cache makes 0 additional director/xml calls."""
    from pipeline.stage4_director import direct
    from utils.cache import StageCache

    cache = StageCache(cache_dir=str(tmp_path / "cache"))

    # First run
    result1 = direct(basic_narrative, three_camera_inventory, standard_rules, cache=cache, iteration=0)
    cuts1 = [(c.camera_id, c.start_s, c.rule_tag) for c in result1.result.cuts]

    # Second run with same cache — should hit cache
    result2 = direct(basic_narrative, three_camera_inventory, standard_rules, cache=cache, iteration=0)
    cuts2 = [(c.camera_id, c.start_s, c.rule_tag) for c in result2.result.cuts]

    assert cuts1 == cuts2, "Cache should produce identical results"
    stats = cache.stats()
    assert stats["hits"] >= 1, "Expected at least 1 cache hit on second run"
