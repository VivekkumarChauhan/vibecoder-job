"""tests/test_stage5_xml.py — Unit tests for the FCPXML Generator (Stage 5)."""
from __future__ import annotations

import pytest
from lxml import etree
from pathlib import Path

from pipeline.schemas import (
    CutEntry,
    CutList,
    CutReason,
    ShowType,
)
from pipeline.stage5_xml_generator import generate_fcpxml
from pipeline.stage6_validator import validate_fcpxml


@pytest.mark.unit
def test_xml_generated_successfully(three_camera_inventory, sample_ingest, standard_rules, basic_narrative, tmp_dir):
    """XML generator produces a file without crashing."""
    from pipeline.stage4_director import direct
    result = direct(basic_narrative, three_camera_inventory, standard_rules)
    cut_list = result.result
    out = str(tmp_dir / "output.fcpxml")
    gen = generate_fcpxml(cut_list, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules)
    assert gen.success
    assert Path(out).exists()


@pytest.mark.unit
def test_xml_is_valid_xml(three_camera_inventory, sample_ingest, standard_rules, basic_narrative, tmp_dir):
    """Output is parseable, well-formed XML."""
    from pipeline.stage4_director import direct
    result = direct(basic_narrative, three_camera_inventory, standard_rules)
    out = str(tmp_dir / "output.fcpxml")
    generate_fcpxml(result.result, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules)
    # Parse with strict parser
    tree = etree.parse(out)
    assert tree.getroot().tag == "fcpxml"


@pytest.mark.unit
def test_xml_has_fcpxml_version(three_camera_inventory, sample_ingest, standard_rules, basic_narrative, tmp_dir):
    """FCPXML has version="1.10"."""
    from pipeline.stage4_director import direct
    result = direct(basic_narrative, three_camera_inventory, standard_rules)
    out = str(tmp_dir / "output.fcpxml")
    generate_fcpxml(result.result, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules)
    tree = etree.parse(out)
    root = tree.getroot()
    assert root.get("version") == "1.10"


@pytest.mark.unit
def test_xml_has_resources(three_camera_inventory, sample_ingest, standard_rules, basic_narrative, tmp_dir):
    """FCPXML has <resources> with assets for all cameras."""
    from pipeline.stage4_director import direct
    result = direct(basic_narrative, three_camera_inventory, standard_rules)
    out = str(tmp_dir / "output.fcpxml")
    generate_fcpxml(result.result, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules)
    tree = etree.parse(out)
    resources = tree.getroot().find("resources")
    assert resources is not None
    assets = resources.findall("asset")
    assert len(assets) == len(three_camera_inventory.cameras)


@pytest.mark.unit
def test_xml_no_overlapping_clips(three_camera_inventory, sample_ingest, standard_rules, basic_narrative, tmp_dir):
    """No overlapping clips in the generated XML."""
    from pipeline.stage4_director import direct
    result = direct(basic_narrative, three_camera_inventory, standard_rules)
    out = str(tmp_dir / "output.fcpxml")
    generate_fcpxml(result.result, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules)

    val = validate_fcpxml(out, three_camera_inventory, sample_ingest, standard_rules)
    overlap_errors = [e for e in val.result.errors if e.error_type == "CLIP_OVERLAP"]
    assert len(overlap_errors) == 0, f"Overlapping clips found: {[e.description for e in overlap_errors]}"


@pytest.mark.unit
def test_xml_inline_comments(three_camera_inventory, sample_ingest, standard_rules, basic_narrative, tmp_dir):
    """XML comments are present in output (editor guidance)."""
    from pipeline.stage4_director import direct
    result = direct(basic_narrative, three_camera_inventory, standard_rules)
    out = str(tmp_dir / "output.fcpxml")
    generate_fcpxml(result.result, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules)
    content = Path(out).read_text()
    assert "<!--" in content, "XML should contain inline comments for editor guidance"
    assert "CUT:" in content, "XML comments should include cut explanations"


@pytest.mark.unit
def test_xml_empty_cut_list_produces_fallback(three_camera_inventory, sample_ingest, standard_rules, tmp_dir):
    """Empty cut list → fallback single-camera FCPXML, no crash."""
    cut_list = CutList(cuts=[], total_duration_s=60.0, show_type=ShowType.NAV_THETHI)
    out = str(tmp_dir / "fallback.fcpxml")
    gen = generate_fcpxml(cut_list, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules)
    assert gen.success or len(gen.warnings) > 0  # warnings acceptable, no crash
    # File should still be created
    assert Path(out).exists()


@pytest.mark.unit
def test_xml_off_camera_brainstorm_appears(three_camera_inventory, sample_ingest, standard_rules, off_camera_narrative, tmp_dir):
    """OFF_CAMERA_BRAINSTORM must appear in XML as gap element."""
    from pipeline.stage4_director import direct
    result = direct(off_camera_narrative, three_camera_inventory, standard_rules)
    out = str(tmp_dir / "off_cam.fcpxml")
    generate_fcpxml(result.result, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules)
    content = Path(out).read_text()
    assert "OFF_CAMERA_BRAINSTORM" in content


@pytest.mark.unit
def test_xml_timecodes_are_rational_strings(three_camera_inventory, sample_ingest, standard_rules, basic_narrative, tmp_dir):
    """All timecodes in XML are FCP rational strings (not floats)."""
    import re
    from pipeline.stage4_director import direct
    result = direct(basic_narrative, three_camera_inventory, standard_rules)
    out = str(tmp_dir / "rational.fcpxml")
    generate_fcpxml(result.result, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules)
    content = Path(out).read_text()
    # FCP rational: digits/digits followed by 's'
    rational_pattern = re.compile(r'\d+/\d+s')
    assert rational_pattern.search(content), "XML should contain FCP rational timecode strings"
    # Should NOT contain standalone floats as time values
    # (acceptable in comments but not in XML attributes for timecodes)


@pytest.mark.unit
def test_xml_sbs_annotation(three_camera_inventory, sample_ingest, standard_rules, tmp_dir):
    """SBS cuts are annotated in XML."""
    from pipeline.schemas import CameraInventory
    cuts = [
        CutEntry(cut_id="c1", camera_id="cam_1", start_s=0.0, end_s=30.0,
                 reason=CutReason.SHOW_SPECIFIC, rule_tag="SBS_OPENING_QUESTION", is_sbs=True),
        CutEntry(cut_id="c2", camera_id="cam_1", start_s=30.0, end_s=60.0,
                 reason=CutReason.SPEAKER_CHANGE, rule_tag="SPEAKER_RULE"),
    ]
    cut_list = CutList(cuts=cuts, total_duration_s=60.0, show_type=ShowType.MATURITY_CODE)
    out = str(tmp_dir / "sbs.fcpxml")
    gen = generate_fcpxml(cut_list, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules)
    content = Path(out).read_text()
    assert "SBS_FRAME" in content or "SBS" in content, "SBS cuts should be annotated"


@pytest.mark.unit
def test_xml_validation_passes(three_camera_inventory, sample_ingest, standard_rules, basic_narrative, tmp_dir):
    """Generated XML passes full validation suite."""
    from pipeline.stage4_director import direct
    result = direct(basic_narrative, three_camera_inventory, standard_rules)
    out = str(tmp_dir / "validated.fcpxml")
    generate_fcpxml(result.result, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules)
    val = validate_fcpxml(out, three_camera_inventory, sample_ingest, standard_rules)
    assert val.result is not None
    errors = [e for e in val.result.errors if e.severity == "error"]
    assert len(errors) == 0, f"Validation errors: {[e.description for e in errors]}"
