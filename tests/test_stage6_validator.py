"""tests/test_stage6_validator.py — Unit tests for the FCPXML Validator (Stage 6)."""
from __future__ import annotations

import pytest
from pathlib import Path


def _write_xml(path: str, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


MINIMAL_VALID_FCPXML = """<?xml version='1.0' encoding='UTF-8'?>
<fcpxml version="1.10">
  <resources>
    <format id="f_cam_1" frameDuration="1001/30000s" width="1920" height="1080"/>
    <asset id="r_cam_1" name="cam_1" src="file:///test.mp4" start="0s"
           duration="3600000000/254016000000s" hasVideo="1" hasAudio="1"
           audioSources="1" audioChannels="2" audioRate="48000" format="f_cam_1">
      <media-rep kind="original-media" src="file:///test.mp4"/>
    </asset>
  </resources>
  <library location="file:///output/">
    <event name="AI_Cut">
      <project name="AI_Rough_Cut">
        <sequence format="f_cam_1" duration="3600000000/254016000000s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
          <spine>
            <!-- CUT: cam_1 | Rule: SPEAKER_RULE -->
            <clip name="cam_1_00_00_00_000" offset="0s" duration="3600000000/254016000000s" start="0s" format="f_cam_1">
              <video offset="0s" ref="r_cam_1" duration="3600000000/254016000000s" start="0s"/>
              <audio offset="0s" ref="r_cam_1" duration="3600000000/254016000000s" start="0s" srcCh="1, 2" role="dialogue"/>
            </clip>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>"""

OVERLAPPING_FCPXML = """<?xml version='1.0' encoding='UTF-8'?>
<fcpxml version="1.10">
  <resources>
    <format id="f_cam_1" frameDuration="1001/30000s" width="1920" height="1080"/>
    <asset id="r_cam_1" name="cam_1" src="file:///test.mp4" start="0s"
           duration="12700800000000/254016000000s" hasVideo="1" hasAudio="1"
           audioSources="1" audioChannels="2" audioRate="48000" format="f_cam_1">
      <media-rep kind="original-media" src="file:///test.mp4"/>
    </asset>
  </resources>
  <library location="file:///output/">
    <event name="AI_Cut">
      <project name="AI_Rough_Cut">
        <sequence format="f_cam_1" duration="12700800000000/254016000000s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
          <spine>
            <clip name="clip_a" offset="0s" duration="6350400000000/254016000000s" start="0s" format="f_cam_1">
              <video offset="0s" ref="r_cam_1" duration="6350400000000/254016000000s" start="0s"/>
            </clip>
            <!-- Overlaps with clip_a -->
            <clip name="clip_b" offset="5080320000000/254016000000s" duration="7620480000000/254016000000s" start="0s" format="f_cam_1">
              <video offset="0s" ref="r_cam_1" duration="7620480000000/254016000000s" start="0s"/>
            </clip>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>"""

BAD_ASSET_REF_FCPXML = """<?xml version='1.0' encoding='UTF-8'?>
<fcpxml version="1.10">
  <resources>
    <format id="f_cam_1" frameDuration="1001/30000s" width="1920" height="1080"/>
    <asset id="r_cam_1" name="cam_1" src="file:///test.mp4" start="0s"
           duration="3600000000/254016000000s" hasVideo="1" hasAudio="1"
           audioSources="1" audioChannels="2" audioRate="48000" format="f_cam_1">
      <media-rep kind="original-media" src="file:///test.mp4"/>
    </asset>
  </resources>
  <library location="file:///output/">
    <event name="AI_Cut">
      <project name="AI_Rough_Cut">
        <sequence format="f_cam_1" duration="3600000000/254016000000s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
          <spine>
            <clip name="clip_a" offset="0s" duration="3600000000/254016000000s" start="0s" format="f_cam_1">
              <video offset="0s" ref="r_cam_999" duration="3600000000/254016000000s" start="0s"/>
            </clip>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>"""


@pytest.mark.unit
def test_validator_valid_fcpxml(tmp_dir):
    """Valid FCPXML passes validation."""
    from pipeline.stage6_validator import validate_fcpxml
    p = str(tmp_dir / "valid.fcpxml")
    _write_xml(p, MINIMAL_VALID_FCPXML)
    result = validate_fcpxml(p)
    assert result.result is not None
    errors = [e for e in result.result.errors if e.severity == "error"]
    assert len(errors) == 0, f"Valid FCPXML should pass: {[e.description for e in errors]}"


@pytest.mark.unit
def test_validator_detects_overlaps(tmp_dir):
    """Overlapping clips are detected as errors."""
    from pipeline.stage6_validator import validate_fcpxml
    p = str(tmp_dir / "overlap.fcpxml")
    _write_xml(p, OVERLAPPING_FCPXML)
    result = validate_fcpxml(p)
    assert result.result is not None
    overlap_errors = [e for e in result.result.errors if e.error_type == "CLIP_OVERLAP"]
    assert len(overlap_errors) >= 1, "Should detect overlapping clips"


@pytest.mark.unit
def test_validator_detects_bad_asset_refs(tmp_dir):
    """Missing asset refs are detected."""
    from pipeline.stage6_validator import validate_fcpxml
    p = str(tmp_dir / "bad_ref.fcpxml")
    _write_xml(p, BAD_ASSET_REF_FCPXML)
    result = validate_fcpxml(p)
    assert result.result is not None
    ref_errors = [e for e in result.result.errors if e.error_type == "MISSING_ASSET_REF"]
    assert len(ref_errors) >= 1, "Should detect missing asset ref r_cam_999"


@pytest.mark.unit
def test_validator_file_not_found(tmp_dir):
    """Non-existent file → graceful failure, not crash."""
    from pipeline.stage6_validator import validate_fcpxml
    result = validate_fcpxml(str(tmp_dir / "does_not_exist.fcpxml"))
    assert result.result is not None
    assert not result.result.passed
    assert any("FILE_NOT_FOUND" in e.error_type for e in result.result.errors)


@pytest.mark.unit
def test_validator_malformed_xml(tmp_dir):
    """Malformed XML → graceful failure, not crash."""
    from pipeline.stage6_validator import validate_fcpxml
    p = str(tmp_dir / "malformed.fcpxml")
    _write_xml(p, "<fcpxml version='1.10'><unclosed>")
    result = validate_fcpxml(p)
    assert result.result is not None
    assert not result.result.passed


@pytest.mark.unit
def test_validator_returns_itemized_errors(tmp_dir):
    """Validator returns specific, itemized errors (not just true/false)."""
    from pipeline.stage6_validator import validate_fcpxml
    p = str(tmp_dir / "bad_ref2.fcpxml")
    _write_xml(p, BAD_ASSET_REF_FCPXML)
    result = validate_fcpxml(p)
    # Each error should have error_type and description
    for err in result.result.errors:
        assert err.error_type, "Error must have a type"
        assert err.description, "Error must have a description"


@pytest.mark.unit
def test_validator_clip_count(tmp_dir):
    """Validator reports correct clip count."""
    from pipeline.stage6_validator import validate_fcpxml
    p = str(tmp_dir / "valid2.fcpxml")
    _write_xml(p, MINIMAL_VALID_FCPXML)
    result = validate_fcpxml(p)
    assert result.result.clip_count == 1


@pytest.mark.unit
def test_validator_missing_version(tmp_dir):
    """Missing version attribute → detected as error."""
    from pipeline.stage6_validator import validate_fcpxml
    xml = MINIMAL_VALID_FCPXML.replace('version="1.10"', '')
    p = str(tmp_dir / "no_version.fcpxml")
    _write_xml(p, xml)
    result = validate_fcpxml(p)
    version_errors = [e for e in result.result.errors if e.error_type == "MISSING_VERSION"]
    assert len(version_errors) >= 1
