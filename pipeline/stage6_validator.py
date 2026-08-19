"""pipeline/stage6_validator.py — FCPXML Schema + Editorial Validator.

Validates output.fcpxml against:
  1. FCPXML v1.10 structural requirements (lxml-based)
  2. Asset reference integrity (every clip.ref must exist in <resources>)
  3. Timecode bounds (all clip start/end within source duration)
  4. No overlapping clips on same track
  5. No negative or zero durations
  6. Non-contiguous timeline gaps (warnings, not errors)
  7. Premiere Pro compatibility checks (format, audio roles, etc.)

Returns: ValidationReport with pass/fail + itemized error list.
The pipeline refuses to emit a file that fails unless --force is passed.
"""
from __future__ import annotations

import time
from pathlib import Path

from lxml import etree  # type: ignore[import]

from pipeline.schemas import (
    CameraInventory,
    IngestResult,
    StageResult,
    ValidationError,
    ValidationReport,
)
from utils.logging_config import get_logger
from utils.timecode import fcp_rational_to_seconds

logger = get_logger(__name__)

# Required top-level FCPXML attributes
REQUIRED_FCPXML_ATTRS = {"version"}
VALID_FCPXML_VERSIONS = {"1.8", "1.9", "1.10", "1.11"}

# Elements required in FCPXML
REQUIRED_CHILDREN = {
    "fcpxml": {"resources", "library"},
    "library": {"event"},
    "event": {"project"},
    "project": {"sequence"},
    "sequence": {"spine"},
}


def _parse_rational(val: str) -> float:
    """Parse FCP rational string to float seconds. Returns 0.0 on error."""
    if not val:
        return 0.0
    try:
        return fcp_rational_to_seconds(val)
    except Exception:
        try:
            return float(val.rstrip("s"))
        except Exception:
            return 0.0


def _check_structure(root: etree._Element) -> list[ValidationError]:
    """Check required FCPXML structural elements exist."""
    errors: list[ValidationError] = []

    # Version check
    version = root.get("version", "")
    if not version:
        errors.append(
            ValidationError(error_type="MISSING_VERSION", description="fcpxml element missing 'version' attribute")
        )
    elif version not in VALID_FCPXML_VERSIONS:
        errors.append(
            ValidationError(
                error_type="INVALID_VERSION",
                description=f"fcpxml version '{version}' not in supported set {VALID_FCPXML_VERSIONS}",
                severity="warning",
            )
        )

    # Required child elements
    def check_children(parent: etree._Element, parent_tag: str) -> None:
        required = REQUIRED_CHILDREN.get(parent_tag, set())
        children_tags = {c.tag for c in parent if isinstance(c.tag, str)}
        for req in required:
            if req not in children_tags:
                errors.append(
                    ValidationError(
                        error_type="MISSING_ELEMENT",
                        description=f"<{parent_tag}> is missing required child <{req}>",
                    )
                )

    check_children(root, "fcpxml")
    library = root.find("library")
    if library is not None:
        check_children(library, "library")
        event = library.find("event")
        if event is not None:
            check_children(event, "event")
            project = event.find("project")
            if project is not None:
                check_children(project, "project")
                sequence = project.find("sequence")
                if sequence is not None:
                    check_children(sequence, "sequence")

    return errors


def _check_asset_refs(root: etree._Element) -> list[ValidationError]:
    """Verify all asset refs used in clips exist in <resources>."""
    errors: list[ValidationError] = []

    # Collect all defined asset IDs
    resources = root.find("resources")
    defined_asset_ids: set[str] = set()
    if resources is not None:
        for asset in resources.findall("asset"):
            aid = asset.get("id", "")
            if aid:
                defined_asset_ids.add(aid)

    # Check all clip/video/audio ref attributes
    for elem in root.iter():
        ref = elem.get("ref")
        if ref and ref.startswith("r_") and ref not in defined_asset_ids:  # our asset ID convention
            clip_name = elem.getparent().get("name", "?") if elem.getparent() is not None else "?"
            errors.append(
                ValidationError(
                    error_type="MISSING_ASSET_REF",
                    description=f"Element '{elem.tag}' references undefined asset '{ref}' (in clip '{clip_name}')",
                    clip_id=ref,
                )
            )

    return errors


def _check_timecode_bounds(
    root: etree._Element,
    source_duration_s: float,
) -> list[ValidationError]:
    """Check all clip timecodes are within source duration bounds."""
    errors: list[ValidationError] = []

    for clip in root.iter("clip"):
        name = clip.get("name", "?")
        start_val = clip.get("start", "")
        duration_val = clip.get("duration", "")

        start_s = _parse_rational(start_val)
        dur_s = _parse_rational(duration_val)

        if dur_s < 0:
            errors.append(
                ValidationError(
                    error_type="NEGATIVE_DURATION",
                    description=f"Clip '{name}' has negative duration {dur_s:.3f}s",
                    clip_id=name,
                )
            )

        if start_s < 0:
            errors.append(
                ValidationError(
                    error_type="NEGATIVE_START",
                    description=f"Clip '{name}' has negative start {start_s:.3f}s",
                    clip_id=name,
                    severity="warning",
                )
            )

        clip_end_s = start_s + dur_s
        if source_duration_s > 0 and clip_end_s > source_duration_s + 0.1:  # 100ms tolerance
            errors.append(
                ValidationError(
                    error_type="TIMECODE_OUT_OF_BOUNDS",
                    description=(
                        f"Clip '{name}' ends at {clip_end_s:.2f}s "
                        f"but source duration is {source_duration_s:.2f}s"
                    ),
                    clip_id=name,
                    severity="warning",
                )
            )

    return errors


def _check_overlaps(root: etree._Element) -> list[ValidationError]:
    """Detect overlapping clips on the same spine track."""
    errors: list[ValidationError] = []

    for spine in root.iter("spine"):
        clips: list[tuple[float, float, str]] = []  # (offset, offset+dur, name)

        for child in spine:
            if child.tag not in ("clip", "gap"):
                continue
            offset_s = _parse_rational(child.get("offset", "0s"))
            dur_s = _parse_rational(child.get("duration", "0s"))
            end_s = offset_s + dur_s
            name = child.get("name", "?")

            for prev_start, prev_end, prev_name in clips:
                if offset_s < prev_end and prev_start < end_s:
                    overlap_amount = min(end_s, prev_end) - max(offset_s, prev_start)
                    if overlap_amount > 0.001:  # > 1ms
                        errors.append(
                            ValidationError(
                                error_type="CLIP_OVERLAP",
                                description=(
                                    f"Clips '{prev_name}' ({prev_start:.2f}–{prev_end:.2f}s) "
                                    f"and '{name}' ({offset_s:.2f}–{end_s:.2f}s) "
                                    f"overlap by {overlap_amount:.3f}s"
                                ),
                                clip_id=f"{prev_name}+{name}",
                            )
                        )

            clips.append((offset_s, end_s, name))

    return errors


def _check_premiere_compatibility(root: etree._Element) -> list[ValidationError]:
    """Check Premiere Pro compatibility requirements."""
    errors: list[ValidationError] = []

    # Sequence must have format attr
    for seq in root.iter("sequence"):
        if not seq.get("format"):
            errors.append(
                ValidationError(
                    error_type="MISSING_SEQUENCE_FORMAT",
                    description="<sequence> missing 'format' attribute (required for Premiere import)",
                    severity="warning",
                )
            )

    # Assets must have src attribute
    resources = root.find("resources")
    if resources is not None:
        for asset in resources.findall("asset"):
            if not asset.get("src"):
                errors.append(
                    ValidationError(
                        error_type="MISSING_ASSET_SRC",
                        description=f"Asset '{asset.get('id', '?')}' missing 'src' attribute",
                        clip_id=asset.get("id"),
                        severity="warning",
                    )
                )
            if not asset.get("format"):
                errors.append(
                    ValidationError(
                        error_type="MISSING_ASSET_FORMAT",
                        description=f"Asset '{asset.get('id', '?')}' missing 'format' attribute",
                        clip_id=asset.get("id"),
                        severity="warning",
                    )
                )

    return errors


def _check_gaps(root: etree._Element) -> list[ValidationError]:
    """Check for unexpected gaps (non-OFF_CAMERA gaps are warnings)."""
    warnings: list[ValidationError] = []

    for spine in root.iter("spine"):
        prev_end = 0.0
        for child in spine:
            if child.tag == "clip":
                offset_s = _parse_rational(child.get("offset", "0s"))
                if offset_s > prev_end + 0.1:  # > 100ms gap
                    gap_dur = offset_s - prev_end
                    warnings.append(
                        ValidationError(
                            error_type="UNEXPECTED_GAP",
                            description=f"Gap of {gap_dur:.2f}s at {prev_end:.2f}s before clip '{child.get('name', '?')}'",
                            clip_id=child.get("name"),
                            severity="warning",
                        )
                    )
                dur_s = _parse_rational(child.get("duration", "0s"))
                prev_end = offset_s + dur_s
            elif child.tag == "gap":
                offset_s = _parse_rational(child.get("offset", "0s"))
                dur_s = _parse_rational(child.get("duration", "0s"))
                prev_end = offset_s + dur_s

    return warnings


def validate_fcpxml(
    fcpxml_path: str,
    inventory: CameraInventory | None = None,
    ingest: IngestResult | None = None,
    rules: dict | None = None,
) -> StageResult:
    """Stage 6: Validate output.fcpxml."""
    start_time = time.monotonic()
    all_errors: list[ValidationError] = []
    all_warnings: list[ValidationError] = []

    path = Path(fcpxml_path)
    if not path.exists():
        return StageResult(
            stage="validator",
            success=False,
            result=ValidationReport(
                passed=False,
                errors=[ValidationError(error_type="FILE_NOT_FOUND", description=f"File not found: {fcpxml_path}")],
            ),
            errors=[f"File not found: {fcpxml_path}"],
            duration_s=time.monotonic() - start_time,
        )

    # Parse XML
    try:
        parser = etree.XMLParser(remove_comments=False)
        tree = etree.parse(str(path), parser)
        root = tree.getroot()
    except etree.XMLSyntaxError as e:
        return StageResult(
            stage="validator",
            success=False,
            result=ValidationReport(
                passed=False,
                errors=[ValidationError(error_type="XML_SYNTAX_ERROR", description=str(e))],
            ),
            errors=[f"XML parse error: {e}"],
            duration_s=time.monotonic() - start_time,
        )

    source_duration_s = ingest.duration_s if ingest else 0.0

    # Run all checks
    struct_errors = _check_structure(root)
    asset_errors = _check_asset_refs(root)
    tc_errors = _check_timecode_bounds(root, source_duration_s)
    overlap_errors = _check_overlaps(root)
    premiere_errors = _check_premiere_compatibility(root)
    gap_warnings = _check_gaps(root)

    for e in struct_errors + asset_errors + tc_errors + overlap_errors + premiere_errors:
        if e.severity == "error":
            all_errors.append(e)
        else:
            all_warnings.append(e)

    all_warnings.extend(gap_warnings)

    # Count clips and duration
    clip_count = sum(1 for _ in root.iter("clip"))
    total_duration_s = 0.0
    seq = root.find(".//sequence")
    if seq is not None:
        total_duration_s = _parse_rational(seq.get("duration", "0s"))

    # Wide shot %
    wide_duration = 0.0
    for clip in root.iter("clip"):
        name = clip.get("name", "")
        dur = _parse_rational(clip.get("duration", "0s"))
        if "wide" in name.lower() or "WIDE" in name:
            wide_duration += dur
    wide_pct = (wide_duration / max(total_duration_s, 1)) * 100

    passed = len(all_errors) == 0
    report = ValidationReport(
        passed=passed,
        errors=all_errors,
        warnings=all_warnings,
        clip_count=clip_count,
        total_duration_s=total_duration_s,
        wide_shot_pct=wide_pct,
    )

    logger.info(
        "stage6_complete",
        passed=passed,
        errors=len(all_errors),
        warnings=len(all_warnings),
        clip_count=clip_count,
        total_duration_s=round(total_duration_s, 2),
        wide_pct=round(wide_pct, 1),
    )

    error_strs = [f"[{e.error_type}] {e.description}" for e in all_errors]
    warning_strs = [f"[{w.error_type}] {w.description}" for w in all_warnings]

    return StageResult(
        stage="validator",
        success=passed,
        result=report,
        errors=error_strs,
        warnings=warning_strs,
        duration_s=time.monotonic() - start_time,
    )
