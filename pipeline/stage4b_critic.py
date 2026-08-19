"""pipeline/stage4b_critic.py — Critic Agent.

Second-pass deterministic validator of the Director's cut list.
Acts as a peer reviewer — checks every cut against the rule config and flags violations.

On violations found:
  - Emits critic_report.json with itemized violations
  - If violations exceed threshold → triggers self-correction:
    re-runs Director with adjusted parameters (up to max_self_correction_iterations)
"""
from __future__ import annotations

import time

from pipeline.schemas import (
    CameraInventory,
    CriticReport,
    CutEntry,
    CutList,
    CutReason,
    NarrativeLabel,
    NarrativeResult,
    RuleViolation,
    ShowType,
    StageResult,
)
from utils.cache import StageCache
from utils.logging_config import get_logger
from utils.timecode import detect_overlap

logger = get_logger(__name__)


def _check_mid_word_cuts(
    cuts: list[CutEntry],
    narrative: NarrativeResult,
) -> list[RuleViolation]:
    """Safety Rule: Detect mid-word or mid-sentence cuts."""
    violations: list[RuleViolation] = []
    # Build word index for fast lookup
    word_intervals: list[tuple[float, float, str]] = []  # (start, end, segment_id)
    for seg in narrative.segments:
        for word in seg.words:
            word_intervals.append((word.start, word.end, seg.segment_id))

    for cut in cuts:
        if cut.is_off_camera:
            continue
        cut_time = cut.start_s
        for ws, we, _ in word_intervals:
            if ws < cut_time < we:
                violations.append(
                    RuleViolation(
                        cut_id=cut.cut_id,
                        rule="SAFETY_RULE",
                        description=f"Cut at {cut_time:.2f}s falls mid-word ({ws:.2f}–{we:.2f}s)",
                        severity="warning",
                    )
                )
                break

    return violations


def _is_wide_cut(cut: CutEntry) -> bool:
    return (
        "WIDE" in cut.camera_id.upper()
        or "wide" in cut.camera_id.lower()
        or "WIDE" in cut.rule_tag.upper()
        or cut.reason == CutReason.REFRESH_WIDE
        or cut.reason == CutReason.DIALOGUE_WIDE
    )


def _check_wide_shot_cap(
    cuts: list[CutEntry],
    total_duration_s: float,
    show_type: ShowType,
    rules: dict,
) -> list[RuleViolation]:
    """Check wide shot % against show-specific cap."""
    violations: list[RuleViolation] = []
    show_rules = rules.get("show_types", {})

    cap_pct: float | None = None
    if show_type == ShowType.NAV_THETHI:
        cap_pct = show_rules.get("The Nav Thethi Show", {}).get("wide_shot_max_pct")
    elif show_type == ShowType.MATURITY_CODE:
        cap_pct = show_rules.get("Cracking the Maturity Code", {}).get("wide_shot_max_pct")

    if cap_pct is None or total_duration_s <= 0:
        return violations

    wide_duration = sum(c.duration_s for c in cuts if _is_wide_cut(c))
    actual_pct = (wide_duration / total_duration_s) * 100

    if actual_pct > cap_pct:
        violations.append(
            RuleViolation(
                cut_id="GLOBAL",
                rule="SHOW_SPECIFIC_WIDE_CAP",
                description=(
                    f"Wide shot usage {actual_pct:.1f}% exceeds cap {cap_pct}% "
                    f"for {show_type.value}"
                ),
                severity="error",
            )
        )

    return violations


def _check_phy_adj_tags(
    cuts: list[CutEntry],
    narrative: NarrativeResult,
    rules: dict,
) -> list[RuleViolation]:
    """Physical Adjustment Rule: Ensure PHY_ADJ_CUT is tagged for all physical events."""
    violations: list[RuleViolation] = []
    phy_segments = {s.segment_id for s in narrative.segments if s.physical_event}

    for seg_id in phy_segments:
        # Find matching cut
        matching_cuts = [c for c in cuts if c.segment_id == seg_id]
        if not matching_cuts:
            continue
        has_phy_tag = any("PHY_ADJ" in c.rule_tag for c in matching_cuts)
        if not has_phy_tag:
            violations.append(
                RuleViolation(
                    cut_id=matching_cuts[0].cut_id if matching_cuts else "UNKNOWN",
                    rule="PHYSICAL_ADJUSTMENT_RULE",
                    description=f"Segment {seg_id} has physical event but no PHY_ADJ_CUT tag",
                    severity="warning",
                )
            )

    return violations


def _check_off_camera_segments(
    cuts: list[CutEntry],
    narrative: NarrativeResult,
) -> list[RuleViolation]:
    """Off-Camera Rule: Ensure OFF_CAMERA_BRAINSTORM is emitted for all off-camera triggers."""
    violations: list[RuleViolation] = []
    off_cam_segs = {s.segment_id for s in narrative.segments if s.off_camera_trigger}

    for seg_id in off_cam_segs:
        matching_cuts = [c for c in cuts if c.segment_id == seg_id and c.is_off_camera]
        if not matching_cuts:
            violations.append(
                RuleViolation(
                    cut_id="UNKNOWN",
                    rule="OFF_CAMERA_RULE",
                    description=f"Segment {seg_id} has off-camera trigger but no OFF_CAMERA_BRAINSTORM cut",
                    severity="error",
                )
            )

    return violations


def _check_overlapping_cuts(cuts: list[CutEntry]) -> list[RuleViolation]:
    """Detect any overlapping clips on the timeline."""
    violations: list[RuleViolation] = []
    sorted_cuts = sorted(cuts, key=lambda c: c.start_s)
    for i in range(len(sorted_cuts) - 1):
        a = sorted_cuts[i]
        b = sorted_cuts[i + 1]
        if detect_overlap(a.start_s, a.end_s, b.start_s, b.end_s):
            violations.append(
                RuleViolation(
                    cut_id=f"{a.cut_id}+{b.cut_id}",
                    rule="TIMELINE_OVERLAP",
                    description=(
                        f"Overlapping cuts: {a.cut_id} ({a.start_s:.2f}–{a.end_s:.2f}s) "
                        f"and {b.cut_id} ({b.start_s:.2f}–{b.end_s:.2f}s)"
                    ),
                    severity="error",
                )
            )
    return violations


def _check_negative_durations(cuts: list[CutEntry]) -> list[RuleViolation]:
    """Detect cuts with negative or zero duration."""
    violations: list[RuleViolation] = []
    for cut in cuts:
        if cut.duration_s <= 0:
            violations.append(
                RuleViolation(
                    cut_id=cut.cut_id,
                    rule="NEGATIVE_DURATION",
                    description=f"Cut {cut.cut_id} has duration {cut.duration_s:.3f}s",
                    severity="error",
                )
            )
    return violations


def _check_opening_sbs(
    cuts: list[CutEntry],
    narrative: NarrativeResult,
    show_type: ShowType,
    rules: dict,
) -> list[RuleViolation]:
    """Maturity Code: Opening question must remain in SBS."""
    violations: list[RuleViolation] = []
    if show_type != ShowType.MATURITY_CODE:
        return violations
    show_rules = rules.get("show_types", {}).get("Cracking the Maturity Code", {})
    if not show_rules.get("opening_in_sbs", True):
        return violations

    first_questions = [s for s in narrative.segments if s.label == NarrativeLabel.QUESTION]
    if not first_questions:
        return violations

    first_q = first_questions[0]
    matching_cuts = [c for c in cuts if c.segment_id == first_q.segment_id]
    if matching_cuts and not any(c.is_sbs for c in matching_cuts):
        violations.append(
            RuleViolation(
                cut_id=matching_cuts[0].cut_id,
                rule="MATURITY_CODE_OPENING_SBS",
                description="Opening question segment is not marked as SBS (required for Cracking the Maturity Code)",
                severity="error",
            )
        )

    return violations


def _compute_quality_score(
    cuts: list[CutEntry],
    violations: list[RuleViolation],
    total_duration_s: float,
    show_type: ShowType,
    rules: dict,
) -> float:
    """Compute a 0–1 quality score for the cut list."""
    if not cuts or total_duration_s <= 0:
        return 0.0

    score = 1.0

    # Penalize violations
    error_count = sum(1 for v in violations if v.severity == "error")
    warning_count = sum(1 for v in violations if v.severity == "warning")
    score -= error_count * 0.1
    score -= warning_count * 0.03

    # Check cut frequency (target: 1–4 cuts/min for cinematic, 2–6 for dynamic)
    cuts_per_min = (len(cuts) / max(total_duration_s, 60)) * 60
    if show_type == ShowType.NAV_THETHI:
        target_low, target_high = 1.0, 4.0
    else:
        target_low, target_high = 2.0, 6.0

    if cuts_per_min < target_low:
        score -= 0.05  # too few cuts
    elif cuts_per_min > target_high * 1.5:
        score -= 0.1  # too many cuts

    # Check confidence
    avg_confidence = sum(c.confidence for c in cuts) / len(cuts)
    score += (avg_confidence - 0.5) * 0.1

    return max(0.0, min(1.0, round(score, 3)))


def critique(
    cut_list: CutList,
    narrative: NarrativeResult,
    inventory: CameraInventory,
    rules: dict,
    cache: StageCache | None = None,
) -> StageResult:
    """Stage 4b: Critic Agent — validate cut list against all rules."""
    start_time = time.monotonic()
    warnings: list[str] = []
    errors: list[str] = []

    cuts = cut_list.cuts
    total_duration = cut_list.total_duration_s
    show_type = cut_list.show_type

    all_violations: list[RuleViolation] = []

    # Run all checks
    all_violations.extend(_check_negative_durations(cuts))
    all_violations.extend(_check_overlapping_cuts(cuts))
    all_violations.extend(_check_mid_word_cuts(cuts, narrative))
    all_violations.extend(_check_wide_shot_cap(cuts, total_duration, show_type, rules))
    all_violations.extend(_check_phy_adj_tags(cuts, narrative, rules))
    all_violations.extend(_check_off_camera_segments(cuts, narrative))
    all_violations.extend(_check_opening_sbs(cuts, narrative, show_type, rules))

    error_violations = [v for v in all_violations if v.severity == "error"]
    passed = len(error_violations) == 0

    quality_score = _compute_quality_score(cuts, all_violations, total_duration, show_type, rules)

    wide_duration = sum(c.duration_s for c in cuts if _is_wide_cut(c))
    actual_wide_pct = (wide_duration / max(total_duration, 1)) * 100
    cuts_per_min = (len(cuts) / max(total_duration, 60)) * 60

    critic_report = CriticReport(
        violations=all_violations,
        passed=passed,
        quality_score=quality_score,
        wide_shot_pct=actual_wide_pct,
        cut_frequency_per_min=cuts_per_min,
        iteration=cut_list.iteration,
        warnings=warnings,
        errors=errors,
    )

    if all_violations:
        for v in all_violations:
            msg = f"[{v.severity.upper()}] {v.rule}: {v.description}"
            if v.severity == "error":
                errors.append(msg)
            else:
                warnings.append(msg)

    logger.info(
        "stage4b_complete",
        passed=passed,
        total_violations=len(all_violations),
        errors=len(error_violations),
        warnings=len(all_violations) - len(error_violations),
        quality_score=quality_score,
        wide_pct=round(actual_wide_pct, 1),
        cuts_per_min=round(cuts_per_min, 2),
        iteration=cut_list.iteration,
    )

    return StageResult(
        stage="critic",
        success=True,
        result=critic_report,
        warnings=warnings,
        errors=errors,
        duration_s=time.monotonic() - start_time,
    )
