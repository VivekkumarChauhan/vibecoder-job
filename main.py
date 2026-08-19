"""main.py — AI Narrative Video Director — CLI Entrypoint.

Orchestrates all pipeline stages with:
  - Pydantic-validated inter-stage communication
  - Groq budget enforcement and caching
  - Self-correcting Director → Critic retry loop
  - Human-in-the-loop hook (needs_review.json)
  - Full editing_report.json generation
  - HTML timeline visualization (bonus)
  - --force flag to bypass validation and emit anyway

Usage:
  python main.py --input SyncMaster.mp4 --show-type show_type.txt [options]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any module that needs GROQ_API_KEY
load_dotenv()

from pipeline.schemas import (
    CameraInventory,
    EditingReport,
    IngestResult,
    NarrativeResult,
    QualityMetrics,
    ShowType,
    SpeakerMapResult,
)
from pipeline.stage0_ingest import detect_show_type, ingest_syncmaster
from pipeline.stage1_camera_discovery import discover_cameras
from pipeline.stage2_speaker_mapping import map_speakers
from pipeline.stage3_narrative import understand_narrative
from pipeline.stage4_director import direct
from pipeline.stage4b_critic import critique
from pipeline.stage5_xml_generator import generate_fcpxml
from pipeline.stage6_validator import validate_fcpxml
from utils.cache import StageCache
from utils.groq_client import GroqClient, GroqClientConfig
from utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def load_rules(rules_path: str = "editorial_rules.yaml") -> dict:
    """Load editorial_rules.yaml. Returns empty dict on failure (with warning)."""
    try:
        import yaml  # type: ignore[import]

        with open(rules_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("rules_not_found", path=rules_path, msg="Using empty rules")
        return {}
    except Exception as e:
        logger.error("rules_load_error", error=str(e))
        return {}


def _write_needs_review(cuts: list[dict], output_dir: Path) -> int:
    """Write needs_review.json for human-in-the-loop. Returns count of flagged cuts."""
    review_cuts = [c for c in cuts if c.get("needs_review")]
    if review_cuts:
        review_path = output_dir / "needs_review.json"
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "description": (
                        "These cuts have low confidence or are at rule boundaries. "
                        "Edit camera_id or approve by removing 'needs_review' flag before "
                        "re-running the XML generator."
                    ),
                    "cuts": review_cuts,
                },
                f,
                indent=2,
            )
        logger.info("needs_review_written", path=str(review_path), count=len(review_cuts))
    return len(review_cuts)


def _write_editing_report(
    report: EditingReport,
    output_path: Path,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, default=str)
    logger.info("editing_report_written", path=str(output_path))


def run_pipeline(
    source_path: str,
    show_type_path: str,
    output_dir: str = ".",
    rules_path: str = "editorial_rules.yaml",
    cache_dir: str = "./cache",
    force: bool = False,
    skip_hitl: bool = False,
    no_timeline: bool = False,
) -> tuple[bool, EditingReport]:
    """Run the full AI Narrative Director pipeline. Returns (success, report)."""
    pipeline_start = time.monotonic()
    all_warnings: list[str] = []
    stage_times: dict[str, float] = {}
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # ── Load config ──────────────────────────────────────────────────────────
    rules = load_rules(rules_path)
    logger.info("pipeline_start", source=source_path, show_type_file=show_type_path)

    # ── Initialize shared resources ─────────────────────────────────────────
    cache = StageCache(cache_dir)

    groq_config = GroqClientConfig(
        api_key=os.getenv("GROQ_API_KEY", ""),
        model=rules.get("groq", {}).get("model", "llama-3.3-70b-versatile"),
        vision_model=rules.get("groq", {}).get("vision_model", "meta-llama/llama-4-scout-17b-16e-instruct"),
        budget_per_run=int(os.getenv("GROQ_BUDGET", rules.get("groq", {}).get("budget_per_run", 50))),
        max_retries=rules.get("groq", {}).get("max_retries", 3),
        base_delay_seconds=rules.get("groq", {}).get("base_delay_seconds", 2.0),
        max_delay_seconds=rules.get("groq", {}).get("max_delay_seconds", 30.0),
    )
    groq_client = GroqClient(config=groq_config, cache=cache)

    report = EditingReport()

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 0: Ingest
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("stage0_starting")
    s0 = ingest_syncmaster(source_path, show_type_path, cache=cache)
    stage_times["ingest"] = s0.duration_s
    all_warnings.extend(s0.warnings)

    if not s0.success or s0.result is None:
        logger.error("stage0_failed", errors=s0.errors)
        report.warnings = all_warnings
        report.validation_passed = False
        _write_editing_report(report, output_dir_path / "editing_report.json")
        return False, report

    ingest: IngestResult = s0.result
    show_type, _ = detect_show_type(show_type_path)

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1: Camera Discovery
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("stage1_starting")
    cam_rules = rules.get("camera_discovery", {})
    s1 = discover_cameras(ingest, source_path, cam_rules, cache=cache, groq_client=groq_client)
    stage_times["camera_discovery"] = s1.duration_s
    all_warnings.extend(s1.warnings)

    if s1.result is None:
        logger.error("stage1_failed", errors=s1.errors)
        all_warnings.extend(s1.errors)
        # Fallback: create minimal inventory from ingest streams
        from pipeline.schemas import CameraInfo, CameraRole
        cameras = [
            CameraInfo(
                camera_id=f"cam_{i + 1}",
                stream_index=st.stream_index,
                role=CameraRole.HOST_HERO if i == 0 else CameraRole.UNKNOWN,
                is_active=True,
            )
            for i, st in enumerate(ingest.video_streams)
        ]
        inventory = CameraInventory(
            cameras=cameras,
            role_map={CameraRole.HOST_HERO.value: "cam_1"} if cameras else {},
            total_cameras=len(cameras),
            active_cameras=len(cameras),
            empty_cameras=0,
            warnings=["Camera discovery failed — using stream-based fallback"],
        )
    else:
        inventory: CameraInventory = s1.result

    report.camera_inventory = inventory.model_dump()

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2: Speaker Mapping
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("stage2_starting")
    s2 = map_speakers(ingest, source_path, inventory, rules, cache=cache, groq_client=groq_client)
    stage_times["speaker_mapping"] = s2.duration_s
    all_warnings.extend(s2.warnings)

    if s2.result is None:
        logger.error("stage2_failed", errors=s2.errors)
        all_warnings.extend(s2.errors)
        # Fallback: empty speaker result
        from pipeline.schemas import SpeakerMapping, TranscriptSegment
        speaker_result = SpeakerMapResult(
            mapping=SpeakerMapping(host_label="SPEAKER_00", all_labels=["SPEAKER_00"]),
            transcript=[
                TranscriptSegment(
                    segment_id="seg_0000",
                    speaker_label="SPEAKER_00",
                    start=0.0,
                    end=ingest.duration_s,
                    text="[Speaker mapping unavailable]",
                )
            ],
            warnings=["Speaker mapping failed — using fallback single speaker"],
        )
    else:
        speaker_result: SpeakerMapResult = s2.result

    report.speaker_mapping = speaker_result.mapping.model_dump()

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 3: Narrative Understanding
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("stage3_starting")
    s3 = understand_narrative(speaker_result, show_type, rules, cache=cache, groq_client=groq_client)
    stage_times["narrative"] = s3.duration_s
    all_warnings.extend(s3.warnings)

    if s3.result is None:
        all_warnings.extend(s3.errors)
        # Fallback: create narrative from transcript
        from pipeline.schemas import NarrativeLabel, NarrativeSegment
        narrative = NarrativeResult(
            segments=[
                NarrativeSegment(
                    segment_id=seg.segment_id,
                    speaker_label=seg.speaker_label,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    label=NarrativeLabel.UNKNOWN,
                )
                for seg in speaker_result.transcript
            ],
            show_type=show_type,
            warnings=["Narrative understanding failed — using fallback labels"],
        )
    else:
        narrative: NarrativeResult = s3.result

    # Off-camera segments for report
    report.off_camera_segments = [
        {"segment_id": s.segment_id, "start": s.start, "end": s.end, "text": s.text}
        for s in narrative.segments
        if s.off_camera_trigger
    ]

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 4 + 4b: Director → Critic → Self-Correction Loop
    # ═══════════════════════════════════════════════════════════════════════
    max_iterations = rules.get("director", {}).get("max_self_correction_iterations", 3)
    final_cut_list = None
    final_critic_report = None
    effective_rules = dict(rules)

    for iteration in range(max_iterations):
        logger.info("stage4_starting", iteration=iteration)
        s4 = direct(narrative, inventory, effective_rules, cache=cache, iteration=iteration)
        stage_times[f"director_iter{iteration}"] = s4.duration_s
        all_warnings.extend(s4.warnings)

        if s4.result is None:
            all_warnings.extend(s4.errors)
            break

        cut_list = s4.result

        # Critic pass
        logger.info("stage4b_starting", iteration=iteration)
        s4b = critique(cut_list, narrative, inventory, effective_rules, cache=cache)
        stage_times[f"critic_iter{iteration}"] = s4b.duration_s
        all_warnings.extend(s4b.warnings)

        critic_report = s4b.result
        final_cut_list = cut_list
        final_critic_report = critic_report

        if critic_report is None or critic_report.passed:
            logger.info("critic_passed", iteration=iteration)
            break

        # Violations found — attempt self-correction
        error_violations = [v for v in critic_report.violations if v.severity == "error"]
        logger.warning(
            "critic_violations_found",
            iteration=iteration,
            errors=len(error_violations),
            violations=[v.rule for v in error_violations[:5]],
        )

        if iteration < max_iterations - 1:
            # Adjust parameters for next iteration
            # e.g., tighten wide cap slightly, increase safety hold
            effective_rules = _adjust_rules_for_retry(effective_rules, critic_report.violations)
            all_warnings.append(
                f"Self-correction iteration {iteration + 1}: adjusting rules based on {len(error_violations)} violations"
            )
        else:
            all_warnings.append(
                f"Max self-correction iterations ({max_iterations}) reached with {len(error_violations)} errors. "
                "Proceeding with best available cut list."
            )

    if final_cut_list is None:
        logger.error("director_produced_no_cut_list")
        all_warnings.append("Director failed to produce a cut list — aborting")
        report.warnings = all_warnings
        _write_editing_report(report, output_dir_path / "editing_report.json")
        return False, report

    # Update report with cuts and critic data
    report.cuts = [c.model_dump() for c in final_cut_list.cuts]
    if final_critic_report:
        report.critic_violations = [v.model_dump() for v in final_critic_report.violations]

    # ── Human-in-the-Loop ───────────────────────────────────────────────────
    if not skip_hitl and rules.get("hitl", {}).get("enabled", True):
        hitl_count = _write_needs_review(report.cuts, output_dir_path)
        if hitl_count > 0:
            all_warnings.append(
                f"{hitl_count} cuts written to needs_review.json for human review. "
                "Edit the file and re-run with --skip-hitl to proceed."
            )
            if not force:
                logger.info(
                    "hitl_pause",
                    count=hitl_count,
                    msg="Proceeding automatically (use --skip-hitl to suppress this message)",
                )

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 5: XML Generator
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("stage5_starting")
    output_fcpxml = str(output_dir_path / "output.fcpxml")
    s5 = generate_fcpxml(
        cut_list=final_cut_list,
        inventory=inventory,
        ingest=ingest,
        source_path=source_path,
        output_path=output_fcpxml,
        rules=rules,
        show_type=show_type,
    )
    stage_times["xml_generator"] = s5.duration_s
    all_warnings.extend(s5.warnings)

    if not s5.success:
        all_warnings.extend(s5.errors)
        logger.error("stage5_failed", errors=s5.errors)
        report.warnings = all_warnings
        _write_editing_report(report, output_dir_path / "editing_report.json")
        return False, report

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 6: Validation
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("stage6_starting")
    s6 = validate_fcpxml(output_fcpxml, inventory=inventory, ingest=ingest, rules=rules)
    stage_times["validator"] = s6.duration_s
    all_warnings.extend(s6.warnings)

    validation_passed = s6.success
    report.validation_passed = validation_passed

    if not validation_passed:
        val_errors = [f"[{e.error_type}] {e.description}" for e in (s6.result.errors if s6.result else [])]
        all_warnings.extend(val_errors)
        if not force:
            logger.error(
                "validation_failed",
                errors=val_errors,
                msg="Use --force to emit anyway",
            )
        else:
            logger.warning("validation_failed_force_emit", errors=val_errors)

    # ── Quality Metrics ──────────────────────────────────────────────────────
    cache_stats = cache.stats()
    wide_cap = None
    if show_type == ShowType.NAV_THETHI:
        wide_cap = rules.get("show_types", {}).get("The Nav Thethi Show", {}).get("wide_shot_max_pct")
    elif show_type == ShowType.MATURITY_CODE:
        wide_cap = rules.get("show_types", {}).get("Cracking the Maturity Code", {}).get("wide_shot_max_pct")

    report.metadata = QualityMetrics(
        quality_score=final_critic_report.quality_score if final_critic_report else 0.0,
        cut_frequency_per_min=final_critic_report.cut_frequency_per_min if final_critic_report else 0.0,
        wide_shot_pct=final_cut_list.wide_shot_pct,
        wide_shot_cap_pct=wide_cap,
        wide_shot_cap_met=(
            final_cut_list.wide_shot_pct <= wide_cap if wide_cap is not None else True
        ),
        rule_violation_count=len(final_critic_report.violations) if final_critic_report else 0,
        groq_calls_made=groq_client.calls_used,
        groq_cache_hits=cache_stats["hits"],
        processing_time_per_stage=stage_times,
        self_correction_iterations=final_cut_list.iteration,
    )
    report.warnings = all_warnings

    # ── Write final reports ──────────────────────────────────────────────────
    _write_editing_report(report, output_dir_path / "editing_report.json")

    # Optional: HTML timeline
    if not no_timeline:
        try:
            from generate_timeline_html import generate_timeline

            timeline_path = str(output_dir_path / "timeline.html")
            generate_timeline(final_cut_list, inventory, timeline_path)
            logger.info("timeline_generated", path=timeline_path)
        except Exception as e:
            all_warnings.append(f"Timeline generation failed (non-critical): {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    total_time = time.monotonic() - pipeline_start
    logger.info(
        "pipeline_complete",
        success=validation_passed or force,
        total_duration_s=round(total_time, 2),
        fcpxml=output_fcpxml,
        report=str(output_dir_path / "editing_report.json"),
        groq_calls=groq_client.calls_used,
        cache_hits=cache_stats["hits"],
        wide_pct=round(final_cut_list.wide_shot_pct, 1),
        quality_score=report.metadata.quality_score,
        total_cuts=len(final_cut_list.cuts),
        warnings=len(all_warnings),
    )

    _print_summary(
        output_dir_path,
        final_cut_list,
        report,
        validation_passed,
        total_time,
        groq_client,
        cache_stats,
    )

    return validation_passed or force, report


def _adjust_rules_for_retry(rules: dict, violations: list) -> dict:
    """Adjust rule thresholds for self-correction retry."""
    import copy

    new_rules = copy.deepcopy(rules)
    violation_types = {v.rule for v in violations}

    # If wide cap exceeded, tighten it further
    if "SHOW_SPECIFIC_WIDE_CAP" in violation_types:
        for show_key in new_rules.get("show_types", {}):
            cap = new_rules["show_types"][show_key].get("wide_shot_max_pct", 40.0)
            new_rules["show_types"][show_key]["wide_shot_max_pct"] = max(5.0, cap - 5.0)

    # If overlaps detected, increase safety hold
    if "TIMELINE_OVERLAP" in violation_types:
        new_rules.setdefault("director", {})["safety_min_hold_s"] = (
            new_rules.get("director", {}).get("safety_min_hold_s", 1.0) + 0.5
        )

    return new_rules


def _print_summary(
    output_dir: Path,
    cut_list: object,
    report: EditingReport,
    validation_passed: bool,
    total_time: float,
    groq_client: GroqClient,
    cache_stats: dict,
) -> None:
    """Print human-readable pipeline summary."""
    cuts = getattr(cut_list, "cuts", [])
    wide_pct = getattr(cut_list, "wide_shot_pct", 0.0)

    print("\n" + "=" * 60)
    print("  AI NARRATIVE VIDEO DIRECTOR - Pipeline Complete")
    print("=" * 60)
    print(f"  Output FCPXML:     {output_dir / 'output.fcpxml'}")
    print(f"  Editing Report:    {output_dir / 'editing_report.json'}")
    print(f"  Total Cuts:        {len(cuts)}")
    print(f"  Wide Shot %:       {wide_pct:.1f}%")
    print(f"  Quality Score:     {report.metadata.quality_score:.3f}")
    print(f"  Violations:        {report.metadata.rule_violation_count}")
    print(f"  Groq API Calls:    {groq_client.calls_used} used / cache hits: {cache_stats['hits']}")
    print(f"  Total Time:        {total_time:.1f}s")
    print(f"  Validation:        {'[PASS] PASSED' if validation_passed else '[FAIL] FAILED'}")
    print(f"  Warnings:          {len(report.warnings)}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Narrative Video Director — multicam podcast editor"
    )
    parser.add_argument("--input", required=True, help="Path to SyncMaster video file")
    parser.add_argument("--show-type", required=True, help="Path to show_type.txt")
    parser.add_argument("--output-dir", default=".", help="Output directory (default: current)")
    parser.add_argument("--rules", default="editorial_rules.yaml", help="Path to editorial_rules.yaml")
    parser.add_argument("--cache-dir", default="./cache", help="Cache directory")
    parser.add_argument("--force", action="store_true", help="Emit FCPXML even if validation fails")
    parser.add_argument("--skip-hitl", action="store_true", help="Skip human-in-the-loop review step")
    parser.add_argument("--no-timeline", action="store_true", help="Skip HTML timeline generation")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-format", default="console", choices=["json", "console"])

    args = parser.parse_args()
    configure_logging(level=args.log_level, fmt=args.log_format)

    success, report = run_pipeline(
        source_path=args.input,
        show_type_path=args.show_type,
        output_dir=args.output_dir,
        rules_path=args.rules,
        cache_dir=args.cache_dir,
        force=args.force,
        skip_hitl=args.skip_hitl,
        no_timeline=args.no_timeline,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
