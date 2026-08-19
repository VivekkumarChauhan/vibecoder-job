"""validate_xml.py — Standalone FCPXML Validator (demo-ready).

Runnable standalone:
  python validate_xml.py output.fcpxml

Runs all validation checks and prints a human-readable pass/fail report.
No pipeline state needed — can be run at any time on any FCPXML file.
"""
from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python validate_xml.py <path_to_fcpxml>")
        sys.exit(1)

    fcpxml_path = sys.argv[1]

    # Configure logging
    from utils.logging_config import configure_logging
    configure_logging(level="WARNING", fmt="console")

    from pipeline.stage6_validator import validate_fcpxml

    print(f"\n{'=' * 55}")
    print("  FCPXML Standalone Validator")
    print(f"  File: {fcpxml_path}")
    print(f"{'=' * 55}\n")

    result = validate_fcpxml(fcpxml_path)

    if result.result is None:
        print("[FAIL] VALIDATION FAILED: Could not parse file")
        for e in result.errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    report = result.result

    if report.passed:
        print("[PASS] VALIDATION PASSED\n")
    else:
        print("[FAIL] VALIDATION FAILED\n")

    print(f"  Clip count:       {report.clip_count}")
    print(f"  Total duration:   {report.total_duration_s:.2f}s")
    print(f"  Wide shot %:      {report.wide_shot_pct:.1f}%\n")

    if report.errors:
        print(f"  ERRORS ({len(report.errors)}):")
        for e in report.errors:
            print(f"    [ERROR] [{e.error_type}] {e.description}")
    else:
        print("  No errors found.")

    if report.warnings:
        print(f"\n  WARNINGS ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"    [WARN]  [{w.error_type}] {w.description}")
    else:
        print("  No warnings.")

    print(f"\n{'=' * 55}\n")
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
