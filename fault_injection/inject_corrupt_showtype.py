"""fault_injection/inject_corrupt_showtype.py — Demo: corrupt show_type.txt.

Tests all corruption scenarios:
- Empty file
- Unrecognized show type
- Missing file
- Garbled content

Shows graceful degradation to ShowType.UNKNOWN with informative warnings.

Run:
  python fault_injection/inject_corrupt_showtype.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.schemas import ShowType
from pipeline.stage0_ingest import detect_show_type
from utils.logging_config import configure_logging

configure_logging(level="WARNING", fmt="console")


def test_case(label: str, content: str | None, expected: ShowType) -> bool:
    """Run a single show_type detection test case."""
    with tempfile.TemporaryDirectory() as tmpdir:
        show_type_path = Path(tmpdir) / "show_type.txt"

        if content is None:
            # Missing file scenario
            path_to_use = Path(tmpdir) / "nonexistent.txt"
        else:
            show_type_path.write_text(content, encoding="utf-8")
            path_to_use = show_type_path

        detected, warnings = detect_show_type(str(path_to_use))

        status = "[PASS]" if detected == expected else "[FAIL]"
        print(f"  {status} [{label}]")
        print(f"     Content:  {repr(content)[:60] if content is not None else '<missing>'}")
        print(f"     Detected: {detected.value}")
        print(f"     Expected: {expected.value}")
        if warnings:
            print(f"     Warning:  {warnings[0][:80]}")
        print()
        return detected == expected


def run_demo() -> None:
    print("\n[FAULT INJECTION] Corrupt show_type.txt Demo")
    print("=" * 55)
    print("Testing all corruption scenarios:\n")

    results = []

    # ── Test cases ──────────────────────────────────────────────────────────
    results.append(test_case(
        "Valid: exact match",
        "The Nav Thethi Show",
        ShowType.NAV_THETHI,
    ))
    results.append(test_case(
        "Valid: case insensitive",
        "the nav thethi show",
        ShowType.NAV_THETHI,
    ))
    results.append(test_case(
        "Valid: with whitespace",
        "  Cracking the Maturity Code  \n",
        ShowType.MATURITY_CODE,
    ))
    results.append(test_case(
        "Valid: partial match",
        "Nav Thethi Show - Episode 42",
        ShowType.NAV_THETHI,
    ))
    results.append(test_case(
        "Corrupt: empty file",
        "",
        ShowType.UNKNOWN,
    ))
    results.append(test_case(
        "Corrupt: completely unrecognized",
        "My Fake Podcast Show",
        ShowType.UNKNOWN,
    ))
    results.append(test_case(
        "Corrupt: garbled binary-like content",
        "\x00\xff\x00invalid\xfe\xfd",
        ShowType.UNKNOWN,
    ))
    results.append(test_case(
        "Corrupt: missing file",
        None,
        ShowType.UNKNOWN,
    ))
    results.append(test_case(
        "Corrupt: numbers only",
        "12345678",
        ShowType.UNKNOWN,
    ))

    passed = sum(results)
    total = len(results)
    print(f"{'-' * 55}")
    print(f"  Results: {passed}/{total} passed")
    if passed == total:
        print("  [PASS] All show_type corruption scenarios handled gracefully")
    else:
        print("  [FAIL] Some scenarios failed - check logic above")
    print("=" * 55 + "\n")

    assert passed == total, f"Only {passed}/{total} show_type corruption tests passed"


if __name__ == "__main__":
    run_demo()
