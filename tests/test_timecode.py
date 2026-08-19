"""tests/test_timecode.py — Unit tests for timecode utilities."""
from __future__ import annotations

import pytest
from fractions import Fraction

from utils.timecode import (
    Timecode,
    FCP_TICKS_PER_SECOND,
    detect_overlap,
    fcp_rational_to_seconds,
    format_hms,
    frames_to_fcp_rational,
    seconds_to_fcp_rational,
)


FPS_NUM = 30000
FPS_DEN = 1001


@pytest.mark.unit
def test_timecode_from_seconds():
    tc = Timecode.from_seconds(10.0, FPS_NUM, FPS_DEN)
    assert tc.frames == 300  # 10 * 30000/1001 ≈ 299.7 → 300 frames
    # Just check it's close
    assert abs(tc.to_seconds() - 10.0) < 0.1


@pytest.mark.unit
def test_timecode_zero():
    tc = Timecode.from_seconds(0.0, FPS_NUM, FPS_DEN)
    assert tc.frames == 0
    assert tc.to_fcp_ticks() == 0


@pytest.mark.unit
def test_timecode_ticks_integer():
    tc = Timecode.from_seconds(1.0, FPS_NUM, FPS_DEN)
    ticks = tc.to_fcp_ticks()
    assert isinstance(ticks, int), "Ticks must be integer"


@pytest.mark.unit
def test_timecode_rational_string():
    tc = Timecode.from_seconds(5.0, FPS_NUM, FPS_DEN)
    rational = tc.to_fcp_rational()
    assert rational.endswith("s"), "Rational must end with 's'"
    assert "/" in rational, "Rational must contain '/'"


@pytest.mark.unit
def test_fcp_rational_roundtrip():
    for seconds in [0.0, 1.0, 10.5, 60.0, 3600.0]:
        tc = Timecode.from_seconds(seconds, FPS_NUM, FPS_DEN)
        rational = tc.to_fcp_rational()
        back = fcp_rational_to_seconds(rational)
        assert abs(back - tc.to_seconds()) < 0.01, f"Roundtrip failed for {seconds}s"


@pytest.mark.unit
def test_detect_overlap_true():
    assert detect_overlap(0.0, 10.0, 5.0, 15.0) is True


@pytest.mark.unit
def test_detect_overlap_false_adjacent():
    assert detect_overlap(0.0, 10.0, 10.0, 20.0) is False


@pytest.mark.unit
def test_detect_overlap_false_separated():
    assert detect_overlap(0.0, 5.0, 10.0, 20.0) is False


@pytest.mark.unit
def test_format_hms():
    assert format_hms(0.0) == "00:00:00.000"
    assert format_hms(3661.5) == "01:01:01.500"


@pytest.mark.unit
def test_timecode_addition():
    tc1 = Timecode(100, FPS_NUM, FPS_DEN)
    tc2 = Timecode(200, FPS_NUM, FPS_DEN)
    result = tc1 + tc2
    assert result.frames == 300


@pytest.mark.unit
def test_timecode_comparison():
    tc1 = Timecode(100, FPS_NUM, FPS_DEN)
    tc2 = Timecode(200, FPS_NUM, FPS_DEN)
    assert tc1 < tc2
    assert tc2 > tc1
    assert tc1 <= tc1


@pytest.mark.unit
def test_seconds_to_fcp_rational_no_float_issues():
    """Critical: timecode math must produce integer ticks, not floats."""
    rational = seconds_to_fcp_rational(1.0, FPS_NUM, FPS_DEN)
    # Parse back
    num_str, den_str = rational.rstrip("s").split("/")
    num, den = int(num_str), int(den_str)
    # Numerator and denominator must be integers
    assert isinstance(num, int)
    assert isinstance(den, int)
    assert den == FCP_TICKS_PER_SECOND
