"""utils/timecode.py — Frame-accurate timecode math.

All timecodes are handled as integer frame counts or FCP ticks.
NEVER use floating-point arithmetic for timecode calculations in XML generation.

FCP tick rate: 254016000000 ticks/second (lcm of common frame rates).
"""
from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple

# FCP internal tick rate
FCP_TICKS_PER_SECOND = 254016000000


class Timecode(NamedTuple):
    """Immutable timecode as integer frames."""
    frames: int
    fps_num: int    # frame rate numerator
    fps_den: int    # frame rate denominator

    @classmethod
    def from_seconds(cls, seconds: float, fps_num: int, fps_den: int) -> Timecode:
        """Convert seconds (float OK as input, rounded to nearest frame)."""
        fps = Fraction(fps_num, fps_den)
        frames = int(round(float(Fraction(seconds) * fps)))
        return cls(frames=max(0, frames), fps_num=fps_num, fps_den=fps_den)

    def to_seconds(self) -> float:
        """Convert to float seconds."""
        fps = Fraction(self.fps_num, self.fps_den)
        return float(Fraction(self.frames) / fps)

    def to_fcp_ticks(self) -> int:
        """Convert to FCP internal ticks (integer, never float)."""
        fps = Fraction(self.fps_num, self.fps_den)
        ticks_per_frame = Fraction(FCP_TICKS_PER_SECOND) / fps
        return int(self.frames * ticks_per_frame)

    def to_fcp_rational(self) -> str:
        """Return FCP rational string: 'Nticks/1s' format used in FCPXML."""
        ticks = self.to_fcp_ticks()
        return f"{ticks}/{FCP_TICKS_PER_SECOND}s"

    def __add__(self, other: Timecode) -> Timecode:  # type: ignore[override]
        assert self.fps_num == other.fps_num and self.fps_den == other.fps_den
        return Timecode(self.frames + other.frames, self.fps_num, self.fps_den)

    def __sub__(self, other: Timecode) -> Timecode:  # type: ignore[override]
        assert self.fps_num == other.fps_num and self.fps_den == other.fps_den
        return Timecode(max(0, self.frames - other.frames), self.fps_num, self.fps_den)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Timecode):
            return NotImplemented
        return self.frames < other.frames

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Timecode):
            return NotImplemented
        return self.frames <= other.frames

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Timecode):
            return NotImplemented
        return self.frames > other.frames

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Timecode):
            return NotImplemented
        return self.frames >= other.frames

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Timecode):
            return NotImplemented
        return self.frames == other.frames and self.fps_num == other.fps_num and self.fps_den == other.fps_den

    def __hash__(self) -> int:
        return hash((self.frames, self.fps_num, self.fps_den))


def seconds_to_fcp_rational(seconds: float, fps_num: int, fps_den: int) -> str:
    """Direct conversion from seconds to FCP rational string."""
    tc = Timecode.from_seconds(seconds, fps_num, fps_den)
    return tc.to_fcp_rational()


def frames_to_fcp_rational(frames: int, fps_num: int, fps_den: int) -> str:
    """Convert integer frame count to FCP rational string."""
    tc = Timecode(frames=frames, fps_num=fps_num, fps_den=fps_den)
    return tc.to_fcp_rational()


def fcp_rational_to_seconds(rational_str: str) -> float:
    """Parse '1234567890/254016000000s' → float seconds."""
    s = rational_str.rstrip("s")
    num_str, den_str = s.split("/")
    return float(Fraction(int(num_str), int(den_str)))


def detect_overlap(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> bool:
    """Return True if two time intervals overlap (exclusive end)."""
    return a_start < b_end and b_start < a_end


def format_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for human-readable display."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
