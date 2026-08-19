"""tests/conftest.py — Shared fixtures for all test suites.

Provides synthetic test data (no real video needed) for:
  - Camera inventories
  - Transcript/narrative segments
  - Cut lists
  - Groq mock fixtures
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pipeline.schemas import (
    CameraInfo,
    CameraInventory,
    CameraRole,
    CutEntry,
    CutList,
    CutReason,
    IngestResult,
    NarrativeLabel,
    NarrativeResult,
    NarrativeSegment,
    ShowType,
    SpeakerMapping,
    SpeakerMapResult,
    SpeakerRole,
    StreamInfo,
    TranscriptSegment,
    WordToken,
)
from utils.cache import StageCache


# ─── Base fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def cache(tmp_path: Path) -> StageCache:
    return StageCache(cache_dir=str(tmp_path / "cache"))


@pytest.fixture
def show_type_nav(tmp_path: Path) -> str:
    p = tmp_path / "show_type.txt"
    p.write_text("The Nav Thethi Show")
    return str(p)


@pytest.fixture
def show_type_maturity(tmp_path: Path) -> str:
    p = tmp_path / "show_type.txt"
    p.write_text("Cracking the Maturity Code")
    return str(p)


@pytest.fixture
def show_type_empty(tmp_path: Path) -> str:
    p = tmp_path / "show_type.txt"
    p.write_text("")
    return str(p)


@pytest.fixture
def show_type_corrupt(tmp_path: Path) -> str:
    p = tmp_path / "show_type.txt"
    p.write_text("Unknown Podcast Show XYZ 12345")
    return str(p)


# ─── Camera inventory fixtures ────────────────────────────────────────────────

@pytest.fixture
def three_camera_inventory() -> CameraInventory:
    """Standard 3-camera setup: HOST_HERO, GUEST_HERO, WIDE."""
    return CameraInventory(
        cameras=[
            CameraInfo(
                camera_id="cam_1",
                stream_index=0,
                role=CameraRole.HOST_HERO,
                is_active=True,
                face_detected=True,
                face_area_ratio=0.15,
            ),
            CameraInfo(
                camera_id="cam_2",
                stream_index=1,
                role=CameraRole.GUEST_HERO,
                is_active=True,
                face_detected=True,
                face_area_ratio=0.13,
            ),
            CameraInfo(
                camera_id="cam_3",
                stream_index=2,
                role=CameraRole.WIDE,
                is_active=True,
                face_detected=False,
                face_area_ratio=0.02,
                is_wide_shot=True,
            ),
        ],
        role_map={
            CameraRole.HOST_HERO.value: "cam_1",
            CameraRole.GUEST_HERO.value: "cam_2",
            CameraRole.WIDE.value: "cam_3",
        },
        total_cameras=3,
        active_cameras=3,
        empty_cameras=0,
    )


@pytest.fixture
def single_camera_inventory() -> CameraInventory:
    """Edge case: only 1 camera."""
    return CameraInventory(
        cameras=[
            CameraInfo(
                camera_id="cam_1",
                stream_index=0,
                role=CameraRole.HOST_HERO,
                is_active=True,
                face_detected=True,
                face_area_ratio=0.15,
            ),
        ],
        role_map={CameraRole.HOST_HERO.value: "cam_1"},
        total_cameras=1,
        active_cameras=1,
        empty_cameras=0,
        warnings=["Only 1 camera available — limited editing options"],
    )


@pytest.fixture
def empty_inventory() -> CameraInventory:
    """Edge case: no cameras."""
    return CameraInventory(
        cameras=[],
        role_map={},
        total_cameras=0,
        active_cameras=0,
        empty_cameras=0,
        errors=["No cameras detected"],
    )


@pytest.fixture
def frozen_camera_inventory() -> CameraInventory:
    """Inventory with one frozen camera."""
    return CameraInventory(
        cameras=[
            CameraInfo(
                camera_id="cam_1",
                stream_index=0,
                role=CameraRole.HOST_HERO,
                is_active=False,
                is_frozen=True,
                face_detected=False,
            ),
            CameraInfo(
                camera_id="cam_2",
                stream_index=1,
                role=CameraRole.GUEST_HERO,
                is_active=True,
                face_detected=True,
                face_area_ratio=0.12,
            ),
        ],
        role_map={
            CameraRole.HOST_HERO.value: "cam_1",
            CameraRole.GUEST_HERO.value: "cam_2",
        },
        total_cameras=2,
        active_cameras=1,
        empty_cameras=0,
        warnings=["cam_1 is frozen"],
    )


# ─── Ingest result fixture ────────────────────────────────────────────────────

@pytest.fixture
def sample_ingest() -> IngestResult:
    return IngestResult(
        source_path="test_syncmaster.mp4",
        duration_s=120.0,
        video_streams=[
            StreamInfo(stream_index=0, codec="h264", width=1920, height=1080, fps=29.97, frame_count=3596, duration_s=120.0, stream_type="video"),
            StreamInfo(stream_index=1, codec="h264", width=1920, height=1080, fps=29.97, frame_count=3596, duration_s=120.0, stream_type="video"),
            StreamInfo(stream_index=2, codec="h264", width=1920, height=1080, fps=29.97, frame_count=3596, duration_s=120.0, stream_type="video"),
        ],
        audio_streams=[
            StreamInfo(stream_index=3, codec="aac", sample_rate=48000, channels=2, duration_s=120.0, stream_type="audio"),
        ],
        frame_rate_num=30000,
        frame_rate_den=1001,
        total_frames=3596,
    )


# ─── Transcript/speaker fixtures ──────────────────────────────────────────────

@pytest.fixture
def basic_speaker_result() -> SpeakerMapResult:
    """Host asks question, guest answers."""
    return SpeakerMapResult(
        mapping=SpeakerMapping(
            host_label="SPEAKER_00",
            guest_label="SPEAKER_01",
            all_labels=["SPEAKER_00", "SPEAKER_01"],
            confidence=0.9,
        ),
        transcript=[
            TranscriptSegment(
                segment_id="seg_0001",
                speaker_label="SPEAKER_00",
                start=0.0, end=8.0,
                text="Welcome to the show. How are you today?",
                words=[
                    WordToken(word="Welcome", start=0.0, end=0.5),
                    WordToken(word="to", start=0.5, end=0.7),
                    WordToken(word="the", start=0.7, end=0.9),
                    WordToken(word="show", start=0.9, end=1.2),
                ],
            ),
            TranscriptSegment(
                segment_id="seg_0002",
                speaker_label="SPEAKER_01",
                start=8.0, end=20.0,
                text="I'm doing great. Really excited to be here.",
            ),
            TranscriptSegment(
                segment_id="seg_0003",
                speaker_label="SPEAKER_00",
                start=20.0, end=40.0,
                text="Let me tell you a story about leadership.",
            ),
        ],
    )


# ─── Narrative segment fixtures ───────────────────────────────────────────────

def _make_segment(
    seg_id: str,
    speaker: str,
    role: SpeakerRole,
    start: float,
    end: float,
    text: str,
    label: NarrativeLabel = NarrativeLabel.ANSWER,
    **kwargs: Any,
) -> NarrativeSegment:
    return NarrativeSegment(
        segment_id=seg_id,
        speaker_label=speaker,
        speaker_role=role,
        start=start,
        end=end,
        text=text,
        label=label,
        confidence=0.9,
        **kwargs,
    )


@pytest.fixture
def basic_narrative() -> NarrativeResult:
    """Standard host/guest podcast narrative."""
    return NarrativeResult(
        segments=[
            _make_segment("seg_1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 10.0, "Welcome.", NarrativeLabel.INTRO),
            _make_segment("seg_2", "SPEAKER_01", SpeakerRole.GUEST, 10.0, 25.0, "Thanks.", NarrativeLabel.ANSWER),
            _make_segment("seg_3", "SPEAKER_00", SpeakerRole.HOST, 25.0, 60.0, "Tell me about yourself.", NarrativeLabel.QUESTION),
            _make_segment("seg_4", "SPEAKER_01", SpeakerRole.GUEST, 60.0, 120.0, "Sure, I grew up...", NarrativeLabel.STORYTELLING),
        ],
        show_type=ShowType.NAV_THETHI,
    )


@pytest.fixture
def physical_adjustment_narrative() -> NarrativeResult:
    """Narrative where speaker has a physical adjustment event."""
    return NarrativeResult(
        segments=[
            _make_segment("seg_1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 15.0, "Let me adjust my mic.", NarrativeLabel.ANSWER, physical_event="mic_adjust"),
            _make_segment("seg_2", "SPEAKER_01", SpeakerRole.GUEST, 15.0, 40.0, "Ok continuing.", NarrativeLabel.ANSWER),
        ],
        show_type=ShowType.NAV_THETHI,
    )


@pytest.fixture
def off_camera_narrative() -> NarrativeResult:
    """Narrative with 'stop rolling' trigger."""
    return NarrativeResult(
        segments=[
            _make_segment("seg_1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 10.0, "Hey stop rolling.", NarrativeLabel.OFF_CAMERA, off_camera_trigger=True),
            _make_segment("seg_2", "SPEAKER_00", SpeakerRole.HOST, 10.0, 30.0, "Off camera brainstorm.", NarrativeLabel.UNKNOWN),
            _make_segment("seg_3", "SPEAKER_00", SpeakerRole.HOST, 30.0, 35.0, "Restart rolling.", NarrativeLabel.UNKNOWN, resume_trigger=True),
            _make_segment("seg_4", "SPEAKER_01", SpeakerRole.GUEST, 35.0, 60.0, "Back on camera.", NarrativeLabel.ANSWER),
        ],
        show_type=ShowType.NAV_THETHI,
    )


@pytest.fixture
def laughter_narrative() -> NarrativeResult:
    """Narrative with shared laughter (triggers listener reaction rule)."""
    return NarrativeResult(
        segments=[
            _make_segment("seg_1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 15.0, "That's hilarious!", NarrativeLabel.LAUGHTER, has_laughter=True),
            _make_segment("seg_2", "SPEAKER_01", SpeakerRole.GUEST, 15.0, 40.0, "I know right?", NarrativeLabel.ANSWER),
        ],
        show_type=ShowType.NAV_THETHI,
    )


@pytest.fixture
def long_monologue_narrative() -> NarrativeResult:
    """Narrative with a long monologue (>90s)."""
    return NarrativeResult(
        segments=[
            _make_segment("seg_1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 5.0, "Tell me your story.", NarrativeLabel.QUESTION),
            _make_segment("seg_2", "SPEAKER_01", SpeakerRole.GUEST, 5.0, 120.0, "Sure, so it started...", NarrativeLabel.MONOLOGUE),
        ],
        show_type=ShowType.NAV_THETHI,
    )


@pytest.fixture
def rapid_dialogue_narrative() -> NarrativeResult:
    """Rapid back-and-forth (triggers dialogue rule)."""
    return NarrativeResult(
        segments=[
            _make_segment("seg_1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 4.0, "Really?", NarrativeLabel.QUESTION),
            _make_segment("seg_2", "SPEAKER_01", SpeakerRole.GUEST, 4.0, 7.0, "Yes!", NarrativeLabel.ANSWER),
            _make_segment("seg_3", "SPEAKER_00", SpeakerRole.HOST, 7.0, 11.0, "How?", NarrativeLabel.QUESTION),
            _make_segment("seg_4", "SPEAKER_01", SpeakerRole.GUEST, 11.0, 14.0, "Like this.", NarrativeLabel.ANSWER),
            _make_segment("seg_5", "SPEAKER_00", SpeakerRole.HOST, 14.0, 18.0, "Wow.", NarrativeLabel.QUESTION),
        ],
        show_type=ShowType.NAV_THETHI,
    )


@pytest.fixture
def emotional_narrative() -> NarrativeResult:
    """Emotional moment (triggers emotional priority rule)."""
    return NarrativeResult(
        segments=[
            _make_segment("seg_1", "SPEAKER_01", SpeakerRole.GUEST, 0.0, 30.0, "I lost my father that year.", NarrativeLabel.EMOTIONAL_MOMENT, has_emotion=True),
        ],
        show_type=ShowType.NAV_THETHI,
    )


@pytest.fixture
def refresh_needed_narrative() -> NarrativeResult:
    """50s of silence/unknown — should trigger refresh wide rule."""
    return NarrativeResult(
        segments=[
            _make_segment("seg_1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 5.0, "Ok.", NarrativeLabel.ANSWER),
            _make_segment("seg_2", "SPEAKER_00", SpeakerRole.HOST, 5.0, 55.0, "Umm...", NarrativeLabel.SILENCE),
            _make_segment("seg_3", "SPEAKER_01", SpeakerRole.GUEST, 55.0, 80.0, "Right.", NarrativeLabel.ANSWER),
        ],
        show_type=ShowType.NAV_THETHI,
    )


# ─── Standard rules fixture ───────────────────────────────────────────────────

@pytest.fixture
def standard_rules() -> dict:
    """Standard editorial rules for testing."""
    return {
        "groq": {"budget_per_run": 50, "max_retries": 3, "base_delay_seconds": 0.01},
        "director": {
            "listener_reaction_min_s": 3.0,
            "listener_reaction_max_s": 5.0,
            "refresh_interval_s": 45.0,
            "refresh_wide_duration_s": 3.0,
            "monologue_threshold_s": 30.0,
            "monologue_alternate_interval_s": 90.0,
            "physical_adjustment_gaze_threshold_s": 3.0,
            "safety_min_hold_s": 0.5,
            "blink_buffer_frames": 3,
            "rapid_exchange_max_turn_s": 8.0,
            "rapid_exchange_min_turns": 3,
            "max_self_correction_iterations": 3,
        },
        "show_types": {
            "The Nav Thethi Show": {
                "wide_shot_max_pct": 20.0,
                "pacing": "cinematic",
                "wide_reset_on_topic_change": True,
                "emotional_close_up_priority": True,
            },
            "Cracking the Maturity Code": {
                "wide_shot_max_pct": 40.0,
                "opening_in_sbs": True,
                "sbs_on_shared_laughter": True,
                "sbs_on_framework": True,
                "pacing": "dynamic",
            },
        },
        "xml": {
            "frame_rate_numerator": 30000,
            "frame_rate_denominator": 1001,
            "add_cut_comments": True,
            "fcpxml_version": "1.10",
        },
        "validation": {
            "allow_zero_duration_gap": False,
            "max_overlap_frames": 0,
        },
        "narrative": {
            "max_retries_on_bad_json": 3,
            "batch_size_segments": 20,
            "safe_default_label": "unknown",
        },
        "hitl": {"confidence_threshold": 0.6, "enabled": False},
    }


# ─── Mock Groq client ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_groq_client() -> MagicMock:
    """Mock Groq client that returns safe defaults without making API calls."""
    mock = MagicMock()
    mock.calls_used = 0
    mock.calls_remaining = 50

    def mock_call(messages=None, model=None, safe_default=None, **kwargs):
        # Return a plausible narrative classification
        mock.calls_used += 1
        return {
            "segments": [
                {
                    "segment_id": "seg_0001",
                    "label": "answer",
                    "sub_labels": [],
                    "confidence": 0.85,
                    "has_laughter": False,
                    "has_emotion": False,
                    "has_interruption": False,
                }
            ]
        }

    mock.call.side_effect = mock_call
    return mock


@pytest.fixture
def rate_limited_groq_client() -> MagicMock:
    """Mock Groq client that simulates rate limiting."""
    mock = MagicMock()
    mock.calls_used = 0
    call_count = [0]

    def mock_call_rate_limited(messages=None, safe_default=None, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:
            raise Exception("Rate limit exceeded: 429 Too Many Requests")
        return safe_default  # After retries, return safe default

    mock.call.side_effect = mock_call_rate_limited
    return mock


@pytest.fixture
def budget_exceeded_groq_client() -> MagicMock:
    """Mock Groq client that immediately hits budget."""
    from utils.groq_client import BudgetExceededError
    mock = MagicMock()
    mock.calls_used = 50
    mock.calls_remaining = 0
    mock.call.side_effect = BudgetExceededError("Budget exceeded")
    return mock
