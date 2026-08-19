"""
pipeline/schemas.py — All Pydantic v2 models (shared inter-stage contracts).

Every stage consumes/produces models defined here.
Stage boundaries are enforced: data passing through must validate against these schemas.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ─── Enums ────────────────────────────────────────────────────────────────────

class CameraRole(str, Enum):
    HOST_HERO = "CAM_HOST_HERO"
    GUEST_HERO = "CAM_GUEST_HERO"
    WIDE = "CAM_WIDE"
    SECONDARY = "CAM_SECONDARY"
    UNKNOWN = "CAM_UNKNOWN"


class SpeakerRole(str, Enum):
    HOST = "host"
    GUEST = "guest"
    UNKNOWN = "unknown"


class NarrativeLabel(str, Enum):
    QUESTION = "question"
    ANSWER = "answer"
    STORYTELLING = "storytelling"
    EMOTIONAL_MOMENT = "emotional_moment"
    LAUGHTER = "laughter"
    INTERRUPTION = "interruption"
    SILENCE = "silence"
    TRANSITION = "transition"
    TOPIC_CHANGE = "topic_change"
    MONOLOGUE = "monologue"
    FRAMEWORK_DISCUSSION = "framework_discussion"
    SHARED_LAUGHTER = "shared_laughter"
    INTRO = "intro"
    OUTRO = "outro"
    OFF_CAMERA = "off_camera"
    UNKNOWN = "unknown"


class CutReason(str, Enum):
    SPEAKER_CHANGE = "speaker_change"
    LISTENER_REACTION = "listener_reaction"
    REFRESH_WIDE = "refresh_wide"
    EMOTIONAL_PRIORITY = "emotional_priority"
    PHYSICAL_ADJUSTMENT = "physical_adjustment"
    TECH_FAILURE = "tech_failure"
    OFF_CAMERA = "off_camera"
    LONG_MONOLOGUE_ALTERNATE = "long_monologue_alternate"
    DIALOGUE_WIDE = "dialogue_wide"
    SHOW_SPECIFIC = "show_specific"
    INITIAL = "initial"
    SAFETY_HOLD = "safety_hold"
    GAP_FILL = "gap_fill"


class ShowType(str, Enum):
    NAV_THETHI = "The Nav Thethi Show"
    MATURITY_CODE = "Cracking the Maturity Code"
    UNKNOWN = "unknown"


# ─── Stage 0: Ingest ──────────────────────────────────────────────────────────

class StreamInfo(BaseModel):
    stream_index: int
    codec: str
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    duration_s: float
    sample_rate: int | None = None
    channels: int | None = None
    stream_type: str  # "video" or "audio"


class IngestResult(BaseModel):
    source_path: str
    duration_s: float
    video_streams: list[StreamInfo]
    audio_streams: list[StreamInfo]
    frame_rate_num: int
    frame_rate_den: int
    total_frames: int
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ─── Stage 1: Camera Discovery ────────────────────────────────────────────────

class CameraInfo(BaseModel):
    camera_id: str                  # e.g. "cam_1"
    stream_index: int
    role: CameraRole = CameraRole.UNKNOWN
    is_active: bool = True
    is_empty: bool = False
    is_frozen: bool = False
    face_detected: bool = False
    face_area_ratio: float = 0.0    # avg face_bbox_area / frame_area
    is_wide_shot: bool = False
    on_screen_text: list[str] = Field(default_factory=list)  # OCR results
    confidence: float = 0.0         # 0–1 role-assignment confidence
    notes: list[str] = Field(default_factory=list)


class CameraInventory(BaseModel):
    cameras: list[CameraInfo]
    role_map: dict[str, str]        # CameraRole → camera_id
    total_cameras: int
    active_cameras: int
    empty_cameras: int
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def get_camera_by_role(self, role: CameraRole) -> CameraInfo | None:
        cam_id = self.role_map.get(role.value)
        if cam_id is None:
            return None
        return next((c for c in self.cameras if c.camera_id == cam_id), None)

    def get_valid_cameras(self) -> list[CameraInfo]:
        return [c for c in self.cameras if c.is_active and not c.is_frozen and not c.is_empty]


# ─── Stage 2: Speaker Mapping ─────────────────────────────────────────────────

class WordToken(BaseModel):
    word: str
    start: float   # seconds
    end: float     # seconds
    confidence: float = 1.0


class TranscriptSegment(BaseModel):
    segment_id: str
    speaker_label: str             # raw pyannote label e.g. "SPEAKER_00"
    start: float
    end: float
    text: str
    words: list[WordToken] = Field(default_factory=list)
    camera_id: str | None = None   # which camera this speaker is on


class SpeakerMapping(BaseModel):
    host_label: str                # e.g. "SPEAKER_00"
    guest_label: str | None = None
    host_camera_id: str | None = None
    guest_camera_id: str | None = None
    all_labels: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class SpeakerMapResult(BaseModel):
    mapping: SpeakerMapping
    transcript: list[TranscriptSegment]
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ─── Stage 3: Narrative Understanding ────────────────────────────────────────

class NarrativeSegment(BaseModel):
    segment_id: str
    speaker_label: str
    speaker_role: SpeakerRole = SpeakerRole.UNKNOWN
    start: float
    end: float
    text: str
    label: NarrativeLabel = NarrativeLabel.UNKNOWN
    sub_labels: list[NarrativeLabel] = Field(default_factory=list)
    confidence: float = 0.0
    has_laughter: bool = False
    has_emotion: bool = False
    has_interruption: bool = False
    physical_event: str | None = None   # e.g. "mic_adjust", "face_scratch"
    words: list[WordToken] = Field(default_factory=list)
    off_camera_trigger: bool = False    # "stop rolling" detected
    resume_trigger: bool = False        # "restart rolling" detected

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info: Any) -> float:
        if "start" in info.data and v < info.data["start"]:
            raise ValueError("end must be >= start")
        return v

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class NarrativeResult(BaseModel):
    segments: list[NarrativeSegment]
    show_type: ShowType = ShowType.UNKNOWN
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ─── Stage 4: Camera Direction (Cut List) ────────────────────────────────────

class CutEntry(BaseModel):
    cut_id: str
    camera_id: str
    start_s: float
    end_s: float
    reason: CutReason
    rule_tag: str                    # e.g. "SPEAKER_RULE", "PHY_ADJ_CUT"
    is_sbs: bool = False             # side-by-side frame?
    is_off_camera: bool = False      # OFF_CAMERA_BRAINSTORM segment?
    confidence: float = 1.0
    needs_review: bool = False       # flagged for human-in-the-loop
    comment: str = ""                # inline XML comment text
    segment_id: str | None = None    # originating narrative segment

    @field_validator("end_s")
    @classmethod
    def end_after_start(cls, v: float, info: Any) -> float:
        if "start_s" in info.data and v < info.data["start_s"]:
            raise ValueError("cut end_s must be >= start_s")
        return v

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


class CutList(BaseModel):
    cuts: list[CutEntry]
    total_duration_s: float
    wide_shot_pct: float = 0.0
    show_type: ShowType = ShowType.UNKNOWN
    quality_score: float | None = None
    iteration: int = 0              # self-correction iteration number
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def compute_wide_pct(self) -> CutList:
        if self.total_duration_s > 0:
            wide_s = sum(
                c.duration_s for c in self.cuts
                if "WIDE" in c.camera_id.upper() or "wide" in c.camera_id.lower()
            )
            self.wide_shot_pct = (wide_s / self.total_duration_s) * 100
        return self


# ─── Stage 4b: Critic Report ─────────────────────────────────────────────────

class RuleViolation(BaseModel):
    cut_id: str
    rule: str
    description: str
    severity: str = "warning"   # "warning" or "error"


class CriticReport(BaseModel):
    violations: list[RuleViolation] = Field(default_factory=list)
    passed: bool = True
    quality_score: float = 1.0
    wide_shot_pct: float = 0.0
    cut_frequency_per_min: float = 0.0
    iteration: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ─── Stage 6: Validation ──────────────────────────────────────────────────────

class ValidationError(BaseModel):
    error_type: str
    description: str
    clip_id: str | None = None
    severity: str = "error"   # "error" or "warning"


class ValidationReport(BaseModel):
    passed: bool
    errors: list[ValidationError] = Field(default_factory=list)
    warnings: list[ValidationError] = Field(default_factory=list)
    clip_count: int = 0
    total_duration_s: float = 0.0
    wide_shot_pct: float = 0.0


# ─── Final: Editing Report ────────────────────────────────────────────────────

class QualityMetrics(BaseModel):
    quality_score: float = 0.0
    cut_frequency_per_min: float = 0.0
    wide_shot_pct: float = 0.0
    wide_shot_cap_pct: float | None = None
    wide_shot_cap_met: bool = True
    rule_violation_count: int = 0
    groq_calls_made: int = 0
    groq_cache_hits: int = 0
    processing_time_per_stage: dict[str, float] = Field(default_factory=dict)
    self_correction_iterations: int = 0


class EditingReport(BaseModel):
    camera_inventory: dict[str, Any] = Field(default_factory=dict)
    speaker_mapping: dict[str, Any] = Field(default_factory=dict)
    cuts: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    off_camera_segments: list[dict[str, Any]] = Field(default_factory=list)
    metadata: QualityMetrics = Field(default_factory=QualityMetrics)
    validation_passed: bool = True
    critic_violations: list[dict[str, Any]] = Field(default_factory=list)


# ─── Stage Result Wrapper ─────────────────────────────────────────────────────

class StageResult(BaseModel):
    """Wraps every stage output with warnings/errors — never raises."""
    stage: str
    success: bool
    result: Any | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
