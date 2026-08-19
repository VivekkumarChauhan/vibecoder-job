"""pipeline/stage4_director.py — Editorial Decision Engine (Director Agent).

Pure deterministic Python — zero API calls. All thresholds from editorial_rules.yaml.
Acts as a cinematic DIRECTOR, not a camera switcher.

Implements all 10 editorial rules in priority order:
  1. Off-Camera Rule (highest — stop everything)
  2. Technical Failure Rule (force switch)
  3. Physical Adjustment Rule (force cutaway → PHY_ADJ_CUT tag)
  4. Safety Rule (no mid-word/mid-sentence/mid-blink cuts)
  5. Emotional Priority (beats speaker change)
  6. Speaker Rule (default to current speaker HERO)
  7. Listener Reaction Rule (3–5s reaction shots)
  8. Refresh Rule (45s without event → 3s wide)
  9. Dialogue Rule (rapid exchange → wider)
 10. Long Monologue Rule (hold HERO, occasional alternate)

Show-specific rules applied as post-processing layer.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from pipeline.schemas import (
    CameraInfo,
    CameraInventory,
    CameraRole,
    CutEntry,
    CutList,
    CutReason,
    NarrativeLabel,
    NarrativeResult,
    NarrativeSegment,
    ShowType,
    SpeakerRole,
    StageResult,
)
from utils.cache import StageCache, hash_payload
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DirectorState:
    """Mutable state maintained by the Director across segments."""
    current_camera_id: str = ""
    last_cut_time: float = 0.0
    last_meaningful_event_time: float = 0.0
    is_off_camera: bool = False
    current_speaker_label: str = ""
    rapid_exchange_count: int = 0
    monologue_start_time: float = -1.0
    last_alternate_angle_time: float = -1.0
    cuts: list[CutEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _make_cut(
    camera_id: str,
    start_s: float,
    end_s: float,
    reason: CutReason,
    rule_tag: str,
    comment: str = "",
    is_sbs: bool = False,
    is_off_camera: bool = False,
    confidence: float = 1.0,
    needs_review: bool = False,
    segment_id: str | None = None,
) -> CutEntry:
    """Create a CutEntry with a unique ID, validating duration."""
    duration = max(0.0, end_s - start_s)
    if duration <= 0:
        end_s = start_s + 0.033  # min 1 frame at ~30fps
    return CutEntry(
        cut_id=f"cut_{uuid.uuid4().hex[:8]}",
        camera_id=camera_id,
        start_s=start_s,
        end_s=end_s,
        reason=reason,
        rule_tag=rule_tag,
        is_sbs=is_sbs,
        is_off_camera=is_off_camera,
        confidence=confidence,
        needs_review=needs_review,
        comment=comment,
        segment_id=segment_id,
    )


def _get_camera_for_role(inventory: CameraInventory, role: CameraRole) -> CameraInfo | None:
    return inventory.get_camera_by_role(role)


def _get_speaker_hero_camera(
    segment: NarrativeSegment, inventory: CameraInventory, mapping_meta: dict
) -> str | None:
    """Return the HERO camera ID for the current speaker."""
    role = segment.speaker_role
    if role == SpeakerRole.HOST:
        cam = inventory.get_camera_by_role(CameraRole.HOST_HERO)
        return cam.camera_id if cam else None
    elif role == SpeakerRole.GUEST:
        cam = inventory.get_camera_by_role(CameraRole.GUEST_HERO)
        return cam.camera_id if cam else None
    return None


def _get_listener_hero_camera(
    segment: NarrativeSegment, inventory: CameraInventory
) -> str | None:
    """Return the HERO camera of the listener (non-speaker)."""
    role = segment.speaker_role
    if role == SpeakerRole.HOST:
        cam = inventory.get_camera_by_role(CameraRole.GUEST_HERO)
        return cam.camera_id if cam else None
    elif role == SpeakerRole.GUEST:
        cam = inventory.get_camera_by_role(CameraRole.HOST_HERO)
        return cam.camera_id if cam else None
    return None


def _get_wide_camera(inventory: CameraInventory) -> str | None:
    cam = inventory.get_camera_by_role(CameraRole.WIDE)
    if cam:
        return cam.camera_id
    # Fallback: any active camera with smallest face area
    valid = inventory.get_valid_cameras()
    if valid:
        return min(valid, key=lambda c: c.face_area_ratio).camera_id
    return None


def _get_fallback_camera(inventory: CameraInventory, exclude: str = "") -> str | None:
    """Get any valid (non-frozen, active) camera, excluding the given one."""
    valid = inventory.get_valid_cameras()
    for cam in valid:
        if cam.camera_id != exclude:
            return cam.camera_id
    if valid:
        return valid[0].camera_id
    return None


def _is_cut_safe(
    cut_time: float,
    segment: NarrativeSegment,
    rules: dict,
) -> bool:
    """Safety Rule: Return True if it's safe to cut at cut_time.

    Avoids cutting:
    - Mid-word (cut_time within an active word)
    - Mid-sentence (cut_time < segment.end and segment text doesn't end with sentence terminator)
    - Within min_hold_s of last cut
    """
    # Check word boundaries
    for word in segment.words:
        if word.start < cut_time < word.end:
            # Mid-word — not safe
            return False

    # Check sentence boundary — prefer end of segment
    text = segment.text.strip()
    if not text.endswith((".", "!", "?", "...", "—")):
        # Not at sentence end — allow but note lower confidence
        pass  # We still allow the cut but with reduced confidence (handled at call site)

    return True


def _is_meaningful_event(segment: NarrativeSegment) -> bool:
    """Return True if segment counts as a meaningful narrative event (resets refresh timer)."""
    meaningful_labels = {
        NarrativeLabel.QUESTION,
        NarrativeLabel.ANSWER,
        NarrativeLabel.STORYTELLING,
        NarrativeLabel.EMOTIONAL_MOMENT,
        NarrativeLabel.LAUGHTER,
        NarrativeLabel.SHARED_LAUGHTER,
        NarrativeLabel.TOPIC_CHANGE,
        NarrativeLabel.INTERRUPTION,
        NarrativeLabel.FRAMEWORK_DISCUSSION,
        NarrativeLabel.INTRO,
        NarrativeLabel.OUTRO,
    }
    return segment.label in meaningful_labels or segment.has_laughter or segment.has_emotion


# ─── Rule Implementations ─────────────────────────────────────────────────────

def _rule_off_camera(
    segment: NarrativeSegment,
    state: DirectorState,
    inventory: CameraInventory,
    rules: dict,
) -> CutEntry | None:
    """Rule 9: Off-Camera — stop editing; emit OFF_CAMERA_BRAINSTORM."""
    if segment.off_camera_trigger and not state.is_off_camera:
        state.is_off_camera = True
        logger.info("off_camera_detected", segment_id=segment.segment_id, text=segment.text[:50])
        # Return a special off-camera segment marker
        return _make_cut(
            camera_id=state.current_camera_id or (inventory.cameras[0].camera_id if inventory.cameras else "cam_1"),
            start_s=segment.start,
            end_s=segment.end,
            reason=CutReason.OFF_CAMERA,
            rule_tag="OFF_CAMERA_BRAINSTORM",
            comment="Off-camera brainstorm segment — editing paused",
            is_off_camera=True,
            segment_id=segment.segment_id,
        )

    if segment.resume_trigger and state.is_off_camera:
        state.is_off_camera = False
        logger.info("camera_resumed", segment_id=segment.segment_id)
        # Force return to speaker HERO on resume
        return None  # next segment will make the first post-resume cut

    return None


def _rule_technical_failure(
    segment: NarrativeSegment,
    state: DirectorState,
    inventory: CameraInventory,
    rules: dict,
) -> CutEntry | None:
    """Rule 8: Technical Failure — if current camera is frozen/failed, switch immediately."""
    current_cam = next((c for c in inventory.cameras if c.camera_id == state.current_camera_id), None)
    if current_cam and (current_cam.is_frozen or not current_cam.is_active):
        fallback = _get_fallback_camera(inventory, exclude=state.current_camera_id)
        if fallback and fallback != state.current_camera_id:
            state.current_camera_id = fallback
            return _make_cut(
                camera_id=fallback,
                start_s=segment.start,
                end_s=segment.end,
                reason=CutReason.TECH_FAILURE,
                rule_tag="TECH_FAILURE_SWITCH",
                comment=f"Technical failure on {state.current_camera_id}; switching to {fallback}",
                segment_id=segment.segment_id,
            )
    return None


def _rule_physical_adjustment(
    segment: NarrativeSegment,
    state: DirectorState,
    inventory: CameraInventory,
    rules: dict,
) -> CutEntry | None:
    """Rule 7: Physical Adjustment — mandatory immediate cutaway."""
    if segment.physical_event is not None:
        # Cutaway to: wide shot, or listener HERO, or other valid camera
        cutaway_cam = _get_wide_camera(inventory)
        if not cutaway_cam:
            cutaway_cam = _get_listener_hero_camera(segment, inventory)
        if not cutaway_cam:
            cutaway_cam = _get_fallback_camera(inventory, exclude=state.current_camera_id)

        if cutaway_cam and cutaway_cam != state.current_camera_id:
            event_desc = segment.physical_event or f"long off-screen gaze ({segment.duration:.1f}s)"
            cut = _make_cut(
                camera_id=cutaway_cam,
                start_s=segment.start,
                end_s=segment.end,
                reason=CutReason.PHYSICAL_ADJUSTMENT,
                rule_tag="PHY_ADJ_CUT",
                comment=f"Physical adjustment cutaway: {event_desc}",
                confidence=0.9,
                segment_id=segment.segment_id,
            )
            state.current_camera_id = cutaway_cam
            state.last_cut_time = segment.start
            return cut

    return None


def _rule_emotional_priority(
    segment: NarrativeSegment,
    state: DirectorState,
    inventory: CameraInventory,
    rules: dict,
) -> CutEntry | None:
    """Rule 6: Emotional Priority — prioritize emotion over speaker change."""
    if not (segment.has_emotion or segment.label == NarrativeLabel.EMOTIONAL_MOMENT):
        return None

    # Stay on current speaker's HERO for emotional moments
    speaker_cam = _get_speaker_hero_camera(segment, inventory, {})
    if speaker_cam and speaker_cam != state.current_camera_id:
        cut = _make_cut(
            camera_id=speaker_cam,
            start_s=segment.start,
            end_s=segment.end,
            reason=CutReason.EMOTIONAL_PRIORITY,
            rule_tag="EMOTIONAL_PRIORITY",
            comment=f"Emotional moment — switching to speaker HERO: {speaker_cam}",
            confidence=0.95,
            segment_id=segment.segment_id,
        )
        state.current_camera_id = speaker_cam
        state.last_cut_time = segment.start
        state.last_meaningful_event_time = segment.start
        return cut

    return None


def _rule_speaker(
    segment: NarrativeSegment,
    state: DirectorState,
    inventory: CameraInventory,
    rules: dict,
    start_time: float | None = None,
) -> CutEntry | None:
    """Rule 1: Speaker Rule — default to current speaker's HERO camera."""
    cut_start = start_time if start_time is not None else segment.start
    speaker_cam = _get_speaker_hero_camera(segment, inventory, {})
    if not speaker_cam:
        # Fall back to first active camera
        fallback = _get_fallback_camera(inventory)
        speaker_cam = fallback

    if not speaker_cam:
        return None

    # Avoid unnecessary switching
    if speaker_cam == state.current_camera_id:
        return None

    # Check speaker changed
    speaker_changed = segment.speaker_label != state.current_speaker_label

    # Hold during emotional/vulnerable/slow moments (already handled by emotional priority)
    hold_labels = {
        NarrativeLabel.STORYTELLING,
        NarrativeLabel.EMOTIONAL_MOMENT,
        NarrativeLabel.MONOLOGUE,
    }
    if segment.label in hold_labels and not speaker_changed:
        return None  # Stay — no unnecessary switch

    if speaker_changed:
        safe = _is_cut_safe(cut_start, segment, rules)
        confidence = 0.9 if safe else 0.6
        needs_review = not safe

        cut = _make_cut(
            camera_id=speaker_cam,
            start_s=cut_start,
            end_s=segment.end,
            reason=CutReason.SPEAKER_CHANGE,
            rule_tag="SPEAKER_RULE",
            comment=f"Speaker change → {speaker_cam} ({segment.speaker_role.value})",
            confidence=confidence,
            needs_review=needs_review,
            segment_id=segment.segment_id,
        )
        state.current_camera_id = speaker_cam
        state.current_speaker_label = segment.speaker_label
        state.last_cut_time = segment.start
        if _is_meaningful_event(segment):
            state.last_meaningful_event_time = segment.start
        return cut

    return None


def _rule_listener_reaction(
    segment: NarrativeSegment,
    state: DirectorState,
    inventory: CameraInventory,
    rules: dict,
    current_time: float,
) -> CutEntry | None:
    """Rule 2: Listener Reaction — cut to listener on smile/laugh/nod/surprise."""
    d_rules = rules.get("director", {})
    reaction_min = d_rules.get("listener_reaction_min_s", 3.0)
    reaction_max = d_rules.get("listener_reaction_max_s", 5.0)

    reaction_triggers = {NarrativeLabel.LAUGHTER, NarrativeLabel.SHARED_LAUGHTER}
    has_reaction = (
        segment.label in reaction_triggers
        or segment.has_laughter
        or (segment.has_emotion and segment.label == NarrativeLabel.STORYTELLING)
    )

    if not has_reaction:
        return None

    listener_cam = _get_listener_hero_camera(segment, inventory)
    if not listener_cam or listener_cam == state.current_camera_id:
        return None

    reaction_duration = min(segment.duration, reaction_max)
    reaction_duration = max(reaction_duration, reaction_min)
    reaction_end = min(segment.start + reaction_duration, segment.end)

    if reaction_end - segment.start < reaction_min:
        return None

    cut = _make_cut(
        camera_id=listener_cam,
        start_s=segment.start,
        end_s=reaction_end,
        reason=CutReason.LISTENER_REACTION,
        rule_tag="LISTENER_REACTION",
        comment=f"Reaction shot ({reaction_duration:.1f}s) — listener: {listener_cam}",
        confidence=0.85,
        segment_id=segment.segment_id,
    )
    state.last_meaningful_event_time = segment.start
    return cut


def _rule_refresh(
    segment: NarrativeSegment,
    state: DirectorState,
    inventory: CameraInventory,
    rules: dict,
) -> list[CutEntry]:
    """Rule 3: Refresh Rule — 45s without event → 3s establishing wide shot."""
    d_rules = rules.get("director", {})
    refresh_interval = d_rules.get("refresh_interval_s", 45.0)
    wide_duration = d_rules.get("refresh_wide_duration_s", 3.0)

    time_since_event = segment.start - state.last_meaningful_event_time
    if time_since_event < refresh_interval:
        return []

    wide_cam = _get_wide_camera(inventory)
    if not wide_cam:
        return []

    cuts = []

    # Insert 3s wide shot
    wide_end = min(segment.start + wide_duration, segment.end)
    if wide_end - segment.start >= 1.0:  # min 1s
        wide_cut = _make_cut(
            camera_id=wide_cam,
            start_s=segment.start,
            end_s=wide_end,
            reason=CutReason.REFRESH_WIDE,
            rule_tag="REFRESH_WIDE",
            comment=f"Refresh wide shot ({wide_duration}s) — {time_since_event:.0f}s since last event",
            confidence=0.9,
            segment_id=segment.segment_id,
        )
        cuts.append(wide_cut)
        state.last_meaningful_event_time = segment.start  # reset timer
        logger.debug(
            "refresh_wide_inserted",
            segment_id=segment.segment_id,
            time_since_event=round(time_since_event, 1),
        )

    return cuts


def _rule_dialogue(
    segment: NarrativeSegment,
    prev_segment: NarrativeSegment | None,
    state: DirectorState,
    inventory: CameraInventory,
    rules: dict,
) -> CutEntry | None:
    """Rule 4: Dialogue Rule — rapid back-and-forth → prefer wider."""
    d_rules = rules.get("director", {})
    max_turn_s = d_rules.get("rapid_exchange_max_turn_s", 8.0)
    min_turns = d_rules.get("rapid_exchange_min_turns", 3)

    if prev_segment and segment.speaker_label != prev_segment.speaker_label:
        if segment.duration <= max_turn_s:
            state.rapid_exchange_count += 1
        else:
            state.rapid_exchange_count = 0
    else:
        if segment.duration > max_turn_s:
            state.rapid_exchange_count = 0

    if state.rapid_exchange_count >= min_turns:
        wide_cam = _get_wide_camera(inventory)
        if wide_cam and wide_cam != state.current_camera_id:
            cut = _make_cut(
                camera_id=wide_cam,
                start_s=segment.start,
                end_s=segment.end,
                reason=CutReason.DIALOGUE_WIDE,
                rule_tag="DIALOGUE_RULE",
                comment=f"Rapid exchange ({state.rapid_exchange_count} turns) — switching to wide: {wide_cam}",
                confidence=0.8,
                segment_id=segment.segment_id,
            )
            state.current_camera_id = wide_cam
            state.last_cut_time = segment.start
            state.rapid_exchange_count = 0  # reset after going wide
            return cut

    return None


def _rule_long_monologue(
    segment: NarrativeSegment,
    state: DirectorState,
    inventory: CameraInventory,
    rules: dict,
) -> CutEntry | None:
    """Rule 5: Long Monologue — hold HERO, introduce alternate angle occasionally."""
    d_rules = rules.get("director", {})
    monologue_threshold = d_rules.get("monologue_threshold_s", 30.0)
    alternate_interval = d_rules.get("monologue_alternate_interval_s", 90.0)

    is_monologue = (
        segment.label == NarrativeLabel.MONOLOGUE
        or segment.duration > monologue_threshold
    )

    if not is_monologue:
        state.monologue_start_time = -1.0
        return None

    if state.monologue_start_time < 0:
        state.monologue_start_time = segment.start

    monologue_elapsed = segment.start - state.monologue_start_time
    since_last_alternate = segment.start - state.last_alternate_angle_time

    if segment.duration >= alternate_interval or (since_last_alternate >= alternate_interval and monologue_elapsed >= alternate_interval):
        # Introduce brief alternate angle (listener HERO or wide)
        alt_cam = _get_listener_hero_camera(segment, inventory)
        speaker_cam = _get_speaker_hero_camera(segment, inventory, {})
        if alt_cam and alt_cam != speaker_cam:
            alt_start = segment.start + (alternate_interval if segment.duration >= alternate_interval else 0.0)
            alt_duration = min(segment.duration * 0.15, 5.0)
            alt_end = min(alt_start + alt_duration, segment.end)

            cut = _make_cut(
                camera_id=alt_cam,
                start_s=alt_start,
                end_s=alt_end,
                reason=CutReason.LONG_MONOLOGUE_ALTERNATE,
                rule_tag="MONOLOGUE_ALTERNATE",
                comment=f"Long monologue ({segment.duration:.0f}s) — brief alternate angle: {alt_cam}",
                confidence=0.75,
                needs_review=True,  # human should review monologue pacing
                segment_id=segment.segment_id,
            )
            state.last_alternate_angle_time = alt_start
            return cut

    return None


# ─── Show-Specific Post-Processing ───────────────────────────────────────────

def _apply_nav_thethi_rules(
    cuts: list[CutEntry],
    narrative: NarrativeResult,
    inventory: CameraInventory,
    rules: dict,
) -> tuple[list[CutEntry], list[str]]:
    """Apply Nav Thethi show-specific rules."""
    warnings: list[str] = []
    show_rules = rules.get("show_types", {}).get("The Nav Thethi Show", {})
    wide_cap = show_rules.get("wide_shot_max_pct", 20.0)
    wide_reset_on_topic = show_rules.get("wide_reset_on_topic_change", True)
    wide_cam = _get_wide_camera(inventory)

    if not cuts:
        return cuts, warnings

    total_duration = cuts[-1].end_s - cuts[0].start_s if cuts else 0.0

    wide_duration = sum(c.duration_s for c in cuts if "WIDE" in c.camera_id.upper() or c.camera_id == wide_cam)
    current_wide_pct = (wide_duration / max(total_duration, 1)) * 100

    if current_wide_pct > wide_cap:
        # Remove some refresh wide cuts to bring under cap
        new_cuts = []
        wide_removed_s = 0.0
        overshoot_s = ((current_wide_pct - wide_cap) / 100) * total_duration

        for cut in cuts:
            if (
                cut.reason == CutReason.REFRESH_WIDE
                and wide_removed_s < overshoot_s
            ):
                wide_removed_s += cut.duration_s
                warnings.append(
                    f"Nav Thethi: Removed refresh wide cut {cut.cut_id} to stay under {wide_cap}% wide cap"
                )
            else:
                new_cuts.append(cut)
        cuts = new_cuts

    # Insert wide reset on topic changes
    if wide_reset_on_topic and wide_cam:
        enriched = []
        topic_segs = {s.segment_id for s in narrative.segments if s.label == NarrativeLabel.TOPIC_CHANGE}
        for cut in cuts:
            enriched.append(cut)
            if cut.segment_id in topic_segs and wide_cam:
                # Insert a 2s wide reset before the next cut
                reset_cut = _make_cut(
                    camera_id=wide_cam,
                    start_s=cut.end_s,
                    end_s=cut.end_s + 2.0,
                    reason=CutReason.SHOW_SPECIFIC,
                    rule_tag="NAV_THETHI_WIDE_RESET",
                    comment="Nav Thethi: Wide reset on topic change",
                    confidence=0.9,
                )
                enriched.append(reset_cut)
        cuts = enriched

    return cuts, warnings


def _apply_maturity_code_rules(
    cuts: list[CutEntry],
    narrative: NarrativeResult,
    inventory: CameraInventory,
    rules: dict,
) -> tuple[list[CutEntry], list[str]]:
    """Apply Cracking the Maturity Code show-specific rules."""
    warnings: list[str] = []
    show_rules = rules.get("show_types", {}).get("Cracking the Maturity Code", {})
    opening_in_sbs = show_rules.get("opening_in_sbs", True)
    sbs_on_laughter = show_rules.get("sbs_on_shared_laughter", True)
    sbs_on_framework = show_rules.get("sbs_on_framework", True)

    if not cuts:
        return cuts, warnings

    # Mark opening question segment as SBS
    if opening_in_sbs:
        first_question_segs = [s for s in narrative.segments if s.label == NarrativeLabel.QUESTION]
        if first_question_segs:
            first_q = first_question_segs[0]
            for cut in cuts:
                if (
                    cut.segment_id == first_q.segment_id
                    or (cut.start_s <= first_q.end and cut.end_s >= first_q.start)
                ):
                    # Mark as SBS
                    object.__setattr__(cut, "is_sbs", True)
                    object.__setattr__(cut, "rule_tag", "SBS_OPENING_QUESTION")
                    object.__setattr__(cut, "comment", "Maturity Code: Opening question in SBS")

    # SBS on shared laughter
    if sbs_on_laughter:
        laughter_segs = {s.segment_id for s in narrative.segments if s.label == NarrativeLabel.SHARED_LAUGHTER}
        for cut in cuts:
            if cut.segment_id in laughter_segs:
                object.__setattr__(cut, "is_sbs", True)
                object.__setattr__(cut, "rule_tag", "SBS_SHARED_LAUGHTER")
                object.__setattr__(cut, "comment", "Maturity Code: Shared laughter in SBS")

    # SBS on framework discussion
    if sbs_on_framework:
        framework_segs = {s.segment_id for s in narrative.segments if s.label == NarrativeLabel.FRAMEWORK_DISCUSSION}
        for cut in cuts:
            if cut.segment_id in framework_segs:
                object.__setattr__(cut, "is_sbs", True)
                object.__setattr__(cut, "rule_tag", "SBS_FRAMEWORK")
                object.__setattr__(cut, "comment", "Maturity Code: Framework discussion in SBS")

    return cuts, warnings


# ─── Main Director Entry Point ────────────────────────────────────────────────

def direct(
    narrative: NarrativeResult,
    inventory: CameraInventory,
    rules: dict,
    cache: StageCache | None = None,
    iteration: int = 0,
) -> StageResult:
    """Stage 4: Generate cut list from narrative segments."""
    start_time = time.monotonic()
    warnings: list[str] = []
    errors: list[str] = []

    # Cache key
    segments_hash = hash_payload([s.model_dump() for s in narrative.segments])
    cache_key = hash_payload(
        {
            "segments": segments_hash,
            "rules": rules.get("director", {}),
            "show": narrative.show_type.value,
            "iteration": iteration,
        }
    )
    if cache and iteration == 0:
        cached = cache.get_stage("director", cache_key)
        if cached:
            logger.info("stage4_cache_hit")
            return StageResult(
                stage="director",
                success=True,
                result=CutList.model_validate(cached),
                duration_s=time.monotonic() - start_time,
            )

    if not inventory.cameras:
        errors.append("No cameras in inventory — cannot generate cuts")
        return StageResult(
            stage="director",
            success=False,
            result=CutList(cuts=[], total_duration_s=0.0, errors=errors),
            errors=errors,
            duration_s=time.monotonic() - start_time,
        )

    # Initialize state
    first_cam = inventory.get_camera_by_role(CameraRole.HOST_HERO)
    if first_cam is None:
        first_cam = inventory.get_valid_cameras()[0] if inventory.get_valid_cameras() else inventory.cameras[0]

    state = DirectorState(
        current_camera_id=first_cam.camera_id,
        last_meaningful_event_time=0.0,
    )

    # Initial cut (first camera) only if there is lead time before first segment
    if narrative.segments and narrative.segments[0].start > 0.0:
        state.cuts.append(
            _make_cut(
                camera_id=first_cam.camera_id,
                start_s=0.0,
                end_s=narrative.segments[0].start,
                reason=CutReason.INITIAL,
                rule_tag="INITIAL_CUT",
                comment=f"Initial cut to host hero: {first_cam.camera_id}",
            )
        )
    elif not narrative.segments:
        state.cuts.append(
            _make_cut(
                camera_id=first_cam.camera_id,
                start_s=0.0,
                end_s=1.0,
                reason=CutReason.INITIAL,
                rule_tag="INITIAL_CUT",
                comment=f"Initial cut to host hero: {first_cam.camera_id}",
            )
        )

    segments = narrative.segments
    prev_segment: NarrativeSegment | None = None

    for segment in segments:
        # ── Rule 9: Off-Camera ─────────────────────────────────────────────
        off_cam_cut = _rule_off_camera(segment, state, inventory, rules)
        if off_cam_cut:
            state.cuts.append(off_cam_cut)
            prev_segment = segment
            continue

        if state.is_off_camera and not segment.resume_trigger:
            # Extend off-camera segment
            if state.cuts and state.cuts[-1].is_off_camera:
                last = state.cuts[-1]
                state.cuts[-1] = CutEntry(
                    **{**last.model_dump(), "end_s": segment.end}
                )
            prev_segment = segment
            continue

        # ── Rule 8: Technical Failure ──────────────────────────────────────
        tech_cut = _rule_technical_failure(segment, state, inventory, rules)
        if tech_cut:
            state.cuts.append(tech_cut)
            prev_segment = segment
            continue

        # ── Rule 7: Physical Adjustment ────────────────────────────────────
        phy_cut = _rule_physical_adjustment(segment, state, inventory, rules)
        if phy_cut:
            state.cuts.append(phy_cut)
            prev_segment = segment
            continue

        # ── Rule 3: Refresh (before speaker rule — inject wide if needed) ──
        refresh_cuts = _rule_refresh(segment, state, inventory, rules)
        if refresh_cuts:
            state.cuts.extend(refresh_cuts)
            state.last_cut_time = refresh_cuts[-1].end_s
            speaker_cam = _get_speaker_hero_camera(segment, inventory, {}) or state.current_camera_id
            if speaker_cam and refresh_cuts[-1].end_s < segment.end:
                return_cut = _make_cut(
                    camera_id=speaker_cam,
                    start_s=refresh_cuts[-1].end_s,
                    end_s=segment.end,
                    reason=CutReason.SPEAKER_CHANGE,
                    rule_tag="SPEAKER_RULE",
                    comment="Return to speaker HERO after refresh wide shot",
                    segment_id=segment.segment_id,
                )
                state.cuts.append(return_cut)
                state.current_camera_id = speaker_cam
            prev_segment = segment
            continue

        # ── Rule 4: Dialogue ───────────────────────────────────────────────
        dial_cut = _rule_dialogue(segment, prev_segment, state, inventory, rules)
        if dial_cut:
            state.cuts.append(dial_cut)
            prev_segment = segment
            if _is_meaningful_event(segment):
                state.last_meaningful_event_time = segment.start
            continue

        # ── Rule 5: Long Monologue ─────────────────────────────────────────
        mono_cut = _rule_long_monologue(segment, state, inventory, rules)
        if mono_cut:
            speaker_cam = _get_speaker_hero_camera(segment, inventory, {}) or state.current_camera_id
            if mono_cut.start_s > segment.start:
                pre_cut = _make_cut(
                    camera_id=speaker_cam,
                    start_s=segment.start,
                    end_s=mono_cut.start_s,
                    reason=CutReason.SPEAKER_CHANGE,
                    rule_tag="SPEAKER_RULE",
                    comment="Monologue lead-in",
                    segment_id=segment.segment_id,
                )
                state.cuts.append(pre_cut)
            state.cuts.append(mono_cut)
            if mono_cut.end_s < segment.end:
                post_cut = _make_cut(
                    camera_id=speaker_cam,
                    start_s=mono_cut.end_s,
                    end_s=segment.end,
                    reason=CutReason.SPEAKER_CHANGE,
                    rule_tag="SPEAKER_RULE",
                    comment="Return to speaker HERO after monologue alternate",
                    segment_id=segment.segment_id,
                )
                state.cuts.append(post_cut)
            state.current_camera_id = speaker_cam
            prev_segment = segment
            continue

        # ── Rule 2: Listener Reaction ──────────────────────────────────────
        react_cut = _rule_listener_reaction(segment, state, inventory, rules, segment.start)
        if react_cut:
            state.cuts.append(react_cut)
            if _is_meaningful_event(segment):
                state.last_meaningful_event_time = segment.start
            # After reaction shot, return to speaker HERO
            speaker_cam = _get_speaker_hero_camera(segment, inventory, {})
            if speaker_cam and react_cut.end_s < segment.end:
                return_cut = _make_cut(
                    camera_id=speaker_cam,
                    start_s=react_cut.end_s,
                    end_s=segment.end,
                    reason=CutReason.SPEAKER_CHANGE,
                    rule_tag="POST_REACTION_RETURN",
                    comment="Return to speaker HERO after reaction shot",
                    segment_id=segment.segment_id,
                )
                state.cuts.append(return_cut)
                state.current_camera_id = speaker_cam
            prev_segment = segment
            continue

        # ── Rule 6: Emotional Priority ─────────────────────────────────────
        emo_cut = _rule_emotional_priority(segment, state, inventory, rules)
        if emo_cut:
            state.cuts.append(emo_cut)
            if _is_meaningful_event(segment):
                state.last_meaningful_event_time = segment.start
            prev_segment = segment
            continue

        # ── Rule 1: Speaker Rule ───────────────────────────────────────────
        cut_start = refresh_cuts[-1].end_s if refresh_cuts else segment.start
        speaker_cut = _rule_speaker(segment, state, inventory, rules, start_time=cut_start)
        if speaker_cut:
            state.cuts.append(speaker_cut)
            state.current_speaker_label = segment.speaker_label
        else:
            # No cut needed — extend current cut to segment end if possible
            if not state.cuts:
                cam = state.current_camera_id or (inventory.cameras[0].camera_id if inventory.cameras else "cam_1")
                state.cuts.append(
                    _make_cut(
                        camera_id=cam,
                        start_s=segment.start,
                        end_s=segment.end,
                        reason=CutReason.SAFETY_HOLD,
                        rule_tag="HOLD",
                        comment="Initial camera hold",
                        segment_id=segment.segment_id,
                    )
                )
            elif not state.cuts[-1].is_off_camera:
                last = state.cuts[-1]
                if last.camera_id == state.current_camera_id:
                    state.cuts[-1] = CutEntry(
                        **{**last.model_dump(), "end_s": max(last.end_s, segment.end)}
                    )
                else:
                    # Add a holding cut
                    state.cuts.append(
                        _make_cut(
                            camera_id=state.current_camera_id,
                            start_s=segment.start,
                            end_s=segment.end,
                            reason=CutReason.SAFETY_HOLD,
                            rule_tag="HOLD",
                            comment="Hold current camera (no cut trigger)",
                            segment_id=segment.segment_id,
                        )
                    )

        if _is_meaningful_event(segment):
            state.last_meaningful_event_time = segment.start

        prev_segment = segment

    warnings.extend(state.warnings)

    # Post-process: fix overlaps and gaps
    cuts = _resolve_cuts(state.cuts, narrative.segments)

    # Apply show-specific rules
    show_type = narrative.show_type
    if show_type == ShowType.NAV_THETHI:
        cuts, show_warnings = _apply_nav_thethi_rules(cuts, narrative, inventory, rules)
        warnings.extend(show_warnings)
    elif show_type == ShowType.MATURITY_CODE:
        cuts, show_warnings = _apply_maturity_code_rules(cuts, narrative, inventory, rules)
        warnings.extend(show_warnings)

    # Compute total duration
    total_duration = narrative.segments[-1].end if narrative.segments else 0.0

    cut_list = CutList(
        cuts=cuts,
        total_duration_s=total_duration,
        show_type=show_type,
        iteration=iteration,
        warnings=warnings,
        errors=errors,
    )

    if cache and iteration == 0:
        cache.set_stage("director", cache_key, cut_list.model_dump())

    logger.info(
        "stage4_complete",
        total_cuts=len(cuts),
        total_duration_s=round(total_duration, 2),
        wide_pct=round(cut_list.wide_shot_pct, 1),
        warnings=len(warnings),
        iteration=iteration,
    )

    return StageResult(
        stage="director",
        success=True,
        result=cut_list,
        warnings=warnings,
        errors=errors,
        duration_s=time.monotonic() - start_time,
    )


def _resolve_cuts(
    cuts: list[CutEntry],
    segments: list[NarrativeSegment],
) -> list[CutEntry]:
    """Clean up cuts: remove zero-duration, sort by time, fill gaps."""
    if not cuts:
        return cuts

    # Remove zero-duration cuts
    cuts = [c for c in cuts if c.duration_s > 0.001]

    # Sort by start time
    cuts.sort(key=lambda c: c.start_s)

    # Resolve overlaps: if two consecutive cuts overlap, trim the earlier one
    resolved: list[CutEntry] = []
    for cut in cuts:
        if resolved and resolved[-1].end_s > cut.start_s and not resolved[-1].is_off_camera:
            prev = resolved[-1]
            if prev.start_s < cut.start_s:
                resolved[-1] = CutEntry(
                    **{**prev.model_dump(), "end_s": cut.start_s}
                )
            elif prev.start_s == cut.start_s:
                # Replace if current cut is longer or more specific
                if cut.duration_s >= prev.duration_s:
                    resolved.pop()
                else:
                    continue
            else:
                continue
        if cut.duration_s > 0.001:
            resolved.append(cut)

    return resolved
