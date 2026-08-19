"""tests/test_rules_regression.py — One test per editorial rule (all 10 + 2 show-specific).

Each test constructs a scenario that MUST trigger the rule and asserts the expected
tag/cut appears in the cut list. These are the canonical regression tests.
"""
from __future__ import annotations

import pytest

from pipeline.schemas import (
    CameraRole,
    CutEntry,
    CutList,
    CutReason,
    NarrativeLabel,
    NarrativeResult,
    NarrativeSegment,
    ShowType,
    SpeakerRole,
)
from pipeline.stage4_director import direct
from tests.conftest import _make_segment


# ─── Rule 1: Speaker Rule ─────────────────────────────────────────────────────

@pytest.mark.rules
def test_rule1_speaker_rule(three_camera_inventory, standard_rules):
    """Speaker change → cut to new speaker's HERO camera."""
    narrative = NarrativeResult(
        segments=[
            _make_segment("s1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 10.0, "Hello.", NarrativeLabel.QUESTION),
            _make_segment("s2", "SPEAKER_01", SpeakerRole.GUEST, 10.0, 25.0, "Hi.", NarrativeLabel.ANSWER),
        ],
        show_type=ShowType.NAV_THETHI,
    )
    result = direct(narrative, three_camera_inventory, standard_rules)
    cut_list = result.result
    assert cut_list is not None

    # Should have a cut at ~10s to cam_2 (GUEST_HERO)
    speaker_cuts = [c for c in cut_list.cuts if c.rule_tag == "SPEAKER_RULE"]
    assert len(speaker_cuts) >= 1, "Expected at least 1 SPEAKER_RULE cut"

    guest_hero_cut = next((c for c in speaker_cuts if c.camera_id == "cam_2"), None)
    assert guest_hero_cut is not None, "Expected cut to cam_2 (GUEST_HERO) on speaker change"
    assert abs(guest_hero_cut.start_s - 10.0) <= 1.0


@pytest.mark.rules
def test_rule1_no_unnecessary_switch_emotional(three_camera_inventory, standard_rules):
    """Speaker Rule: hold camera during emotional/storytelling — no unnecessary switch."""
    narrative = NarrativeResult(
        segments=[
            _make_segment("s1", "SPEAKER_01", SpeakerRole.GUEST, 0.0, 60.0, "My father passed away...", NarrativeLabel.EMOTIONAL_MOMENT, has_emotion=True),
        ],
        show_type=ShowType.NAV_THETHI,
    )
    result = direct(narrative, three_camera_inventory, standard_rules)
    cut_list = result.result
    # Should NOT switch cameras unnecessarily during emotional moment
    emotional_switch = [c for c in cut_list.cuts if c.rule_tag == "SPEAKER_RULE"]
    assert len(emotional_switch) == 0, "Should not make unnecessary speaker switches during emotional moment"


# ─── Rule 2: Listener Reaction Rule ──────────────────────────────────────────

@pytest.mark.rules
def test_rule2_listener_reaction(three_camera_inventory, standard_rules, laughter_narrative):
    """Laughter segment → cut to listener's HERO for 3–5s."""
    result = direct(laughter_narrative, three_camera_inventory, standard_rules)
    cut_list = result.result
    reaction_cuts = [c for c in cut_list.cuts if c.rule_tag == "LISTENER_REACTION"]
    assert len(reaction_cuts) >= 1, "Expected LISTENER_REACTION cut on laughter segment"

    # Reaction shot should be 3–5s
    rc = reaction_cuts[0]
    assert 3.0 <= rc.duration_s <= 6.0, f"Reaction shot duration {rc.duration_s:.2f}s out of range [3,6]"

    # Should be to the listener camera (host was laughing, listener = guest cam_2)
    assert rc.camera_id == "cam_2", f"Expected listener cam_2, got {rc.camera_id}"


# ─── Rule 3: Refresh Rule ─────────────────────────────────────────────────────

@pytest.mark.rules
def test_rule3_refresh_wide(three_camera_inventory, standard_rules, refresh_needed_narrative):
    """45s without meaningful event → 3s establishing wide shot."""
    result = direct(refresh_needed_narrative, three_camera_inventory, standard_rules)
    cut_list = result.result
    refresh_cuts = [c for c in cut_list.cuts if c.rule_tag == "REFRESH_WIDE"]
    assert len(refresh_cuts) >= 1, "Expected REFRESH_WIDE cut after 45s without event"
    assert refresh_cuts[0].camera_id == "cam_3", f"Expected wide cam_3, got {refresh_cuts[0].camera_id}"
    assert 1.0 <= refresh_cuts[0].duration_s <= 5.0, f"Refresh wide should be ~3s"


# ─── Rule 4: Dialogue Rule ────────────────────────────────────────────────────

@pytest.mark.rules
def test_rule4_dialogue_rule(three_camera_inventory, standard_rules, rapid_dialogue_narrative):
    """Rapid back-and-forth (≥3 short turns) → prefer wider composition."""
    result = direct(rapid_dialogue_narrative, three_camera_inventory, standard_rules)
    cut_list = result.result
    dialogue_cuts = [c for c in cut_list.cuts if c.rule_tag == "DIALOGUE_RULE"]
    assert len(dialogue_cuts) >= 1, "Expected DIALOGUE_RULE cut for rapid exchange"
    # Wide camera should be selected
    assert any(c.camera_id == "cam_3" for c in dialogue_cuts), "Dialogue rule should cut to wide (cam_3)"


# ─── Rule 5: Long Monologue Rule ─────────────────────────────────────────────

@pytest.mark.rules
def test_rule5_long_monologue(three_camera_inventory, standard_rules, long_monologue_narrative):
    """Extended monologue (>90s) → occasional alternate angle."""
    # Need to use a long-enough monologue to trigger alternate angle at 90s mark
    result = direct(long_monologue_narrative, three_camera_inventory, standard_rules)
    cut_list = result.result
    mono_alts = [c for c in cut_list.cuts if c.rule_tag == "MONOLOGUE_ALTERNATE"]
    assert len(mono_alts) >= 1, "Expected MONOLOGUE_ALTERNATE cut for long monologue (115s)"


@pytest.mark.rules
def test_rule5_short_monologue_no_alt(three_camera_inventory, standard_rules):
    """Short monologue (<90s) → no alternate angle injection."""
    narrative = NarrativeResult(
        segments=[
            _make_segment("s1", "SPEAKER_01", SpeakerRole.GUEST, 0.0, 50.0, "Shorter story.", NarrativeLabel.MONOLOGUE),
        ],
        show_type=ShowType.NAV_THETHI,
    )
    result = direct(narrative, three_camera_inventory, standard_rules)
    cut_list = result.result
    mono_alts = [c for c in cut_list.cuts if c.rule_tag == "MONOLOGUE_ALTERNATE"]
    assert len(mono_alts) == 0, "Should not insert alternate angle for short monologue"


# ─── Rule 6: Emotional Priority ──────────────────────────────────────────────

@pytest.mark.rules
def test_rule6_emotional_priority(three_camera_inventory, standard_rules, emotional_narrative):
    """Emotional moment → switch to speaker HERO regardless of who's talking."""
    # Guest has emotional moment — should cut to cam_2 (GUEST_HERO)
    result = direct(emotional_narrative, three_camera_inventory, standard_rules)
    cut_list = result.result
    emo_cuts = [c for c in cut_list.cuts if c.rule_tag == "EMOTIONAL_PRIORITY"]
    assert len(emo_cuts) >= 1, "Expected EMOTIONAL_PRIORITY cut"
    assert emo_cuts[0].camera_id == "cam_2", f"Expected cam_2 (GUEST_HERO), got {emo_cuts[0].camera_id}"


# ─── Rule 7: Physical Adjustment Rule ────────────────────────────────────────

@pytest.mark.rules
def test_rule7_physical_adjustment(three_camera_inventory, standard_rules, physical_adjustment_narrative):
    """Physical event → mandatory cutaway tagged PHY_ADJ_CUT."""
    result = direct(physical_adjustment_narrative, three_camera_inventory, standard_rules)
    cut_list = result.result
    phy_cuts = [c for c in cut_list.cuts if "PHY_ADJ" in c.rule_tag]
    assert len(phy_cuts) >= 1, "Expected PHY_ADJ_CUT for mic adjust event"
    # Must cut away from current camera
    assert phy_cuts[0].camera_id != "cam_1", "PHY_ADJ cutaway should not stay on same camera"


@pytest.mark.rules
def test_rule7_phy_adj_comment_in_xml(three_camera_inventory, standard_rules, physical_adjustment_narrative, sample_ingest, tmp_dir):
    """PHY_ADJ_CUT must appear in XML comments."""
    from pipeline.stage5_xml_generator import generate_fcpxml
    from pipeline.stage4_director import direct

    result = direct(physical_adjustment_narrative, three_camera_inventory, standard_rules)
    cut_list = result.result
    out = str(tmp_dir / "test_output.fcpxml")
    gen_result = generate_fcpxml(cut_list, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules)
    assert gen_result.success

    xml_content = (tmp_dir / "test_output.fcpxml").read_text()
    assert "PHY_ADJ_CUT" in xml_content, "PHY_ADJ_CUT must appear in FCPXML comments"


# ─── Rule 8: Technical Failure Rule ──────────────────────────────────────────

@pytest.mark.rules
def test_rule8_tech_failure_switch(frozen_camera_inventory, standard_rules):
    """Frozen camera → immediate switch to valid angle."""
    narrative = NarrativeResult(
        segments=[
            _make_segment("s1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 30.0, "I'm speaking.", NarrativeLabel.ANSWER),
        ],
        show_type=ShowType.NAV_THETHI,
    )
    result = direct(narrative, frozen_camera_inventory, standard_rules)
    cut_list = result.result
    assert cut_list is not None
    tech_cuts = [c for c in cut_list.cuts if c.rule_tag == "TECH_FAILURE_SWITCH"]
    assert len(tech_cuts) >= 1, "Expected TECH_FAILURE_SWITCH cut for frozen cam_1"
    # Must switch to non-frozen camera
    for tc in tech_cuts:
        assert tc.camera_id == "cam_2", f"Tech failure should switch to cam_2, got {tc.camera_id}"


# ─── Rule 9: Off-Camera Rule ──────────────────────────────────────────────────

@pytest.mark.rules
def test_rule9_off_camera_brainstorm(three_camera_inventory, standard_rules, off_camera_narrative):
    """'Stop rolling' → OFF_CAMERA_BRAINSTORM segment; resume on 'restart rolling'."""
    result = direct(off_camera_narrative, three_camera_inventory, standard_rules)
    cut_list = result.result
    off_cam_cuts = [c for c in cut_list.cuts if c.is_off_camera]
    assert len(off_cam_cuts) >= 1, "Expected OFF_CAMERA_BRAINSTORM segment"
    assert all(c.rule_tag == "OFF_CAMERA_BRAINSTORM" for c in off_cam_cuts)

    # After resume, should have regular cuts
    resume_time = 35.0  # "restart rolling" at 35s
    post_resume_cuts = [c for c in cut_list.cuts if c.start_s >= resume_time and not c.is_off_camera]
    assert len(post_resume_cuts) >= 1, "Should have regular cuts after resume trigger"


@pytest.mark.rules
def test_rule9_off_camera_in_xml(three_camera_inventory, standard_rules, off_camera_narrative, sample_ingest, tmp_dir):
    """OFF_CAMERA_BRAINSTORM must appear in FCPXML."""
    from pipeline.stage4_director import direct
    from pipeline.stage5_xml_generator import generate_fcpxml

    result = direct(off_camera_narrative, three_camera_inventory, standard_rules)
    cut_list = result.result
    out = str(tmp_dir / "test_off_cam.fcpxml")
    gen_result = generate_fcpxml(cut_list, three_camera_inventory, sample_ingest, "test.mp4", out, standard_rules)
    assert gen_result.success

    xml_content = (tmp_dir / "test_off_cam.fcpxml").read_text()
    assert "OFF_CAMERA_BRAINSTORM" in xml_content, "OFF_CAMERA_BRAINSTORM must appear in FCPXML"


# ─── Rule 10: Safety Rule ─────────────────────────────────────────────────────

@pytest.mark.rules
def test_rule10_safety_no_mid_word_cut(three_camera_inventory, standard_rules):
    """Safety Rule: cut should not happen mid-word."""
    from pipeline.stage4_director import _is_cut_safe
    from pipeline.schemas import WordToken

    seg = _make_segment("s1", "SPEAKER_00", SpeakerRole.HOST, 0.0, 10.0, "Hello world.")
    seg = NarrativeSegment(
        **{**seg.model_dump(), "words": [
            WordToken(word="Hello", start=0.0, end=0.5),
            WordToken(word="world", start=0.5, end=1.0),
        ]}
    )
    # Cut mid-word
    assert not _is_cut_safe(0.3, seg, standard_rules), "Mid-word cut should not be safe"
    # Cut between words
    assert _is_cut_safe(1.5, seg, standard_rules), "Between-word cut should be safe"


# ─── Show-Specific: Nav Thethi — Wide Cap ────────────────────────────────────

@pytest.mark.rules
def test_show_nav_thethi_wide_cap(three_camera_inventory, standard_rules):
    """Nav Thethi Show: wide shot usage must stay below 20% of runtime."""
    from pipeline.stage4b_critic import critique

    # Create a cut list with excessive wide shots (30% wide)
    total_dur = 100.0
    cuts = [
        CutEntry(cut_id="c1", camera_id="cam_1", start_s=0.0, end_s=70.0, reason=CutReason.SPEAKER_CHANGE, rule_tag="SPEAKER_RULE"),
        CutEntry(cut_id="c2", camera_id="cam_3", start_s=70.0, end_s=100.0, reason=CutReason.REFRESH_WIDE, rule_tag="REFRESH_WIDE"),  # 30% wide
    ]
    cut_list = CutList(cuts=cuts, total_duration_s=total_dur, show_type=ShowType.NAV_THETHI)

    narrative = NarrativeResult(segments=[], show_type=ShowType.NAV_THETHI)
    result = critique(cut_list, narrative, three_camera_inventory, standard_rules)
    critic = result.result
    assert critic is not None

    wide_violations = [v for v in critic.violations if v.rule == "SHOW_SPECIFIC_WIDE_CAP"]
    assert len(wide_violations) >= 1, "Expected wide cap violation (30% > 20% cap)"


@pytest.mark.rules
def test_show_nav_thethi_wide_cap_compliant(three_camera_inventory, standard_rules):
    """Nav Thethi Show: 15% wide — should pass."""
    from pipeline.stage4b_critic import critique

    cuts = [
        CutEntry(cut_id="c1", camera_id="cam_1", start_s=0.0, end_s=85.0, reason=CutReason.SPEAKER_CHANGE, rule_tag="SPEAKER_RULE"),
        CutEntry(cut_id="c2", camera_id="cam_3", start_s=85.0, end_s=100.0, reason=CutReason.REFRESH_WIDE, rule_tag="REFRESH_WIDE"),  # 15% wide
    ]
    cut_list = CutList(cuts=cuts, total_duration_s=100.0, show_type=ShowType.NAV_THETHI)

    narrative = NarrativeResult(segments=[], show_type=ShowType.NAV_THETHI)
    result = critique(cut_list, narrative, three_camera_inventory, standard_rules)
    critic = result.result

    wide_violations = [v for v in critic.violations if v.rule == "SHOW_SPECIFIC_WIDE_CAP"]
    assert len(wide_violations) == 0, "15% wide should not violate 20% cap"


# ─── Show-Specific: Cracking the Maturity Code — Opening SBS ────────────────

@pytest.mark.rules
def test_show_maturity_opening_sbs(three_camera_inventory, standard_rules):
    """Cracking the Maturity Code: opening question must remain in SBS."""
    from pipeline.stage4b_critic import critique

    # Opening question NOT in SBS — should violate
    narrative = NarrativeResult(
        segments=[
            NarrativeSegment(
                segment_id="seg_q1",
                speaker_label="SPEAKER_00",
                speaker_role=SpeakerRole.HOST,
                start=0.0, end=15.0,
                text="What is maturity?",
                label=NarrativeLabel.QUESTION,
                confidence=0.9,
            ),
        ],
        show_type=ShowType.MATURITY_CODE,
    )

    cuts = [
        CutEntry(
            cut_id="c1",
            camera_id="cam_1",
            start_s=0.0, end_s=15.0,
            reason=CutReason.SPEAKER_CHANGE,
            rule_tag="SPEAKER_RULE",
            segment_id="seg_q1",
            is_sbs=False,  # NOT SBS — should violate
        ),
    ]
    cut_list = CutList(cuts=cuts, total_duration_s=60.0, show_type=ShowType.MATURITY_CODE)

    result = critique(cut_list, narrative, three_camera_inventory, standard_rules)
    critic = result.result

    sbs_violations = [v for v in critic.violations if v.rule == "MATURITY_CODE_OPENING_SBS"]
    assert len(sbs_violations) >= 1, "Expected SBS violation for non-SBS opening question"
