"""generate_timeline_html.py — HTML/SVG Timeline Visualization (Bonus).

Generates a standalone HTML file with an SVG timeline showing:
  - Camera usage % per camera with color coding
  - Cut positions and durations
  - PHY_ADJ_CUT and OFF_CAMERA_BRAINSTORM markers
  - Rule tags per segment
  - Wide shot % indicator vs cap

No external dependencies beyond stdlib + pipeline.schemas.
"""
from __future__ import annotations

from pathlib import Path

from pipeline.schemas import CameraInventory, CutList
from utils.timecode import format_hms

# Camera color palette
CAMERA_COLORS = [
    "#4f46e5",  # indigo (HOST_HERO)
    "#059669",  # emerald (GUEST_HERO)
    "#d97706",  # amber (WIDE)
    "#dc2626",  # red
    "#7c3aed",  # violet
    "#0891b2",  # cyan
    "#be185d",  # pink
]

RULE_TAG_COLORS = {
    "PHY_ADJ_CUT": "#ef4444",
    "OFF_CAMERA_BRAINSTORM": "#374151",
    "REFRESH_WIDE": "#f59e0b",
    "LISTENER_REACTION": "#10b981",
    "EMOTIONAL_PRIORITY": "#8b5cf6",
    "SPEAKER_RULE": "#3b82f6",
    "TECH_FAILURE_SWITCH": "#ef4444",
    "SBS_OPENING_QUESTION": "#ec4899",
    "SBS_SHARED_LAUGHTER": "#ec4899",
    "SBS_FRAMEWORK": "#ec4899",
    "MONOLOGUE_ALTERNATE": "#6366f1",
    "DIALOGUE_RULE": "#14b8a6",
    "DEFAULT": "#6b7280",
}


def generate_timeline(
    cut_list: CutList,
    inventory: CameraInventory,
    output_path: str = "timeline.html",
) -> None:
    """Generate HTML timeline visualization."""
    cuts = cut_list.cuts
    if not cuts:
        return

    total_duration = cut_list.total_duration_s or 1.0
    cameras = [c.camera_id for c in inventory.cameras]
    cam_color_map = {cam: CAMERA_COLORS[i % len(CAMERA_COLORS)] for i, cam in enumerate(cameras)}

    # SVG dimensions
    svg_width = 1400
    svg_height = 60
    label_width = 100
    timeline_width = svg_width - label_width - 20

    def x_pos(t: float) -> float:
        return label_width + (t / total_duration) * timeline_width

    def cut_width(dur: float) -> float:
        return max(2.0, (dur / total_duration) * timeline_width)

    # Build cut rectangles
    rects_html = []
    markers_html = []

    for cut in cuts:
        x = x_pos(cut.start_s)
        w = cut_width(cut.duration_s)
        color = cam_color_map.get(cut.camera_id, "#6b7280")
        if cut.is_off_camera:
            color = "#1f2937"
        label = cut.rule_tag or "DEFAULT"
        tooltip = (
            f"{cut.camera_id} | {format_hms(cut.start_s)} – {format_hms(cut.end_s)} | "
            f"{cut.reason.value} | {cut.rule_tag}"
        )
        if cut.comment:
            tooltip += f" | {cut.comment[:60]}"

        rect = (
            f'<rect x="{x:.1f}" y="5" width="{w:.1f}" height="50" '
            f'fill="{color}" opacity="0.85" rx="2" '
            f'title="{tooltip}">'
            f"<title>{tooltip}</title>"
            f"</rect>"
        )
        rects_html.append(rect)

        # PHY_ADJ marker (red triangle)
        if "PHY_ADJ" in label:
            marker = f'<polygon points="{x:.1f},5 {x+6:.1f},0 {x+12:.1f},5" fill="#ef4444" title="PHY_ADJ_CUT"/>'
            markers_html.append(marker)
        # OFF_CAMERA marker
        if cut.is_off_camera:
            marker = f'<polygon points="{x:.1f},55 {x+6:.1f},60 {x+12:.1f},55" fill="#374151" title="OFF_CAMERA"/>'
            markers_html.append(marker)
        # SBS indicator
        if cut.is_sbs:
            sbs = f'<line x1="{x:.1f}" y1="5" x2="{x:.1f}" y2="55" stroke="#ec4899" stroke-width="1.5" stroke-dasharray="3,2"/>'
            markers_html.append(sbs)
        # Needs review flag
        if cut.needs_review:
            nr = f'<circle cx="{x + w/2:.1f}" cy="10" r="4" fill="#fbbf24" title="Needs Review"/>'
            markers_html.append(nr)

    # Camera usage stats
    cam_usage: dict[str, float] = {}
    for cut in cuts:
        cam_usage[cut.camera_id] = cam_usage.get(cut.camera_id, 0.0) + cut.duration_s

    cam_stats_html = ""
    for cam, dur in sorted(cam_usage.items(), key=lambda x: -x[1]):
        pct = (dur / total_duration) * 100
        color = cam_color_map.get(cam, "#6b7280")
        cam_stats_html += (
            f'<div style="display:inline-flex;align-items:center;margin-right:16px;font-size:13px;">'
            f'<span style="width:12px;height:12px;background:{color};border-radius:2px;display:inline-block;margin-right:6px;"></span>'
            f"{cam}: {pct:.1f}%"
            f"</div>"
        )

    # Rule tag legend
    legend_html = ""
    for tag, color in list(RULE_TAG_COLORS.items())[:8]:
        legend_html += (
            f'<div style="display:inline-flex;align-items:center;margin-right:12px;font-size:12px;color:#9ca3af;">'
            f'<span style="width:10px;height:10px;background:{color};border-radius:1px;display:inline-block;margin-right:4px;"></span>'
            f"{tag}"
            f"</div>"
        )

    # Time axis ticks
    ticks_html = ""
    n_ticks = 10
    for i in range(n_ticks + 1):
        t = (i / n_ticks) * total_duration
        x = x_pos(t)
        ticks_html += f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="60" stroke="#374151" stroke-width="0.5"/>'
        ticks_html += f'<text x="{x:.1f}" y="-5" font-size="10" fill="#9ca3af" text-anchor="middle">{format_hms(t)}</text>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Narrative Director — Timeline</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #111827; color: #f9fafb; font-family: 'Inter', 'Segoe UI', sans-serif; padding: 32px; }}
  h1 {{ font-size: 22px; font-weight: 700; color: #f9fafb; margin-bottom: 4px; }}
  h2 {{ font-size: 14px; font-weight: 500; color: #6b7280; margin-bottom: 24px; }}
  .card {{ background: #1f2937; border-radius: 12px; padding: 24px; margin-bottom: 20px; border: 1px solid #374151; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
  .stat {{ background: #111827; border-radius: 8px; padding: 14px 18px; }}
  .stat-label {{ font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-value {{ font-size: 28px; font-weight: 700; color: #f9fafb; margin-top: 4px; }}
  .stat-sub {{ font-size: 12px; color: #4b5563; margin-top: 2px; }}
  .timeline-svg {{ overflow-x: auto; }}
  .section-title {{ font-size: 13px; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
  .badge-pass {{ background: #065f46; color: #6ee7b7; }}
  .badge-fail {{ background: #7f1d1d; color: #fca5a5; }}
</style>
</head>
<body>
<h1>🎬 AI Narrative Video Director</h1>
<h2>Timeline Visualization — {cut_list.show_type.value}</h2>

<div class="card">
  <div class="section-title">Episode Stats</div>
  <div class="stat-grid">
    <div class="stat">
      <div class="stat-label">Total Duration</div>
      <div class="stat-value">{format_hms(total_duration)}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Total Cuts</div>
      <div class="stat-value">{len(cuts)}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Wide Shot %</div>
      <div class="stat-value">{cut_list.wide_shot_pct:.1f}%</div>
      <div class="stat-sub">Target: &lt;20% (Nav Thethi) / &lt;40% (Maturity)</div>
    </div>
    <div class="stat">
      <div class="stat-label">Quality Score</div>
      <div class="stat-value">{cut_list.quality_score or 0.0:.2f}</div>
    </div>
    <div class="stat">
      <div class="stat-label">PHY_ADJ Cuts</div>
      <div class="stat-value" style="color:#ef4444">{sum(1 for c in cuts if 'PHY_ADJ' in c.rule_tag)}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Off-Camera Segs</div>
      <div class="stat-value" style="color:#6b7280">{sum(1 for c in cuts if c.is_off_camera)}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Needs Review</div>
      <div class="stat-value" style="color:#fbbf24">{sum(1 for c in cuts if c.needs_review)}</div>
    </div>
    <div class="stat">
      <div class="stat-label">SBS Frames</div>
      <div class="stat-value" style="color:#ec4899">{sum(1 for c in cuts if c.is_sbs)}</div>
    </div>
  </div>
</div>

<div class="card">
  <div class="section-title">Camera Usage</div>
  {cam_stats_html}
</div>

<div class="card">
  <div class="section-title">Timeline</div>
  <div style="margin-bottom:8px;font-size:12px;color:#4b5563;">
    Hover over clips for details.
    🔴 Red triangles = PHY_ADJ_CUT |
    🟡 Yellow dots = Needs Review |
    💗 Pink lines = SBS |
    ⬛ Dark = OFF_CAMERA_BRAINSTORM
  </div>
  <div class="timeline-svg">
    <svg width="{svg_width}" height="{svg_height + 20}" xmlns="http://www.w3.org/2000/svg">
      <g transform="translate(0,15)">
        {ticks_html}
        {" ".join(rects_html)}
        {" ".join(markers_html)}
      </g>
    </svg>
  </div>
</div>

<div class="card">
  <div class="section-title">Rule Tag Legend</div>
  {legend_html}
</div>

<div class="card">
  <div class="section-title">Cut List ({len(cuts)} total)</div>
  <table style="width:100%;border-collapse:collapse;font-size:12px;">
    <thead>
      <tr style="color:#6b7280;text-align:left;border-bottom:1px solid #374151;">
        <th style="padding:8px 12px;">#</th>
        <th style="padding:8px 12px;">Camera</th>
        <th style="padding:8px 12px;">Start</th>
        <th style="padding:8px 12px;">End</th>
        <th style="padding:8px 12px;">Duration</th>
        <th style="padding:8px 12px;">Rule</th>
        <th style="padding:8px 12px;">Flags</th>
        <th style="padding:8px 12px;">Comment</th>
      </tr>
    </thead>
    <tbody>
    {"".join(
        f'<tr style="border-bottom:1px solid #1f2937;{'background:#111827' if i%2==0 else ''}">'
        f'<td style="padding:6px 12px;color:#6b7280">{i+1}</td>'
        f'<td style="padding:6px 12px;"><span style="color:{cam_color_map.get(c.camera_id,"#6b7280")};font-weight:600">{c.camera_id}</span></td>'
        f'<td style="padding:6px 12px;font-family:monospace">{format_hms(c.start_s)}</td>'
        f'<td style="padding:6px 12px;font-family:monospace">{format_hms(c.end_s)}</td>'
        f'<td style="padding:6px 12px;color:#9ca3af">{c.duration_s:.2f}s</td>'
        f'<td style="padding:6px 12px;color:{RULE_TAG_COLORS.get(c.rule_tag,"#9ca3af")};font-size:11px">{c.rule_tag}</td>'
        f'<td style="padding:6px 12px">'
        f'{"<span class=\'badge\' style=\'background:#7f1d1d;color:#fca5a5;margin-right:3px\'>PHY_ADJ</span>" if "PHY_ADJ" in c.rule_tag else ""}'
        f'{"<span class=\'badge\' style=\'background:#064e3b;color:#6ee7b7;margin-right:3px\'>SBS</span>" if c.is_sbs else ""}'
        f'{"<span class=\'badge\' style=\'background:#78350f;color:#fcd34d;margin-right:3px\'>REVIEW</span>" if c.needs_review else ""}'
        f'{"<span class=\'badge\' style=\'background:#111827;color:#9ca3af\'>OFF-CAM</span>" if c.is_off_camera else ""}'
        f'</td>'
        f'<td style="padding:6px 12px;color:#6b7280;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{c.comment}">{c.comment[:80] if c.comment else ""}</td>'
        f"</tr>"
        for i, c in enumerate(cuts)
    )}
    </tbody>
  </table>
</div>

<div style="text-align:center;color:#374151;font-size:11px;margin-top:24px;">
  Generated by AI Narrative Video Director | Canon Studio Technical Assessment
</div>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Timeline written to: {output_path}")
