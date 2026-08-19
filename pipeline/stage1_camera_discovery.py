"""pipeline/stage1_camera_discovery.py — Detect cameras, roles, occupancy.

Uses OpenCV (DNN face detector + frame-diff + perceptual hashing) as primary.
Tesseract OCR as supplementary signal for on-screen text / lower-thirds.
Groq vision (optional) as an additional confirmation layer.

Output: CameraInventory with role_map {CameraRole → camera_id}.
"""
from __future__ import annotations

import contextlib
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.schemas import (
    CameraInfo,
    CameraInventory,
    CameraRole,
    IngestResult,
    StageResult,
)
from utils.cache import StageCache, hash_payload
from utils.groq_client import GroqClient, build_json_system_prompt
from utils.logging_config import get_logger
from utils.ocr_utils import detect_screen_text, extract_lower_third

logger = get_logger(__name__)


# ─── OpenCV helpers ───────────────────────────────────────────────────────────

def _load_face_detector() -> Any:
    """Load OpenCV DNN face detector (SSD ResNet). Falls back to Haar."""
    try:
        import cv2  # type: ignore[import]

        # Try DNN detector first (more accurate)
        # Use built-in OpenCV Haar as universal fallback
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(haar_path)
        if cascade.empty():
            raise RuntimeError("Haar cascade not found")
        logger.info("face_detector_loaded", type="haar_cascade")
        return ("haar", cascade)
    except Exception as e:
        logger.warning("face_detector_failed", error=str(e))
        return None


def _detect_faces_haar(frame_gray: Any, cascade: Any, min_area_ratio: float) -> list[dict]:
    """Detect faces using Haar cascade. Returns list of {bbox, area_ratio}."""
    try:
        import cv2  # type: ignore[import]

        h, w = frame_gray.shape[:2]
        frame_area = h * w
        faces = cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        results = []
        if len(faces) == 0:
            return results
        for x, y, fw, fh in faces:
            area_ratio = (fw * fh) / max(frame_area, 1)
            if area_ratio >= min_area_ratio:
                results.append({"bbox": (x, y, fw, fh), "area_ratio": area_ratio})
        return results
    except Exception:
        return []


def _compute_frame_diff(prev: Any, curr: Any) -> float:
    """Mean absolute pixel difference between two grayscale frames."""
    try:
        import cv2  # type: ignore[import]

        diff = cv2.absdiff(prev, curr)
        return float(np.mean(diff))
    except Exception:
        return 0.0


def _perceptual_hash(frame: Any) -> Any:
    """Compute perceptual hash of a frame for frozen-frame detection."""
    try:
        import imagehash  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        pil_img = Image.fromarray(frame)
        return imagehash.phash(pil_img)
    except Exception:
        return None


def _extract_frames_ffmpeg(
    source_path: str,
    stream_index: int,
    n_frames: int,
    duration_s: float,
) -> list[np.ndarray]:
    """Extract N evenly-spaced frames from a specific video stream using FFmpeg."""
    frames = []
    if duration_s <= 0 or n_frames <= 0:
        return frames

    interval = duration_s / (n_frames + 1)
    timestamps = [interval * (i + 1) for i in range(n_frames)]

    for ts in timestamps:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ts),
            "-i", source_path,
            "-map", f"0:v:{stream_index}",
            "-frames:v", "1",
            "-q:v", "2",
            tmp_path,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=30
            )
            if result.returncode == 0:
                import cv2  # type: ignore[import]

                frame = cv2.imread(tmp_path)
                if frame is not None:
                    frames.append(frame)
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug("frame_extract_error", ts=ts, error=str(e))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return frames


def _extract_frames_pyav(
    source_path: str,
    stream_index: int,
    n_frames: int,
    duration_s: float,
) -> list[np.ndarray]:
    """Extract frames using PyAV (fallback to FFmpeg)."""
    try:
        import av  # type: ignore[import]

        container = av.open(source_path)
        video_streams = [s for s in container.streams if s.type == "video"]
        if stream_index >= len(video_streams):
            container.close()
            return []

        stream = video_streams[stream_index]
        total = stream.frames or int(duration_s * 30)
        step = max(1, total // (n_frames + 1))

        frames = []
        frame_num = 0
        target_frames = {step * (i + 1) for i in range(n_frames)}

        container.seek(0)
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame_num in target_frames:
                    img = frame.to_ndarray(format="bgr24")
                    frames.append(img)
                frame_num += 1
                if frame_num > max(target_frames, default=0) + step:
                    break

        container.close()
        return frames
    except Exception as e:
        logger.warning("pyav_extract_failed", error=str(e))
        return []


def _analyze_camera_frames(
    frames: list[np.ndarray],
    rules: dict,
    detector: Any,
) -> dict[str, Any]:
    """Analyze a list of frames to characterize camera occupancy and quality."""
    import cv2  # type: ignore[import]

    face_area_ratios = []
    frame_diffs = []
    hashes = []
    on_screen_texts: list[str] = []
    is_frozen_flags = []

    min_area_ratio = rules.get("face_min_area_ratio", 0.005)
    hash_distance_thresh = rules.get("frozen_frame_hash_distance", 5)
    frozen_consecutive_min = rules.get("frozen_frame_consecutive_min", 3)
    ocr_min_text_area = rules.get("ocr_tablet_min_text_area_ratio", 0.03)
    ocr_conf = rules.get("ocr_confidence_threshold", 60)

    prev_gray = None
    prev_hash = None
    consecutive_frozen = 0

    for frame in frames:
        h, w = frame.shape[:2]
        frame_area = h * w
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection
        if detector is not None:
            dtype, det = detector
            if dtype == "haar":
                detected = _detect_faces_haar(gray, det, min_area_ratio)
                if detected:
                    face_area_ratios.append(max(d["area_ratio"] for d in detected))
                else:
                    face_area_ratios.append(0.0)

        # Frame diff
        if prev_gray is not None:
            diff = _compute_frame_diff(prev_gray, gray)
            frame_diffs.append(diff)
        prev_gray = gray

        # Perceptual hash (frozen frame detection)
        ph = _perceptual_hash(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if ph is not None:
            if prev_hash is not None:
                try:
                    dist = ph - prev_hash
                    if dist <= hash_distance_thresh:
                        consecutive_frozen += 1
                    else:
                        consecutive_frozen = 0
                    is_frozen_flags.append(consecutive_frozen >= frozen_consecutive_min)
                except Exception:
                    is_frozen_flags.append(False)
            hashes.append(ph)
            prev_hash = ph

        # OCR (supplementary — tablet/monitor detection)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        has_screen, texts = detect_screen_text(rgb_frame, frame_area, ocr_min_text_area, ocr_conf)
        if has_screen:
            on_screen_texts.extend(texts)
        # Also check lower thirds
        lower_texts = extract_lower_third(rgb_frame)
        on_screen_texts.extend(lower_texts)

    avg_face_area = float(np.mean(face_area_ratios)) if face_area_ratios else 0.0
    avg_diff = float(np.mean(frame_diffs)) if frame_diffs else 0.0
    is_frozen = bool(any(is_frozen_flags)) and len([f for f in is_frozen_flags if f]) >= frozen_consecutive_min
    face_detected = avg_face_area >= min_area_ratio

    return {
        "avg_face_area_ratio": avg_face_area,
        "avg_frame_diff": avg_diff,
        "is_frozen": is_frozen,
        "face_detected": face_detected,
        "on_screen_texts": list(set(on_screen_texts))[:10],  # deduplicate, cap
    }


def _assign_roles(
    camera_analyses: list[dict[str, Any]],
    rules: dict,
) -> dict[str, str]:
    """Heuristically assign camera roles from analysis results.

    Returns {CameraRole.value → camera_id}.
    """
    wide_max_ratio = rules.get("wide_shot_face_area_ratio_max", 0.05)

    # Sort by face area descending (larger face = closer/hero shot)
    active_cams = [
        c for c in camera_analyses
        if c["is_active"] and not c["is_empty"] and not c["is_frozen"]
    ]

    if not active_cams:
        return {}

    sorted_by_face = sorted(active_cams, key=lambda x: x["avg_face_area_ratio"], reverse=True)

    role_map: dict[str, str] = {}

    # Wide shot: smallest face area ratio below threshold
    wide_candidates = [c for c in active_cams if c["avg_face_area_ratio"] <= wide_max_ratio]
    if wide_candidates:
        wide_cam = min(wide_candidates, key=lambda x: x["avg_face_area_ratio"])
        role_map[CameraRole.WIDE.value] = wide_cam["camera_id"]

    # Hero cameras: two largest face-area cameras (host, guest)
    hero_cams = [c for c in sorted_by_face if c["camera_id"] != role_map.get(CameraRole.WIDE.value)]
    if hero_cams:
        role_map[CameraRole.HOST_HERO.value] = hero_cams[0]["camera_id"]
    if len(hero_cams) >= 2:
        role_map[CameraRole.GUEST_HERO.value] = hero_cams[1]["camera_id"]

    logger.info("camera_roles_assigned", role_map=role_map)
    return role_map


def _groq_confirm_roles(
    camera_analyses: list[dict[str, Any]],
    role_map: dict[str, str],
    groq_client: GroqClient | None,
) -> tuple[dict[str, str], list[str]]:
    """Optionally use Groq to confirm/refine role assignments from heuristics."""
    warnings: list[str] = []
    if groq_client is None:
        return role_map, warnings

    try:
        schema_desc = (
            "{\n"
            '  "CAM_HOST_HERO": "cam_N or null",\n'
            '  "CAM_GUEST_HERO": "cam_N or null",\n'
            '  "CAM_WIDE": "cam_N or null",\n'
            '  "reasoning": "brief explanation"\n'
            "}"
        )
        system_prompt = build_json_system_prompt(
            schema_desc,
            {
                "CAM_HOST_HERO": "cam_1",
                "CAM_GUEST_HERO": "cam_3",
                "CAM_WIDE": "cam_5",
                "reasoning": "cam_1 has largest face area ratio indicating close-up hero shot",
            },
        )

        cam_summary = [
            {
                "camera_id": c["camera_id"],
                "avg_face_area_ratio": round(c["avg_face_area_ratio"], 4),
                "avg_frame_diff": round(c["avg_frame_diff"], 2),
                "is_frozen": c["is_frozen"],
                "face_detected": c["face_detected"],
                "on_screen_texts": c["on_screen_texts"][:3],
            }
            for c in camera_analyses
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Given these camera analysis results, assign camera roles:\n"
                    f"{cam_summary}\n\n"
                    f"Current heuristic assignment: {role_map}\n\n"
                    "Confirm or correct the role assignment. "
                    "CAM_HOST_HERO = person who appears most dominant/close-up. "
                    "CAM_WIDE = wide establishing shot. "
                    "Return JSON only."
                ),
            },
        ]

        result = groq_client.call(messages=messages, safe_default=None)
        if result and isinstance(result, dict):
            # Update role_map with Groq's confirmation
            for role_key in [CameraRole.HOST_HERO.value, CameraRole.GUEST_HERO.value, CameraRole.WIDE.value]:
                if role_key in result and result[role_key] and result[role_key] != "null" and any(c["camera_id"] == result[role_key] for c in camera_analyses):
                    role_map[role_key] = result[role_key]
            reasoning = result.get("reasoning", "")
            logger.info("groq_role_confirmation", reasoning=reasoning[:100])

    except Exception as e:
        warnings.append(f"Groq role confirmation failed (using heuristic result): {e}")

    return role_map, warnings


def discover_cameras(
    ingest_result: IngestResult,
    source_path: str,
    rules: dict,
    cache: StageCache | None = None,
    groq_client: GroqClient | None = None,
) -> StageResult:
    """Stage 1: Discover cameras, detect roles, occupancy, frozen frames."""
    start_time = time.monotonic()
    warnings: list[str] = []
    errors: list[str] = []

    # Cache check
    cache_key = hash_payload({"source": source_path, "rules": rules})
    if cache:
        cached = cache.get_stage("camera_discovery", cache_key)
        if cached:
            logger.info("stage1_cache_hit")
            return StageResult(
                stage="camera_discovery",
                success=True,
                result=CameraInventory.model_validate(cached),
                duration_s=time.monotonic() - start_time,
            )

    video_streams = ingest_result.video_streams
    if not video_streams:
        errors.append("No video streams available for camera discovery")
        return StageResult(
            stage="camera_discovery",
            success=False,
            result=_empty_inventory(errors),
            errors=errors,
            duration_s=time.monotonic() - start_time,
        )

    n_frames = rules.get("frames_to_sample", 20)
    activity_threshold = rules.get("activity_diff_threshold", 10.0)

    detector = _load_face_detector()
    if detector is None:
        warnings.append("Face detector unavailable; using frame-diff only for occupancy detection")

    camera_analyses: list[dict[str, Any]] = []


    for i, stream in enumerate(video_streams):
        cam_id = f"cam_{i + 1}"
        logger.info(
            "analyzing_camera",
            camera_id=cam_id,
            stream_index=stream.stream_index,
        )

        # Extract frames
        frames = _extract_frames_ffmpeg(
            source_path,
            stream_index=i,  # 0-indexed among video streams for ffmpeg map
            n_frames=n_frames,
            duration_s=stream.duration_s,
        )

        if not frames:
            logger.warning("no_frames_extracted", camera_id=cam_id, stream_index=stream.stream_index)
            frames = _extract_frames_pyav(source_path, i, n_frames, stream.duration_s)

        if not frames:
            warnings.append(f"Could not extract frames from {cam_id} (stream {stream.stream_index})")
            camera_analyses.append(
                {
                    "camera_id": cam_id,
                    "stream_index": stream.stream_index,
                    "avg_face_area_ratio": 0.0,
                    "avg_frame_diff": 0.0,
                    "is_frozen": False,
                    "face_detected": False,
                    "on_screen_texts": [],
                    "is_active": False,
                    "is_empty": True,
                }
            )
            continue

        analysis = _analyze_camera_frames(frames, rules, detector)
        is_active = analysis["avg_frame_diff"] >= activity_threshold or analysis["face_detected"]
        is_empty = not analysis["face_detected"] and analysis["avg_frame_diff"] < activity_threshold

        analysis.update(
            {
                "camera_id": cam_id,
                "stream_index": stream.stream_index,
                "is_active": is_active,
                "is_empty": is_empty,
            }
        )
        camera_analyses.append(analysis)
        logger.info(
            "camera_analyzed",
            camera_id=cam_id,
            face_detected=analysis["face_detected"],
            face_area=round(analysis["avg_face_area_ratio"], 4),
            is_frozen=analysis["is_frozen"],
            is_active=is_active,
        )

    # Assign roles heuristically
    role_map = _assign_roles(camera_analyses, rules)

    # Optionally confirm with Groq
    role_map, groq_warnings = _groq_confirm_roles(camera_analyses, role_map, groq_client)
    warnings.extend(groq_warnings)

    # Build CameraInfo objects
    cameras: list[CameraInfo] = []
    for analysis in camera_analyses:
        # Find assigned role
        assigned_role = CameraRole.UNKNOWN
        for role_val, cam_id in role_map.items():
            if cam_id == analysis["camera_id"]:
                with contextlib.suppress(ValueError):
                    assigned_role = CameraRole(role_val)

        cam_info = CameraInfo(
            camera_id=analysis["camera_id"],
            stream_index=analysis["stream_index"],
            role=assigned_role,
            is_active=analysis.get("is_active", True),
            is_empty=analysis.get("is_empty", False),
            is_frozen=analysis.get("is_frozen", False),
            face_detected=analysis.get("face_detected", False),
            face_area_ratio=analysis.get("avg_face_area_ratio", 0.0),
            is_wide_shot=assigned_role == CameraRole.WIDE,
            on_screen_text=analysis.get("on_screen_texts", []),
            confidence=0.8 if assigned_role != CameraRole.UNKNOWN else 0.3,
        )
        cameras.append(cam_info)

    if not role_map.get(CameraRole.HOST_HERO.value):
        warnings.append("Could not identify CAM_HOST_HERO — editorial rules will use first active camera")

    if not role_map.get(CameraRole.WIDE.value):
        warnings.append("No wide shot camera detected — refresh and dialogue rules will use fallback")

    active_count = sum(1 for c in cameras if c.is_active and not c.is_empty)
    empty_count = sum(1 for c in cameras if c.is_empty)

    inventory = CameraInventory(
        cameras=cameras,
        role_map=role_map,
        total_cameras=len(cameras),
        active_cameras=active_count,
        empty_cameras=empty_count,
        warnings=warnings,
        errors=errors,
    )

    if cache:
        cache.set_stage("camera_discovery", cache_key, inventory.model_dump())

    logger.info(
        "stage1_complete",
        total=len(cameras),
        active=active_count,
        empty=empty_count,
        role_map=role_map,
    )

    return StageResult(
        stage="camera_discovery",
        success=True,
        result=inventory,
        warnings=warnings,
        errors=errors,
        duration_s=time.monotonic() - start_time,
    )


def _empty_inventory(errors: list[str]) -> CameraInventory:
    return CameraInventory(
        cameras=[],
        role_map={},
        total_cameras=0,
        active_cameras=0,
        empty_cameras=0,
        errors=errors,
    )
