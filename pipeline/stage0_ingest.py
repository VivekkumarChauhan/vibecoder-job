"""pipeline/stage0_ingest.py — Ingest SyncMaster + show_type.txt.

Demuxes all video/audio streams from the SyncMaster recording using PyAV/FFmpeg.
Outputs: IngestResult (per-stream metadata, durations, fps, frame counts).
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from pipeline.schemas import IngestResult, ShowType, StageResult, StreamInfo
from utils.cache import StageCache, hash_payload
from utils.logging_config import get_logger

logger = get_logger(__name__)


def detect_show_type(show_type_path: str | Path) -> tuple[ShowType, list[str]]:
    """Parse show_type.txt → ShowType enum. Returns (show_type, warnings)."""
    warnings: list[str] = []
    path = Path(show_type_path)

    if not path.exists():
        warnings.append(f"show_type.txt not found at {path}; defaulting to UNKNOWN")
        return ShowType.UNKNOWN, warnings

    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        warnings.append(f"Failed to read show_type.txt: {e}; defaulting to UNKNOWN")
        return ShowType.UNKNOWN, warnings

    if not raw:
        warnings.append("show_type.txt is empty; defaulting to UNKNOWN")
        return ShowType.UNKNOWN, warnings

    # Match known show types (exclude UNKNOWN from matching candidates)
    raw_lower = raw.lower()
    known_shows = [st for st in ShowType if st != ShowType.UNKNOWN]
    for st in known_shows:
        if st.value.lower() in raw_lower:
            logger.info("show_type_detected", show_type=st.value, raw=raw)
            return st, warnings

    # Partial keyword checks
    if "nav thethi" in raw_lower or "thethi" in raw_lower:
        return ShowType.NAV_THETHI, warnings
    if "maturity code" in raw_lower or "maturity" in raw_lower:
        return ShowType.MATURITY_CODE, warnings

    warnings.append(
        f"Unrecognized show type '{raw}'; defaulting to UNKNOWN. "
        f"Valid values: {[s.value for s in ShowType if s != ShowType.UNKNOWN]}"
    )
    return ShowType.UNKNOWN, warnings


def _probe_with_ffprobe(source_path: str) -> dict:
    """Run ffprobe to get stream info as JSON."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        source_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        import json
        return json.loads(result.stdout)
    except FileNotFoundError as err:
        raise RuntimeError("ffprobe not found. Please install FFmpeg.") from err


def ingest_syncmaster(
    source_path: str | Path,
    show_type_path: str | Path,
    cache: StageCache | None = None,
) -> StageResult:
    """Stage 0: Ingest SyncMaster and return stream metadata."""
    start_time = time.monotonic()
    source_path = str(source_path)
    warnings: list[str] = []
    errors: list[str] = []

    # Cache check
    cache_key = hash_payload({"source": source_path})
    if cache:
        cached = cache.get_stage("ingest", cache_key)
        if cached:
            logger.info("stage0_cache_hit", source=source_path)
            show_type, st_warnings = detect_show_type(show_type_path)
            warnings.extend(st_warnings)
            result = IngestResult.model_validate(cached)
            result.warnings.extend(warnings)
            return StageResult(
                stage="ingest",
                success=True,
                result=result,
                warnings=warnings,
                duration_s=time.monotonic() - start_time,
            )

    # Validate source path
    if not Path(source_path).exists():
        errors.append(f"Source file not found: {source_path}")
        return StageResult(
            stage="ingest",
            success=False,
            result=None,
            errors=errors,
            duration_s=time.monotonic() - start_time,
        )

    # Detect show type
    show_type, st_warnings = detect_show_type(show_type_path)
    warnings.extend(st_warnings)

    # Probe streams with ffprobe
    try:
        probe_data = _probe_with_ffprobe(source_path)
    except RuntimeError as e:
        # Fallback: try PyAV
        logger.warning("ffprobe_failed", error=str(e), msg="Falling back to PyAV")
        try:
            probe_data = _probe_with_pyav(source_path)
        except Exception as e2:
            errors.append(f"Cannot probe source file: {e2}")
            return StageResult(
                stage="ingest",
                success=False,
                result=None,
                errors=errors,
                duration_s=time.monotonic() - start_time,
            )

    streams_data = probe_data.get("streams", [])
    format_data = probe_data.get("format", {})

    video_streams: list[StreamInfo] = []
    audio_streams: list[StreamInfo] = []
    fps_num, fps_den = 30000, 1001  # default
    total_frames = 0
    duration_s = float(format_data.get("duration", 0.0))

    for s in streams_data:
        codec_type = s.get("codec_type", "unknown")
        idx = int(s.get("index", 0))
        codec = s.get("codec_name", "unknown")
        stream_duration = float(s.get("duration", duration_s) or duration_s)

        if codec_type == "video":
            width = int(s.get("width", 0))
            height = int(s.get("height", 0))
            # Parse frame rate
            r_frame_rate = s.get("r_frame_rate", "30000/1001")
            try:
                fn_str, fd_str = r_frame_rate.split("/")
                fn, fd = int(fn_str), int(fd_str)
                if fd > 0:
                    fps_num, fps_den = fn, fd
            except (ValueError, ZeroDivisionError):
                warnings.append(f"Could not parse fps for stream {idx}: {r_frame_rate}")

            nb_frames = s.get("nb_frames")
            fc = int(nb_frames) if nb_frames and nb_frames != "N/A" else int(stream_duration * fps_num / fps_den)
            total_frames = max(total_frames, fc)

            si = StreamInfo(
                stream_index=idx,
                codec=codec,
                width=width,
                height=height,
                fps=fps_num / fps_den,
                frame_count=fc,
                duration_s=stream_duration,
                stream_type="video",
            )
            video_streams.append(si)

        elif codec_type == "audio":
            si = StreamInfo(
                stream_index=idx,
                codec=codec,
                sample_rate=int(s.get("sample_rate", 44100) or 44100),
                channels=int(s.get("channels", 2) or 2),
                duration_s=stream_duration,
                stream_type="audio",
            )
            audio_streams.append(si)

    if not video_streams:
        errors.append("No video streams found in source file")
        return StageResult(
            stage="ingest",
            success=False,
            result=None,
            errors=errors,
            warnings=warnings,
            duration_s=time.monotonic() - start_time,
        )

    if len(video_streams) < 2:
        warnings.append(
            f"Only {len(video_streams)} video stream(s) found. "
            "Expected multiple cameras in SyncMaster format."
        )

    result = IngestResult(
        source_path=source_path,
        duration_s=duration_s,
        video_streams=video_streams,
        audio_streams=audio_streams,
        frame_rate_num=fps_num,
        frame_rate_den=fps_den,
        total_frames=total_frames,
        warnings=warnings,
        errors=errors,
    )

    if cache:
        cache.set_stage("ingest", cache_key, result.model_dump())

    logger.info(
        "stage0_complete",
        video_streams=len(video_streams),
        audio_streams=len(audio_streams),
        duration_s=round(duration_s, 2),
        show_type=show_type.value,
    )

    return StageResult(
        stage="ingest",
        success=True,
        result=result,
        warnings=warnings,
        errors=errors,
        duration_s=time.monotonic() - start_time,
    )


def _probe_with_pyav(source_path: str) -> dict:
    """Fallback stream probe using PyAV."""
    try:
        import av  # type: ignore[import]
    except ImportError as err:
        raise RuntimeError("PyAV not installed. Run: pip install av") from err

    container = av.open(source_path)
    streams_list = []
    duration_s = float(container.duration or 0) / 1_000_000

    for s in container.streams:
        info: dict = {
            "index": s.index,
            "codec_name": s.codec_context.name if s.codec_context else "unknown",
            "codec_type": "video" if isinstance(s, av.video.stream.VideoStream) else
                          "audio" if isinstance(s, av.audio.stream.AudioStream) else "unknown",
            "duration": float(s.duration or 0) * float(s.time_base or 1),
        }
        if isinstance(s, av.video.stream.VideoStream):
            info["width"] = s.width
            info["height"] = s.height
            r = s.average_rate or s.guessed_rate or 30
            info["r_frame_rate"] = f"{r.numerator}/{r.denominator}"
            info["nb_frames"] = str(s.frames) if s.frames else "N/A"
        elif isinstance(s, av.audio.stream.AudioStream):
            info["sample_rate"] = s.sample_rate
            info["channels"] = s.channels
        streams_list.append(info)

    container.close()
    return {"streams": streams_list, "format": {"duration": str(duration_s)}}
