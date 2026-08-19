"""utils/ocr_utils.py — Tesseract OCR wrapper for on-screen text detection.

Purpose: supplementary signal (not primary pipeline) to detect:
- Tablets/monitors visible in frame (trigger physical adjustment rule)
- On-screen lower-thirds/labels (help confirm camera role or speaker identity)

Soft-fails gracefully if Tesseract is not installed.
"""
from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

from utils.logging_config import get_logger

logger = get_logger(__name__)

_TESSERACT_AVAILABLE: bool | None = None


def _check_tesseract() -> bool:
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is not None:
        return _TESSERACT_AVAILABLE
    try:
        import pytesseract  # type: ignore[import]

        # Allow override via env var (Windows)
        cmd = os.getenv("TESSERACT_CMD")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

        # Test call
        pytesseract.get_tesseract_version()
        _TESSERACT_AVAILABLE = True
        logger.info("tesseract_available", version=str(pytesseract.get_tesseract_version()))
    except Exception as e:
        _TESSERACT_AVAILABLE = False
        logger.warning("tesseract_unavailable", error=str(e), msg="OCR features disabled")
    return _TESSERACT_AVAILABLE  # type: ignore[return-value]


def extract_text_from_frame(
    frame: np.ndarray,
    min_confidence: int = 60,
) -> list[dict[str, Any]]:
    """Extract text tokens from a BGR/RGB frame using Tesseract.

    Returns list of {text, confidence, bbox} dicts.
    Returns [] if Tesseract is unavailable or no text found.
    """
    if not _check_tesseract():
        return []

    try:
        import pytesseract  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        # Convert numpy array to PIL Image
        if frame.ndim == 3 and frame.shape[2] == 3:
            pil_img = Image.fromarray(frame)
        else:
            pil_img = Image.fromarray(frame)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pytesseract.image_to_data(
                pil_img,
                output_type=pytesseract.Output.DICT,
                config="--psm 3",  # fully automatic page segmentation
            )

        results: list[dict[str, Any]] = []
        n = len(data["text"])
        for i in range(n):
            conf = int(data["conf"][i])
            text = data["text"][i].strip()
            if conf >= min_confidence and text:
                results.append(
                    {
                        "text": text,
                        "confidence": conf,
                        "bbox": {
                            "x": data["left"][i],
                            "y": data["top"][i],
                            "w": data["width"][i],
                            "h": data["height"][i],
                        },
                    }
                )
        return results

    except Exception as e:
        logger.warning("ocr_error", error=str(e))
        return []


def detect_screen_text(
    frame: np.ndarray,
    frame_area: int,
    min_text_area_ratio: float = 0.03,
    min_confidence: int = 60,
) -> tuple[bool, list[str]]:
    """Detect if frame contains on-screen text (tablet/monitor indicator).

    Returns (has_screen_text: bool, text_tokens: list[str]).
    """
    tokens = extract_text_from_frame(frame, min_confidence=min_confidence)
    if not tokens:
        return False, []

    # Check if text area ratio exceeds threshold
    total_text_area = sum(t["bbox"]["w"] * t["bbox"]["h"] for t in tokens)
    ratio = total_text_area / max(frame_area, 1)
    has_screen = ratio >= min_text_area_ratio

    texts = [t["text"] for t in tokens]
    if has_screen:
        logger.debug(
            "screen_text_detected",
            ratio=round(ratio, 4),
            token_count=len(tokens),
            sample=texts[:5],
        )
    return has_screen, texts


def extract_lower_third(
    frame: np.ndarray,
    lower_fraction: float = 0.25,
    min_confidence: int = 60,
) -> list[str]:
    """Extract text from the lower portion of frame (lower-thirds/labels)."""
    h = frame.shape[0]
    lower_start = int(h * (1 - lower_fraction))
    lower_frame = frame[lower_start:, :, :]
    tokens = extract_text_from_frame(lower_frame, min_confidence=min_confidence)
    return [t["text"] for t in tokens]
