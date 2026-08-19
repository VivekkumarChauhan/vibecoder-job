"""utils/cache.py — Disk-based JSON cache for all AI outputs.

Key guarantee: every Groq/Whisper/pyannote output is written here immediately.
Re-running director/xml/validator stages hits cache → 0 additional API calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from utils.logging_config import get_logger

logger = get_logger(__name__)


class DiskCache:
    """Simple, reliable disk-based JSON cache.

    Keys are SHA-256 hashes of the input payload.
    Values are arbitrary JSON-serializable dicts stored as .json files.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or os.getenv("CACHE_DIR", "./cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _key_path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def get(self, key: str) -> Any | None:
        path = self._key_path(key)
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                self._hits += 1
                logger.debug("cache_hit", key_prefix=key[:40], path=str(path))
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("cache_read_error", error=str(e), path=str(path))
                return None
        self._misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        path = self._key_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
            logger.debug("cache_write", key_prefix=key[:40], path=str(path))
        except (OSError, TypeError) as e:
            logger.warning("cache_write_error", error=str(e), path=str(path))

    def __contains__(self, key: str) -> bool:
        return self._key_path(key).exists()

    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.json"):
            f.unlink(missing_ok=True)


class StageCache:
    """Named-key cache for per-stage outputs (by stage name + input hash)."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._disk = DiskCache(cache_dir)

    def get_stage(self, stage: str, input_hash: str) -> Any | None:
        return self._disk.get(f"stage:{stage}:{input_hash}")

    def set_stage(self, stage: str, input_hash: str, value: Any) -> None:
        self._disk.set(f"stage:{stage}:{input_hash}", value)

    def get_groq(self, messages_key: str) -> Any | None:
        return self._disk.get(f"groq:{messages_key}")

    def set_groq(self, messages_key: str, value: Any) -> None:
        self._disk.set(f"groq:{messages_key}", value)

    def stats(self) -> dict[str, int]:
        return self._disk.stats()


def hash_payload(data: Any) -> str:
    """Deterministic SHA-256 hash of any JSON-serializable object."""
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def timed_stage(stage_name: str) -> Any:
    """Context manager that logs stage timing."""
    import contextlib

    @contextlib.contextmanager  # type: ignore[arg-type]
    def _ctx() -> Any:
        start = time.monotonic()
        logger.info("stage_start", stage=stage_name)
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            logger.info("stage_complete", stage=stage_name, duration_s=round(elapsed, 3))

    return _ctx()
