"""utils/groq_client.py — Groq API wrapper with retry/backoff, budget, and cache.

Design:
- Every call is cache-checked first → 0 API calls on cache hit
- Rate-limit / server errors: exponential backoff up to max_delay
- Malformed JSON: retry up to max_retries, then return safe_default
- Budget exceeded: raise BudgetExceededError (never silently skip)
- All responses are Pydantic-validated against the provided schema type
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from utils.cache import StageCache
from utils.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class BudgetExceededError(RuntimeError):
    """Raised when Groq call budget is exhausted."""


class GroqClientConfig(BaseModel):
    api_key: str = ""
    model: str = "llama-3.3-70b-versatile"
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    budget_per_run: int = 50
    max_retries: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 30.0


class GroqClient:
    """Groq API client with retry, backoff, budget enforcement, and caching."""

    def __init__(self, config: GroqClientConfig, cache: StageCache) -> None:
        self.config = config
        self.cache = cache
        self._used = 0
        self._vision_available: bool | None = None  # lazily detected
        self._client: Any = None  # groq.Groq, imported lazily

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import groq  # type: ignore[import]
                api_key = self.config.api_key or os.getenv("GROQ_API_KEY", "")
                if not api_key:
                    logger.warning("groq_no_api_key", msg="GROQ_API_KEY not set; Groq calls will fail")
                self._client = groq.Groq(api_key=api_key)
            except ImportError:
                logger.error("groq_import_error", msg="groq package not installed")
                raise
        return self._client

    @property
    def calls_used(self) -> int:
        return self._used

    @property
    def calls_remaining(self) -> int:
        return self.config.budget_per_run - self._used

    def _make_cache_key(self, messages: list[dict[str, Any]], model: str) -> str:
        payload = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def call(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        safe_default: Any = None,
        response_model: type[T] | None = None,
        temperature: float = 0.1,
    ) -> Any:
        """Make a Groq API call (JSON mode) with retry, cache, and budget."""
        model = model or self.config.model
        cache_key = self._make_cache_key(messages, model)

        # 1. Cache check
        cached = self.cache.get_groq(cache_key)
        if cached is not None:
            logger.debug("groq_cache_hit", model=model, cache_key=cache_key[:16])
            if response_model is not None:
                try:
                    return response_model.model_validate(cached)
                except ValidationError:
                    pass  # fall through to raw dict return
            return cached

        # 2. Budget check
        if self._used >= self.config.budget_per_run:
            logger.error(
                "groq_budget_exceeded",
                used=self._used,
                budget=self.config.budget_per_run,
            )
            raise BudgetExceededError(
                f"Groq budget exceeded: {self._used}/{self.config.budget_per_run} calls used"
            )

        # 3. Call with retry/backoff
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=4096,
                )
                self._used += 1
                content = response.choices[0].message.content or "{}"

                # Parse JSON
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "groq_bad_json",
                        attempt=attempt,
                        error=str(e),
                        content_preview=content[:200],
                    )
                    last_error = e
                    self._backoff(attempt)
                    continue

                # Validate against Pydantic model if provided
                if response_model is not None:
                    try:
                        validated = response_model.model_validate(data)
                        self.cache.set_groq(cache_key, data)
                        logger.info(
                            "groq_success",
                            model=model,
                            attempt=attempt,
                            calls_used=self._used,
                        )
                        return validated
                    except ValidationError as e:
                        logger.warning(
                            "groq_schema_validation_failed",
                            attempt=attempt,
                            errors=str(e),
                        )
                        last_error = e
                        self._backoff(attempt)
                        continue

                self.cache.set_groq(cache_key, data)
                logger.info("groq_success", model=model, attempt=attempt, calls_used=self._used)
                return data

            except Exception as e:
                error_name = type(e).__name__
                logger.warning("groq_api_error", attempt=attempt, error=error_name, msg=str(e))
                last_error = e
                # Check if rate limited
                if "rate" in str(e).lower() or "429" in str(e):
                    self._backoff(attempt, extra_delay=10.0)
                else:
                    self._backoff(attempt)

        # 4. All retries exhausted → safe default
        logger.error(
            "groq_all_retries_failed",
            model=model,
            retries=self.config.max_retries,
            last_error=str(last_error),
        )
        return safe_default

    def call_vision(
        self,
        messages: list[dict[str, Any]],
        safe_default: Any = None,
    ) -> Any:
        """Make a Groq vision API call. Falls back gracefully if model unavailable."""
        return self.call(
            messages=messages,
            model=self.config.vision_model,
            safe_default=safe_default,
        )

    def _backoff(self, attempt: int, extra_delay: float = 0.0) -> None:
        delay = min(
            self.config.base_delay_seconds * (2**attempt) + extra_delay,
            self.config.max_delay_seconds,
        )
        logger.debug("groq_backoff", delay_s=delay, attempt=attempt)
        time.sleep(delay)


def build_json_system_prompt(schema_description: str, example: dict[str, Any]) -> str:
    """Build a system prompt that instructs Groq to return valid JSON only."""
    example_str = json.dumps(example, indent=2)
    return (
        "You are a precise JSON-output assistant. "
        "Respond ONLY with a valid JSON object matching this schema:\n\n"
        f"{schema_description}\n\n"
        f"Example output:\n{example_str}\n\n"
        "Do not include markdown, code fences, or explanatory text. "
        "Return only the JSON object."
    )
