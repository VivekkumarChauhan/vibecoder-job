"""tests/test_groq_failure_modes.py — Groq API failure mode tests.

Tests:
  - Rate limit response (429) → retry then safe default
  - Malformed JSON response → retry then safe default
  - API timeout → safe default
  - Budget exceeded → BudgetExceededError (caught upstream)
  - Zero Groq calls when cache is warm
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from utils.cache import StageCache
from utils.groq_client import BudgetExceededError, GroqClient, GroqClientConfig


def _make_client(budget: int = 10, tmp_dir=None) -> tuple[GroqClient, StageCache]:
    cache = StageCache(cache_dir=str(tmp_dir or "/tmp/test_cache"))
    config = GroqClientConfig(
        api_key="test_key",
        budget_per_run=budget,
        max_retries=2,
        base_delay_seconds=0.01,
        max_delay_seconds=0.1,
    )
    client = GroqClient(config=config, cache=cache)
    return client, cache


@pytest.mark.groq
def test_groq_budget_exceeded(tmp_path):
    """Budget exceeded raises BudgetExceededError."""
    client, _ = _make_client(budget=0, tmp_dir=tmp_path)
    with pytest.raises(BudgetExceededError):
        client.call(messages=[{"role": "user", "content": "test"}])


@pytest.mark.groq
def test_groq_rate_limit_retry_then_safe_default(tmp_path):
    """Rate limit (429) → retries → returns safe default on exhaustion."""
    client, _ = _make_client(budget=10, tmp_dir=tmp_path)

    call_count = [0]
    def fake_create(*args, **kwargs):
        call_count[0] += 1
        raise Exception("429 Rate limit exceeded")

    with patch.object(client, "_get_client") as mock_get:
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = fake_create
        mock_get.return_value = mock_groq

        result = client.call(
            messages=[{"role": "user", "content": "test"}],
            safe_default={"fallback": True},
        )

    assert result == {"fallback": True}, "Should return safe default on rate limit exhaustion"
    assert call_count[0] >= 1  # Retried at least once


@pytest.mark.groq
def test_groq_malformed_json_retry_safe_default(tmp_path):
    """Malformed JSON response → retries → safe default returned."""
    client, _ = _make_client(budget=10, tmp_dir=tmp_path)

    def fake_create(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "NOT VALID JSON }{{{ garbage"
        return mock_resp

    with patch.object(client, "_get_client") as mock_get:
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = fake_create
        mock_get.return_value = mock_groq

        result = client.call(
            messages=[{"role": "user", "content": "classify this"}],
            safe_default={"label": "unknown"},
        )

    assert result == {"label": "unknown"}, "Should return safe default for malformed JSON"


@pytest.mark.groq
def test_groq_timeout_safe_default(tmp_path):
    """API timeout → safe default."""
    import socket
    client, _ = _make_client(budget=10, tmp_dir=tmp_path)

    def fake_create(*args, **kwargs):
        raise TimeoutError("Request timed out")

    with patch.object(client, "_get_client") as mock_get:
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = fake_create
        mock_get.return_value = mock_groq

        result = client.call(
            messages=[{"role": "user", "content": "classify"}],
            safe_default="unknown",
        )

    assert result == "unknown"


@pytest.mark.groq
def test_groq_cache_hit_no_api_call(tmp_path):
    """Cache hit → zero additional API calls."""
    client, cache = _make_client(budget=5, tmp_dir=tmp_path)

    # Pre-populate cache with a known result
    messages = [{"role": "user", "content": "classify this segment"}]
    model = "llama-3.3-70b-versatile"

    import hashlib
    payload = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    cache_key = hashlib.sha256(payload.encode()).hexdigest()
    cache_value = {"segments": [{"label": "answer", "confidence": 0.9}]}
    cache.set_groq(cache_key, cache_value)

    api_called = [False]
    def fake_create(*args, **kwargs):
        api_called[0] = True
        raise AssertionError("API should NOT be called on cache hit")

    with patch.object(client, "_get_client") as mock_get:
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = fake_create
        mock_get.return_value = mock_groq

        result = client.call(messages=messages, model=model)

    assert not api_called[0], "API must not be called on cache hit"
    assert result == cache_value, "Should return cached value"
    assert client.calls_used == 0, "calls_used must be 0 on cache hit"


@pytest.mark.groq
def test_groq_schema_validation_retry_then_fallback(tmp_path):
    """Groq returns valid JSON but wrong schema → retries → safe default."""
    from pydantic import BaseModel
    client, _ = _make_client(budget=10, tmp_dir=tmp_path)

    class ExpectedSchema(BaseModel):
        label: str
        confidence: float

    def fake_create(*args, **kwargs):
        mock_resp = MagicMock()
        # Valid JSON but wrong schema (missing required fields)
        mock_resp.choices[0].message.content = '{"wrong_field": "value"}'
        return mock_resp

    with patch.object(client, "_get_client") as mock_get:
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = fake_create
        mock_get.return_value = mock_groq

        result = client.call(
            messages=[{"role": "user", "content": "classify"}],
            response_model=ExpectedSchema,
            safe_default=None,
        )

    assert result is None, "Should return None (safe_default) on schema validation exhaustion"


@pytest.mark.groq
def test_narrative_groq_fallback_to_heuristic(basic_speaker_result, standard_rules):
    """Narrative stage falls back to heuristic when Groq is None."""
    from pipeline.schemas import ShowType
    from pipeline.stage3_narrative import understand_narrative

    result = understand_narrative(basic_speaker_result, ShowType.NAV_THETHI, standard_rules, groq_client=None)
    assert result.result is not None, "Must produce narrative even without Groq"
    assert len(result.result.segments) == len(basic_speaker_result.transcript)
    # Heuristic warning should be present
    assert any("heuristic" in w.lower() or "groq" in w.lower() for w in result.warnings)


@pytest.mark.groq
def test_groq_budget_tracked_correctly(tmp_path):
    """Groq tracks calls_used correctly."""
    client, _ = _make_client(budget=3, tmp_dir=tmp_path)

    call_count = [0]
    def fake_create(*args, **kwargs):
        call_count[0] += 1
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '{"label": "answer"}'
        return mock_resp

    with patch.object(client, "_get_client") as mock_get:
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = fake_create
        mock_get.return_value = mock_groq

        for i in range(3):
            # Use different messages to avoid cache hits
            client.call(messages=[{"role": "user", "content": f"call {i}"}])

    assert client.calls_used == 3

    # 4th call should raise
    with pytest.raises(BudgetExceededError):
        with patch.object(client, "_get_client") as mock_get:
            client.call(messages=[{"role": "user", "content": "over budget"}])
