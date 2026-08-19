# SCALING.md — AI Narrative Video Director at 500 Episodes/Month

## Current Baseline
- Single-process pipeline: ~5–20 minutes per episode (depending on duration)
- Groq free-tier: ~50 calls/episode
- Bottlenecks: faster-whisper (ASR), pyannote diarization, camera frame extraction

---

## 500 Episodes/Month: Scaling Plan

### Load Estimate
- 500 episodes/month ≈ 17 episodes/day ≈ ~1 episode/hour
- Avg episode: 45–90 minutes of raw video
- Peak: might need 5–10 concurrent episodes during release windows

---

## Architecture Changes for Scale

### 1. Job Queue: Celery + Redis

```
[API / Webhook Trigger]
        │
        ▼
[Redis Queue] ─── episode_id, source_path, show_type
        │
        ├──► [Worker 1] ─── Stage 0 + 1 + 2 (heavy compute)
        ├──► [Worker 2] ─── Stage 0 + 1 + 2
        ├──► [Worker 3] ─── Stage 0 + 1 + 2
        │
        ▼
[Redis Queue: narrative_ready]
        │
        ├──► [Worker 4] ─── Stage 3 (Groq-heavy)
        └──► [Worker 5] ─── Stage 3
                │
        [Redis Queue: narrative_done]
                │
                ├──► [Worker 6] ─── Stages 4 + 4b + 5 + 6 (CPU-light)
                └──► [Worker 7] ─── Stages 4 + 4b + 5 + 6
```

**Why this split?**
- Stages 0–2 are I/O + compute-bound (transcription, video frame extraction)
- Stage 3 is Groq API-bound (rate limit throttle)
- Stages 4–6 are CPU-light pure Python (can share workers with Stage 3)

### 2. Parallelizing Independent Stages

Currently sequential within a single episode. With message passing:

```python
# Stage 1 (camera discovery) + Stage 2 (ASR) can run in PARALLEL
# Both only need the source file — they don't depend on each other

asyncio.gather(
    discover_cameras(ingest_result, ...),   # Stage 1
    transcribe_audio(source_path, ...),     # Stage 2 (whisper portion)
)
# Then: diarization + speaker role mapping (uses both outputs)
```

This cuts per-episode time by ~40% (camera discovery and transcription overlap).

### 3. Groq Request Budget Management at Scale

**Challenge**: 500 episodes × ~50 Groq calls = 25,000 calls/month.
Groq free tier limits: ~14,400 requests/day (~430,000/month), but with rate limits.

**Solution**: Redis-based global Groq rate limiter:

```python
class GlobalGroqBudget:
    def __init__(self, redis_client, daily_budget: int = 1000):
        self.redis = redis_client
        self.daily_budget = daily_budget
    
    def acquire(self, episode_id: str, n: int = 1) -> bool:
        key = f"groq:daily:{date.today().isoformat()}"
        current = self.redis.incr(key)
        self.redis.expire(key, 86400)  # 24h TTL
        return current <= self.daily_budget
```

- Per-episode budget: `budget_per_run: 50` (in editorial_rules.yaml)
- Daily global cap: configurable (e.g., 1000 calls/day = comfortable free-tier headroom)
- Episodes exceeding budget: queued for next day or use heuristic-only mode

### 4. Aggressive Caching of Intermediate JSON

All intermediate JSON is already cached by SHA-256 of the input content. At scale:

| Cache Level | Key | TTL |
|---|---|---|
| Ingest metadata | SHA-256(source_path) | 7 days |
| Camera inventory | SHA-256(source + frames) | 7 days |
| ASR transcript | SHA-256(audio_file) | 30 days |
| Diarization | SHA-256(audio_file) | 30 days |
| Narrative labels | SHA-256(transcript + show_type) | 7 days |
| Groq responses | SHA-256(messages + model) | 30 days |

**Effect**: Re-runs (e.g., rule change on same episode) cost 0 AI calls.
Same-episode reruns: only Stages 4–6 re-execute (~30 seconds total).

**Storage**: 500 episodes × ~5MB JSON/episode = 2.5GB/month. Cheap S3/equivalent.

### 5. Distributed Cache: Redis → S3

Replace local disk cache with:
- **Hot cache** (last 48h): Redis (fast, expensive) 
- **Cold cache** (older): S3/Backblaze B2 (cheap, slightly slower)

```python
class TieredCache:
    def get(self, key):
        result = self.redis.get(key)        # Try hot cache first
        if result: return result
        result = self.s3.get_object(key)    # Fall back to cold cache
        if result:
            self.redis.set(key, result, ex=3600)  # Promote to hot
        return result
```

### 6. Sampling-Based Human QA at Scale

At 500 episodes/month, manual review of every episode is impractical.

**Proposed QA sampling strategy**:
```
IF quality_score < 0.7:
    → full human review (priority queue)
ELIF rule_violation_count > 3:
    → human review (standard queue)  
ELIF random() < 0.05:
    → random sample review (5% of all episodes)
ELSE:
    → auto-approve, emit output.fcpxml
```

This ensures ~25 episodes/month get manual review (5% sample + all low-quality) while 475+ proceed automatically.

### 7. faster-whisper: GPU Workers (Optional Upgrade)

Currently: CPU-based faster-whisper (`compute_type="int8"`) — ~30 min for 60-min episode.

At scale, add optional GPU workers:
- GPU instance with CUDA: ~3 min for 60-min episode (10x speedup)
- Cost: free if using cloud spot instances (GCP/AWS spot T4 ~$0.15/hr)
- Implementation: `WhisperModel("medium", device="cuda", compute_type="float16")`

This is an optional upgrade — the pipeline runs correctly on CPU; GPU only reduces latency.

### 8. Monitoring & Observability

Already built (structlog JSON):
- Per-episode processing time per stage
- Groq calls used + cache hit rate
- Violation counts + quality scores
- Warning counts per episode

At scale, ship these to:
- **Grafana + Prometheus**: real-time dashboards
- **PagerDuty**: alert if `validation_pass_rate < 90%` or `groq_budget > 80%`
- **Weekly report**: average quality scores, top violation types, cache hit rates

### 9. Cost Summary at 500 Episodes/Month

| Component | Cost |
|---|---|
| Groq API (free tier) | $0 |
| faster-whisper (CPU) | ~$0 (local compute) |
| pyannote.audio | $0 (local) |
| Redis (cache) | ~$15/month (e.g., Redis Cloud free → $15 paid) |
| Storage (JSON cache + outputs) | ~$5/month (S3) |
| Compute (Celery workers, 3 machines) | ~$30–50/month (cloud VMs or local) |
| **Total** | **~$50/month** |

---

## Implementation Priority for Scaling

1. **Immediate** (0–500 episodes/month): Current architecture + caching works fine
2. **Phase 1** (500+ episodes/month): Add Celery + Redis queue, parallelize Stage 1 + 2
3. **Phase 2** (1000+ episodes/month): Tiered S3 cache, global Groq rate limiter, sampling QA
4. **Phase 3** (5000+ episodes/month): GPU whisper workers, horizontal scaling, dedicated monitoring
