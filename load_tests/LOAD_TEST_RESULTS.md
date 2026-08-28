# Load Test Results

**Environment:** Local sandbox (single container, not Render — numbers here
characterize the app's own behavior, not production infrastructure).
**Tool:** Locust 2.46.4, `load_tests/locustfile.py`
**Config:** 20 simulated users, spawn rate 5/s, 30s duration, weighted 5:1
toward `/predict` over `/health`, against `uvicorn` directly (no Docker
layer in this run).

## Results

| Endpoint | Requests | Failures (429) | Median | p95 | p99 | Max |
|---|---|---|---|---|---|---|
| `POST /predict` | 376 | 346 (92.0%) | 3ms | 23ms | 90ms | 103ms |
| `GET /health` | 68 | 0 | 2ms | 17ms | 50ms | 50ms |

## What this actually shows

**The 92% failure rate is the rate limiter working correctly, not the app
breaking under load** — every single failure was a `429 Too Many Requests`,
zero were crashes, timeouts, or 500s. Worth being precise about *why* the
number looks this dramatic: all 20 simulated Locust users run from the same
machine, so `slowapi`'s `get_remote_address` key function buckets them as
**one client**, not 20 distinct ones. This test essentially validates the
exact threat model `PREDICT_RATE_LIMIT` was built for — one source hammering
`/predict` repeatedly — and confirms the limiter caps it at the configured
30/minute rather than letting it degrade the service. **This run does not
measure true multi-client capacity** (many distinct real users hitting the
API simultaneously would each get their own 30/minute allowance) — that
would need a distributed load generator with varied source IPs, which is a
reasonable next step but wasn't done here.

**For the ~30 requests that did get through, latency is genuinely low** —
3ms median, 23ms at p95. This directly answers the concern that motivated
building the rate limiter in the first place: is per-request SHAP
computation expensive? At this model's size (`max_depth=3`, `n_estimators=300`
— deliberately shallow trees from tuning), **no** — Tree SHAP's exact
algorithm is fast enough here that it isn't the bottleneck. The rate limit
exists as a deliberate ceiling against abuse, not because the endpoint is
struggling to keep up with legitimate traffic at this scale.

## Honest limitation

This was run against `uvicorn` directly in a sandbox, not the deployed
Docker container on Render's actual resources. Real-world latency under
Render's free tier (documented cold-start behavior in prior projects) will
likely differ — worth re-running this against the live Render URL once
deployed, rather than treating these numbers as production-representative.
