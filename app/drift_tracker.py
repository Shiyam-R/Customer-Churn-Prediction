"""
app/drift_tracker.py
─────────────────────────────────────────────────────────────────────────────
Buffers recent live /predict feature vectors and computes PSI (Population
Stability Index) against the training-time baseline distribution, per
feature.

SCOPE CAVEAT: this buffer is a plain in-process deque. Under a
multi-worker deployment (uvicorn/gunicorn with --workers > 1), each
worker has its OWN separate buffer — /drift only reflects whichever
worker happens to answer that specific request, not a combined view
across all workers. This deployment currently runs a single worker (see
Dockerfile CMD — no --workers flag), so the caveat is dormant, but would
need a shared store (e.g. Redis) before scaling to multiple workers.
"""

import math
from collections import deque
from typing import Any

import pandas as pd

from app.config import DRIFT_BUFFER_SIZE, DRIFT_MIN_SAMPLES

_live_buffer: deque = deque(maxlen=DRIFT_BUFFER_SIZE)


def record_request(feature_row: pd.Series) -> None:
    """Called once per live /predict request with the exact feature
    vector the model saw (post prepare_inference_features)."""
    _live_buffer.append(feature_row.to_dict())


def _psi_for_column(
    live_values: pd.Series, bin_edges: list, baseline_props: list, epsilon: float = 1e-4
) -> float:
    edges = list(bin_edges)
    edges[0] = -math.inf
    edges[-1] = math.inf
    binned = pd.cut(live_values, bins=edges, include_lowest=True)
    live_props = binned.value_counts(normalize=True, sort=False).reindex(
        binned.cat.categories, fill_value=0.0
    ).to_numpy()

    psi = 0.0
    for live_p, base_p in zip(live_props, baseline_props, strict=False):
        live_p = max(live_p, epsilon)
        base_p = max(base_p, epsilon)
        psi += (live_p - base_p) * math.log(live_p / base_p)
    return psi


def _severity(psi: float) -> str:
    if psi < 0.1:
        return "none"
    if psi < 0.2:
        return "moderate"
    return "significant"


def compute_drift_report(baseline_stats: dict[str, Any]) -> dict[str, Any]:
    n = len(_live_buffer)
    if n < DRIFT_MIN_SAMPLES:
        return {
            "status": "insufficient_data",
            "live_sample_size": n,
            "min_samples_required": DRIFT_MIN_SAMPLES,
            "features": [],
        }

    live_df = pd.DataFrame(list(_live_buffer))
    results = []
    for col, stats in baseline_stats.items():
        if col not in live_df.columns:
            continue
        psi = _psi_for_column(live_df[col], stats["bin_edges"], stats["baseline_proportions"])
        results.append({"feature": col, "psi": round(psi, 4), "severity": _severity(psi)})

    results.sort(key=lambda r: -r["psi"])
    return {
        "status": "ok",
        "live_sample_size": n,
        "min_samples_required": DRIFT_MIN_SAMPLES,
        "features": results,
    }
