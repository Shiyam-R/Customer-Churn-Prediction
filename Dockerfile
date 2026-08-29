# ── Stage 1: builder — install dependencies into a local user dir ────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ── Stage 2: runtime — lean image, only what's needed to SERVE ───────────────
FROM python:3.12-slim

# libgomp1 is required at runtime by XGBoost (OpenMP) — a common Docker
# gotcha: xgboost imports fine in dev (already has it via conda/system libs)
# but fails with "libgomp.so.1: cannot open shared object file" in a bare
# slim image without this.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — don't run the serving process as root
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser/app

COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Build identity for /version — injected at build time via
# `docker build --build-arg GIT_SHA=...` (ci.yml's publish-and-deploy job
# passes github.sha automatically). Falls back to "unknown" for a manual
# local `docker build` with no arg passed.
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}
ENV ENVIRONMENT=production

# Only what the SERVING app needs — training scripts, notebooks, the raw
# dataset, and dev-only files never enter the image at all.
COPY app/ ./app/
COPY artifacts/ ./artifacts/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# No --reload in production — that's dev-only and was also the source of
# the local Windows Firewall issue earlier.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
