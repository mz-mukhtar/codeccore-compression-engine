# ─────────────────────────────────────────────────────────────────────────────
# CodecCore: Multi-Codec Compression Laboratory
# Phase 6 — Lossy Image Pipeline, Dynamic LZ78, Audio Waveform Coding
# Base: python:3.11-slim (Debian Bookworm-slim, pre-built wheels available)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# --- Environment Flags -------------------------------------------------------
# Force stdout/stderr to be unbuffered so Docker logs flush immediately.
ENV PYTHONUNBUFFERED=1

# Disable .pyc bytecode generation inside the container (keeps image clean).
ENV PYTHONDONTWRITEBYTECODE=1

# Streamlit telemetry opt-out — avoids network calls on cold start.
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ─────────────────────────────────────────────────────────────────────────────
# OS-level build dependencies
# python:3.11-slim ships without gcc; numpy and pillow ship as pre-built
# manylinux wheels so no compiler is needed.  We only install a minimal set
# of runtime libs that some wheels link against at import time.
# ─────────────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Working Directory -------------------------------------------------------
WORKDIR /app

# --- Dependency Layer --------------------------------------------------------
# Copy requirements first so Docker's layer cache is reused when only app.py
# changes, avoiding a full pip reinstall on every rebuild.
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Application Source ------------------------------------------------------
# Securely copy all Python source modules so they are available inside the
# container. This ensures new engines (like audio_engine.py) are always included.
COPY *.py ./

# --- Network -----------------------------------------------------------------
# Streamlit default port.
EXPOSE 8501

# --- Entrypoint --------------------------------------------------------------
# --server.address=0.0.0.0  → bind to all interfaces so the host can reach it.
# --server.port=8501         → must match the EXPOSE directive above.
# --server.headless=true     → suppresses the "open browser" prompt inside
#                              the container where there is no display.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
