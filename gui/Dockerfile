# ── Stage 1: build ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System tools needed to compile Python extensions and download Julia
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates git gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

# ── Install Julia ─────────────────────────────────────────────────────────────
ARG JULIA_VERSION=1.10.3
RUN wget -qO /tmp/julia.tar.gz \
    "https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
    && mkdir -p /opt/julia \
    && tar -xzf /tmp/julia.tar.gz -C /opt/julia --strip-components=1 \
    && ln -s /opt/julia/bin/julia /usr/local/bin/julia \
    && rm /tmp/julia.tar.gz

# Julia stores downloaded packages and precompiled .ji files here.
# Putting it outside /app means it survives COPY . . and can be a named volume.
ENV JULIA_DEPOT_PATH=/julia_depot
RUN mkdir -p /julia_depot

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only requirements first so pip layer is cached on code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App source ────────────────────────────────────────────────────────────────
COPY . .

# ── Pre-install and precompile Julia packages ─────────────────────────────────
# Running this during build bakes the compiled .ji files into the image so
# container startup is fast (~5 s) instead of requiring a 5-10 min cold-compile.
RUN python3 -c "\
import autoeis as ae; \
import numpy as np; \
print('AutoEIS + Julia packages installed and precompiled.')"

EXPOSE 8050

CMD ["gunicorn", "-c", "gunicorn_conf.py", "app:server"]
