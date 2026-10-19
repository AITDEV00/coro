#!/usr/bin/env bash
# =============================================================================
# ADEO cluster entrypoint for the coro speakrs image.
#
# Follows the omnivoice-server internal entrypoint pattern (REQ-4 there):
# every runtime write is forced under /tmp so the container runs with a
# read-only root filesystem; then the baked speakrs weights are seeded into
# the writable HF cache layout so speakrs finds them offline; then exec coro
# as PID 1.
# =============================================================================
set -euo pipefail

# ── Force all runtime writes into /tmp (read-only-root friendly) ────────────
export HOME="${HOME:-/tmp/runner}"
export TMPDIR="${TMPDIR:-/tmp/runner/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/runner/.cache}"
export HF_HOME="${HF_HOME:-/tmp/runner/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/runner/hf/hub}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$HF_HUB_CACHE"

# ── Seed the baked speakrs weights into the writable HF cache layout ────────
# speakrs' ModelManager resolves the pinned avencera/speakrs-models revision
# through the standard HF hub cache. With HF_HUB_OFFLINE=1 it needs the
# snapshot present under HF_HUB_CACHE; seeding the 9 files (31 MB) into that
# layout at start costs milliseconds from the image layer and avoids any
# first-start download. If the cache volume already has them (a warm
# coro-hf-cache mount), the seed is a no-op by revision.
BAKED="${SPEAKRS_MODEL_DIR:-/home/runner/models/speakrs}"
REPO_DIR="${HF_HUB_CACHE}/models--avencera--speakrs-models"
# The pinned revision speakrs 0.6/0.7 resolves (src/models.rs).
REV="67f963b6a1d94fcd4678948237b3cc8bd69bf1d8"

if [ -d "$BAKED" ]; then
    SNAP="$REPO_DIR/snapshots/$REV"
    mkdir -p "$SNAP"
    copied=0
    for f in "$BAKED"/*; do
        name="$(basename "$f")"
        if [ ! -e "$SNAP/$name" ]; then
            ln "$f" "$SNAP/$name" 2>/dev/null || cp "$f" "$SNAP/$name"
            copied=$((copied + 1))
        fi
    done
    echo "[entrypoint] speakrs weights: seeded $copied file(s) into $SNAP"
else
    echo "[entrypoint] no baked weights at $BAKED; speakrs will resolve them from $HF_HUB_CACHE"
fi

echo "[entrypoint] coro speakrs backend: mode=${CORO_DIARIZATION_SPEAKRS_MODE:-cuda-fast} workers=${CORO_DIARIZATION_WORKERS:-4}"

# ── Exec the server as PID 1 (single foreground process) ───────────────────
exec "$@"
