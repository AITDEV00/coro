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
# through the standard HF hub cache. The Rust hf-hub client (0.5) looks up
# refs/<revision> where revision is the PINNED COMMIT (not "main"), and only
# then falls back to the network — which the ADEO cluster blocks (no egress).
# Seeding refs/<commit> makes resolution a pure cache hit; HF_HUB_OFFLINE is
# irrelevant (that env is read by the Python client, not the Rust crate).
BAKED="${SPEAKRS_MODEL_DIR:-/home/runner/models/speakrs}"
REPO_DIR="${HF_HUB_CACHE}/models--avencera--speakrs-models"
# The pinned revision speakrs 0.6/0.7 resolves (src/models.rs).
REV="67f963b6a1d94fcd4678948237b3cc8bd69bf1d8"

if [ -d "$BAKED" ]; then
    SNAP="$REPO_DIR/snapshots/$REV"
    mkdir -p "$SNAP" "$REPO_DIR/refs"
    if [ ! -e "$REPO_DIR/refs/$REV" ]; then
        ln "$REPO_DIR/snapshots/$REV" "$REPO_DIR/refs/$REV" 2>/dev/null \
            || printf '%s' "$REV" > "$REPO_DIR/refs/$REV"
    fi
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

# ── Seed the baked canary ASR model into the writable HF cache layout ────────
# The default ASR slug (onnx-canary-split) resolves its repo with
# huggingface_hub.snapshot_download, which under HF_HUB_OFFLINE=1 needs the
# snapshot dir AND refs/main present under HF_HUB_CACHE. Same hardlink-or-copy
# seeding as speakrs; a warm cache mount makes it a no-op.
CANARY_BAKED="${CANARY_MODEL_DIR:-/home/runner/models/canary}"
CANARY_REPO_DIR="${HF_HUB_CACHE}/models--collectiveai--canary-1b-v2-onnx-split-int8"
if [ -d "$CANARY_BAKED/snapshots" ]; then
    mkdir -p "$CANARY_REPO_DIR/snapshots"
    canary_copied=0
    for snap in "$CANARY_BAKED/snapshots"/*/; do
        rev="$(basename "$snap")"
        if [ ! -d "$CANARY_REPO_DIR/snapshots/$rev" ]; then
            mkdir -p "$CANARY_REPO_DIR/snapshots/$rev"
        fi
        for f in "$snap"*; do
            name="$(basename "$f")"
            if [ ! -e "$CANARY_REPO_DIR/snapshots/$rev/$name" ]; then
                ln "$f" "$CANARY_REPO_DIR/snapshots/$rev/$name" 2>/dev/null \
                    || cp "$f" "$CANARY_REPO_DIR/snapshots/$rev/$name"
                canary_copied=$((canary_copied + 1))
            fi
        done
    done
    if [ -d "$CANARY_BAKED/refs" ] && [ ! -e "$CANARY_REPO_DIR/refs/main" ]; then
        mkdir -p "$CANARY_REPO_DIR/refs"
        ln "$CANARY_BAKED/refs/main" "$CANARY_REPO_DIR/refs/main" 2>/dev/null \
            || cp "$CANARY_BAKED/refs/main" "$CANARY_REPO_DIR/refs/main"
        canary_copied=$((canary_copied + 1))
    fi
    echo "[entrypoint] canary ASR: seeded $canary_copied file(s) into $CANARY_REPO_DIR"
else
    echo "[entrypoint] no baked canary at $CANARY_BAKED; ASR resolves from $HF_HUB_CACHE"
fi

echo "[entrypoint] coro speakrs backend: mode=${CORO_DIARIZATION_SPEAKRS_MODE:-cuda-fast} workers=${CORO_DIARIZATION_WORKERS:-4}"

# ── Exec the server as PID 1 (single foreground process) ───────────────────
exec "$@"
