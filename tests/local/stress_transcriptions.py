#!/usr/bin/env python3
"""Stress test for coro /v1/audio/transcriptions (speakrs diarization backend).

Runs against http://localhost:8080. Phases:
  1. health        — /health-check gate
  2. single        — 1 diarized_json request (warm path, baseline latency)
  3. concurrency   — N parallel clients x R rounds, 60s clips, diarized_json
                     -> throughput (RTFx), p50/p95/p99 latency, error rate
  4. mixed-length  — a few longer clips to check stability under bigger jobs

Stdlib only (urllib + manual multipart), no external deps.
"""
import json
import os
import ssl
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("CORO_BASE", "http://localhost:8080")
AUDIO = os.environ.get("CORO_AUDIO", "/tmp/smoke-60s.wav")
CONCURRENCY = int(os.environ.get("CORO_CONCURRENCY", "8"))
ROUNDS = int(os.environ.get("CORO_ROUNDS", "5"))
FMT = os.environ.get("CORO_FORMAT", "diarized_json")

results = []          # (ok, elapsed, audio_dur, segments)
results_lock = threading.Lock()


def post_audio(path: str, fmt: str = FMT, timeout: float = 600.0) -> dict:
    """POST one file to /v1/audio/transcriptions, return json + timing."""
    boundary = "----stressBoundary7d9f2a"
    with open(path, "rb") as f:
        payload = f.read()
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + payload + (
        f"\r\n--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"response_format\"\r\n\r\n{fmt}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            elapsed = time.perf_counter() - started
            with results_lock:
                results.append((True, elapsed, data.get("duration") or 0.0,
                                len(data.get("segments") or [])))
            return data
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - started
        with results_lock:
            results.append((False, elapsed, 0.0, 0))
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:200]!r}") from e


def audio_duration(path: str) -> float:
    import wave
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def main():
    print(f"target={BASE} audio={AUDIO} concurrency={CONCURRENCY} rounds={ROUNDS} fmt={FMT}")

    # Phase 1: health gate
    with urllib.request.urlopen(f"{BASE}/health-check", timeout=10) as r:
        health = json.loads(r.read())
    assert health.get("ready") is True, f"not ready: {health}"
    print(f"[health] ok  diarization={health['capability_readiness']['diarization']}")

    dur = audio_duration(AUDIO)
    print(f"[audio] {AUDIO} duration={dur:.1f}s")

    # Phase 2: single warm request
    post_audio(AUDIO)
    ok, elapsed, adur, segs = results[-1]
    print(f"[single] {elapsed:.2f}s wall, {adur:.0f}s audio -> {adur/elapsed:.0f}x RTFx, {segs} segments")

    # Phase 3: concurrency
    print(f"[concurrency] {CONCURRENCY} clients x {ROUNDS} rounds starting...")
    t0 = time.perf_counter()
    errors = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(post_audio, AUDIO) for _ in range(CONCURRENCY * ROUNDS)]
        for fut in futures:
            try:
                fut.result()
            except Exception as e:
                errors += 1
                print(f"  error: {e}", file=sys.stderr)
    wall = time.perf_counter() - t0

    batch = results[1:]  # exclude the warm single
    oks = [r for r in batch if r[0]]
    lats = sorted(r[1] for r in oks)
    total_audio = sum(r[2] for r in oks)
    print(f"[concurrency] done in {wall:.1f}s: {len(oks)} ok, {errors} errors")
    if lats:
        agg_rtfx = total_audio / wall
        per_stream = sum(lats) / len(lats)
        print(f"[concurrency] agg throughput {agg_rtfx:.0f}x RTFx "
              f"({total_audio:.0f}s audio in {wall:.1f}s wall)")
        print(f"[concurrency] per-request: mean {per_stream:.2f}s "
              f"p50 {pct(lats,50):.2f}s p95 {pct(lats,95):.2f}s p99 {pct(lats,99):.2f}s max {lats[-1]:.2f}s")
        per_stream_rtfx = dur / per_stream
        print(f"[concurrency] per-stream RTFx mean {dur/per_stream:.0f}x (single-stream was {adur/elapsed:.0f}x)")

    # Phase 4: long-clip stability (sequential, 1 x ~5 min if available)
    long_candidates = [
        os.path.expanduser("~/DGE/speakrs/test-audio/ES2004a.Mix-Headset.wav"),
    ]
    long_path = next((p for p in long_candidates if os.path.exists(p)), None)
    if long_path:
        ldur = audio_duration(long_path)
        print(f"[long] {os.path.basename(long_path)} duration={ldur:.0f}s")
        try:
            post_audio(long_path, timeout=900)
            ok, elapsed, adur, segs = results[-1]
            print(f"[long] {elapsed:.1f}s wall, {adur:.0f}s audio -> {adur/elapsed:.0f}x RTFx, {segs} segments")
        except Exception as e:
            print(f"[long] FAILED: {e}", file=sys.stderr)

    print("[summary] stress test complete")


if __name__ == "__main__":
    main()
