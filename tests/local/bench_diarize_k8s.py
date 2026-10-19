#!/usr/bin/env python3
"""Benchmark /v1/diarize (model=community-1) through a port-forward.

Usage:
  CORO_BASE=http://localhost:18080 CORO_AUDIO=/tmp/bench-60min.wav \
  CORO_CONCURRENCY=1 CORO_ROUNDS=3 python3 bench_diarize_k8s.py

Single-run mode prints per-request wall time and RTFx; concurrent mode runs
N threads x R rounds and prints aggregate RTFx (audio-seconds per wall-second).
"""
import os
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("CORO_BASE", "http://localhost:18080")
AUDIO = os.environ.get("CORO_AUDIO", "/tmp/bench-60min.wav")
CONCURRENCY = int(os.environ.get("CORO_CONCURRENCY", "1"))
ROUNDS = int(os.environ.get("CORO_ROUNDS", "1"))
AUDIO_MINUTES = float(os.environ.get("CORO_AUDIO_MINUTES", "60"))


def one_request(idx: int) -> tuple[float, float, str]:
    boundary = f"----corobench{uuid.uuid4().hex}"
    with open(AUDIO, "rb") as f:
        audio = f.read()
    lines = []
    for key, value in (
        ("model", "community-1"),
        ("exclusive", "true"),
    ):
        lines.append(
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="{key}"\r\n\r\n{value}\r\n'
        )
    lines.append(
        f"--{boundary}\r\nContent-Disposition: form-data; "
        f'name="file"; filename="audio.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    )
    tail = f"\r\n--{boundary}--\r\n"
    body = "".join(lines).encode() + audio + tail.encode()
    req = urllib.request.Request(
        f"{BASE}/v1/diarize",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            payload = resp.read()
            wall = time.perf_counter() - t0
            n = payload.count(b'"speaker"')
            return wall, n, ""
    except urllib.error.HTTPError as e:
        wall = time.perf_counter() - t0
        return wall, -1, f"HTTP {e.code}: {e.read()[:300]!r}"
    except Exception as e:  # noqa: BLE001
        wall = time.perf_counter() - t0
        return wall, -1, repr(e)


def main() -> None:
    audio_seconds = AUDIO_MINUTES * 60
    print(f"target={BASE} audio={AUDIO} ({AUDIO_MINUTES:.0f} min) "
          f"concurrency={CONCURRENCY} rounds={ROUNDS}")

    # warmup (also validates the endpoint works)
    wall, segs, err = one_request(0)
    if err:
        print(f"WARMUP FAILED in {wall:.1f}s: {err}")
        raise SystemExit(1)
    print(f"warmup ok: {wall:.1f}s, {segs} speaker-tagged segments, "
          f"{audio_seconds / wall:.1f}x RTFx")

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = list(pool.map(one_request, range(CONCURRENCY * ROUNDS)))
    total_wall = time.perf_counter() - t_start

    walls = sorted(w for w, _, _ in results)
    errors = [e for _, _, e in results if e]
    ok = [w for w, _, e in results if not e]
    total_audio = audio_seconds * len(ok)
    print(f"\nrequests: {len(results)} ({len(errors)} errors)")
    for w, n, e in results[:CONCURRENCY]:  # per-request detail for first round
        tag = f"err={e}" if e else f"segs={n}"
        print(f"  {w:7.1f}s  {audio_seconds / w:8.1f}x RTFx  {tag}")
    if ok:
        print(f"\np50 wall: {walls[len(walls) // 2]:.1f}s")
        print(f"aggregate RTFx: {total_audio / total_wall:.1f}x  "
              f"({total_audio / 60:.0f} audio-min in {total_wall:.1f}s)")
    if errors:
        print(f"first error: {errors[0]}")


if __name__ == "__main__":
    main()
