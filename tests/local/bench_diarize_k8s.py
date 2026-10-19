#!/usr/bin/env python3
"""Benchmark /v1/diarize with explicit network-overhead decomposition.

Each request is split into three timed phases over one HTTP connection:

  upload    time spent sending the multipart body (dominated by the audio
            bytes; == the network transfer cost of large files)
  server    time from the last body byte sent to the first response byte
            (queueing + decode + diarization; uvicorn writes the response
            only after the handler returns, so this is processing time)
  download  time to read the (small) JSON job body

Reports e2e RTFx (audio-seconds / total wall) AND engine RTFx
(audio-seconds / server time), plus upload throughput in MB/s, so a slow
transport (kubectl port-forward tunnels ~2-6 MB/s) can be told apart from
the engine's real speed. For engine-true numbers run this INSIDE the
cluster (kubectl exec, CORO_BASE=http://localhost:8080) where upload is
loopback-speed.

Usage:
  CORO_BASE=http://localhost:18080 CORO_AUDIO=/tmp/bench-60min.wav \
  CORO_MODEL=pyannote/community-1 CORO_CONCURRENCY=1 \
  python3 bench_diarize_k8s.py
"""
import http.client
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

BASE = os.environ.get("CORO_BASE", "http://localhost:18080")
AUDIO = os.environ.get("CORO_AUDIO", "/tmp/bench-60min.wav")
MODEL = os.environ.get("CORO_MODEL", "pyannote/community-1")
CONCURRENCY = int(os.environ.get("CORO_CONCURRENCY", "1"))
ROUNDS = int(os.environ.get("CORO_ROUNDS", "1"))
AUDIO_SECONDS = float(os.environ.get("CORO_AUDIO_SECONDS", "3600"))
CHUNK = 1 << 20  # 1 MiB send chunks so upload time is measurable


def build_body() -> tuple[bytes, str]:
    """Multipart body with the model id and the audio file."""
    boundary = f"----corobench{uuid.uuid4().hex}"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n{MODEL}\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="exclusive"\r\n\r\ntrue\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode()
    with open(AUDIO, "rb") as f:
        audio = f.read()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + audio + tail, boundary


BODY, BOUNDARY = build_body()
PARSED = urlparse(BASE)


def one_request(_: int) -> dict:
    """One request with upload / server / download phase timing."""
    conn = http.client.HTTPConnection(
        PARSED.hostname, PARSED.port or 80, timeout=3600
    )
    t0 = time.perf_counter()
    conn.putrequest("POST", "/v1/diarize")
    conn.putheader("Content-Type", f"multipart/form-data; boundary={BOUNDARY}")
    conn.putheader("Content-Length", str(len(BODY)))
    conn.endheaders()
    t_send0 = time.perf_counter()
    for off in range(0, len(BODY), CHUNK):
        conn.send(BODY[off : off + CHUNK])
    t_sent = time.perf_counter()
    resp = conn.getresponse()
    t_hdr = time.perf_counter()
    data = resp.read()
    t_done = time.perf_counter()
    status = resp.status
    conn.close()
    return {
        "upload": t_sent - t_send0,
        "server": t_hdr - t_sent,
        "download": t_done - t_hdr,
        "total": t_done - t0,
        "status": status,
        "segs": data.count(b'"speaker"') if status == 200 else -1,
        "body": data[:200] if status != 200 else b"",
    }


def report(results: list[dict], total_wall: float) -> None:
    ok = [r for r in results if r["status"] == 200]
    bad = [r for r in results if r["status"] != 200]
    audio_mb = os.path.getsize(AUDIO) / 2**20
    print(f"\n{'upload_s':>9} {'MB/s':>6} {'server_s':>9} {'dl_s':>6} "
          f"{'total_s':>8} {'e2e_RTFx':>9} {'engine_RTFx':>11}  segs")
    for r in results[: max(CONCURRENCY, 8)]:
        if r["status"] != 200:
            print(f"  ERROR HTTP {r['status']}: {r['body'][:120]!r}")
            continue
        mbps = audio_mb / r["upload"] if r["upload"] > 0 else 0.0
        print(f"{r['upload']:9.2f} {mbps:6.1f} {r['server']:9.2f} "
              f"{r['download']:6.2f} {r['total']:8.2f} "
              f"{AUDIO_SECONDS / r['total']:9.1f} "
              f"{AUDIO_SECONDS / r['server']:11.1f}  {r['segs']}")
    if not ok:
        return
    mean = lambda k: sum(r[k] for r in ok) / len(ok)  # noqa: E731
    total_audio = AUDIO_SECONDS * len(ok)
    print(f"\nrequests ok/err: {len(ok)}/{len(bad)}  "
          f"body={audio_mb:.0f} MiB audio={AUDIO_SECONDS / 60:.0f} min")
    print(f"mean upload {mean('upload'):.2f}s @ "
          f"{audio_mb / mean('upload'):.1f} MB/s | "
          f"mean server {mean('server'):.2f}s | mean total {mean('total'):.2f}s")
    print(f"aggregate e2e RTFx: {total_audio / total_wall:8.1f}x  "
          f"({total_audio / 60:.0f} audio-min in {total_wall:.1f}s wall)")
    print(f"per-request engine RTFx (audio/server): "
          f"{AUDIO_SECONDS / mean('server'):.1f}x mean")


def main() -> None:
    print(f"target={BASE} audio={AUDIO} "
          f"({os.path.getsize(AUDIO) / 2**20:.0f} MiB, "
          f"{AUDIO_SECONDS / 60:.0f} min) model={MODEL} "
          f"concurrency={CONCURRENCY} rounds={ROUNDS}")

    # warmup (also validates the endpoint + model id)
    warm = one_request(0)
    if warm["status"] != 200:
        print(f"WARMUP FAILED: HTTP {warm['status']}: {warm['body'][:300]!r}")
        raise SystemExit(1)
    print(f"warmup ok: total {warm['total']:.1f}s "
          f"(upload {warm['upload']:.1f}s @ "
          f"{os.path.getsize(AUDIO) / 2**20 / warm['upload']:.1f} MB/s, "
          f"server {warm['server']:.1f}s), {warm['segs']} segments")

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = list(pool.map(one_request, range(CONCURRENCY * ROUNDS)))
    total_wall = time.perf_counter() - t_start
    report(results, total_wall)


if __name__ == "__main__":
    main()
