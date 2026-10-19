"""pyannoteAI-compatible /v1/diarize API — spec conformance tests.

The route must answer with the exact pyannoteAI `DiarizationJob` shape
(SDK: pyannoteai-sdk 0.4.0; OpenAPI: DiarizationJob / DiarizationJobOutput /
DiarizationSegment / ValidationErrorResponse) and reject unsupported fields
with the spec's validation-error body instead of silently ignoring them.
"""

from __future__ import annotations

import io
import struct
import wave

import pytest
from httpx import ASGITransport, AsyncClient

from coro.app import create_app
from coro.pipelines.full_memory import DiarizeResult
from coro.runtime import RuntimeState
from coro.settings import ServerSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_wav_bytes() -> bytes:
    """Produce a tiny valid WAV file (100 ms silence, 16 kHz mono 16-bit)."""
    buf = io.BytesIO()
    n_frames = 1600  # 100 ms at 16 kHz
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * n_frames, *([0] * n_frames)))
    return buf.getvalue()


class _Segment:
    """Mimics a speakrs Segment (start/end + native string speaker label)."""

    def __init__(self, start: float, end: float, speaker: str) -> None:
        self.start = start
        self.end = end
        self.speaker = speaker


class _FakeDiarization:
    """Fake adapter exposing the native diarize_pcm_full contract."""

    def __init__(self) -> None:
        self.calls: list[bytes] = []

    async def diarize_pcm_full(self, pcm: bytes) -> tuple[list, list]:
        self.calls.append(pcm)
        return (
            [
                _Segment(0.0, 1.5, "SPEAKER_00"),
                _Segment(1.4, 3.0, "SPEAKER_01"),  # overlap with the first
            ],
            [
                _Segment(0.0, 1.45, "SPEAKER_00"),  # exclusive view resolves it
                _Segment(1.45, 3.0, "SPEAKER_01"),
            ],
        )


class _FakePipeline:
    """Fake pipeline whose diarize_full mirrors FullMemoryPipeline's."""

    def __init__(self, diarization) -> None:
        self._diarization = diarization

    async def diarize_full(self, audio, *, exclusive: bool = False) -> DiarizeResult:
        if self._diarization is None:
            msg = "Diarization is not enabled on this server (CORO_BACKEND_DIARIZATION)."
            raise ValueError(msg)
        data = await audio.read_bytes()
        duration = len(data) / (16000 * 2)
        segments, exclusive_segments = await self._diarization.diarize_pcm_full(data)
        return DiarizeResult(
            duration=duration,
            segments=segments,
            exclusive_segments=exclusive_segments if exclusive else None,
        )


def _app_with_pipeline(pipeline):
    application = create_app(ServerSettings())
    runtime = RuntimeState(asr_adapter=object())  # non-None → ready=True
    runtime.pipeline = pipeline
    application.state.runtime = runtime
    return application


async def _post_diarize(app, *, data: bytes | None = None, extra: dict | None = None) -> tuple[int, dict]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("clip.wav", data if data is not None else _minimal_wav_bytes(), "audio/wav")}
        form = {"model": "community-1", **(extra or {})}
        resp = await client.post("/v1/diarize", files=files, data=form)
        return resp.status_code, resp.json()


# ---------------------------------------------------------------------------
# POST /v1/diarize — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diarize_returns_pyannote_job_shape():
    status, body = await _post_diarize(_app_with_pipeline(_FakePipeline(_FakeDiarization())))
    assert status == 200
    # Exact pyannoteAI DiarizationJob fields.
    assert body["status"] == "succeeded"
    assert set(body) >= {"jobId", "status", "createdAt", "updatedAt", "output"}
    diarization = body["output"]["diarization"]
    assert diarization == [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.5},
        {"speaker": "SPEAKER_01", "start": 1.4, "end": 3.0},
    ]
    # Overlaps preserved by default; exclusive view absent unless requested.
    assert "exclusiveDiarization" not in body["output"]


@pytest.mark.asyncio
async def test_diarize_exclusive_returns_exclusive_diarization():
    status, body = await _post_diarize(
        _app_with_pipeline(_FakePipeline(_FakeDiarization())),
        extra={"exclusive": "true"},
    )
    assert status == 200
    assert body["output"]["exclusiveDiarization"] == [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.45},
        {"speaker": "SPEAKER_01", "start": 1.45, "end": 3.0},
    ]


# ---------------------------------------------------------------------------
# Unsupported spec fields — spec-shaped 400s
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("numSpeakers", "2"),
        ("minSpeakers", "1"),
        ("maxSpeakers", "4"),
        ("confidence", "true"),
        ("turnLevelConfidence", "true"),
        ("transcription", "true"),
    ],
)
async def test_diarize_rejects_unsupported_fields(field, value):
    status, body = await _post_diarize(
        _app_with_pipeline(_FakePipeline(_FakeDiarization())),
        extra={field: value},
    )
    assert status == 400
    assert body["message"] == "Invalid request"
    fields = {e["field"] for e in body["errors"]}
    assert field in fields


@pytest.mark.asyncio
async def test_diarize_rejects_unsupported_model():
    status, body = await _post_diarize(
        _app_with_pipeline(_FakePipeline(_FakeDiarization())),
        extra={"model": "precision-2"},
    )
    assert status == 400
    assert any(e["field"] == "model" for e in body["errors"])


@pytest.mark.asyncio
async def test_diarize_rejects_empty_file():
    status, body = await _post_diarize(
        _app_with_pipeline(_FakePipeline(_FakeDiarization())), data=b""
    )
    assert status == 400
    assert any(e["field"] == "file" for e in body["errors"])


# ---------------------------------------------------------------------------
# GET /v1/jobs/{jobId}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_job_returns_posted_job():
    _, posted = await _post_diarize(_app_with_pipeline(_FakePipeline(_FakeDiarization())))
    job_id = posted["jobId"]
    transport = ASGITransport(app=_app_with_pipeline(_FakePipeline(_FakeDiarization())))
    # NOTE: the in-memory job store is module-global, so a fresh app still
    # sees the job — that is exactly the retrieve() flow the SDK uses.
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/v1/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["jobId"] == job_id
    assert resp.json()["status"] == "succeeded"


@pytest.mark.asyncio
async def test_get_job_unknown_id_is_validation_shaped():
    transport = ASGITransport(app=_app_with_pipeline(_FakePipeline(_FakeDiarization())))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/jobs/nonexistent")
    assert resp.status_code == 400
    body = resp.json()
    assert "message" in body and "errors" in body
