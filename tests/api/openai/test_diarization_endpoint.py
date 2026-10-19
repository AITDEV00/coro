"""Public Diarization Endpoint behavior — /v1/audio/diarization.

Tests use a fake pipeline injected into RuntimeState so no real model is
loaded. The route must answer with the speaker timeline and never touch ASR;
errors mirror the Transcription Endpoint's OpenAI-style contract.
"""

from __future__ import annotations

import io
import struct
import wave

import pytest
from httpx import ASGITransport, AsyncClient

from coro.app import create_app
from coro.core.models import SpeakerSegment
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


class _RecordingDiarization:
    """Fake diarization adapter that records the PCM it receives."""

    def __init__(self) -> None:
        self.calls: list[bytes] = []

    async def diarize_pcm(self, pcm: bytes) -> list[SpeakerSegment]:
        self.calls.append(pcm)
        return [
            SpeakerSegment(start=0.0, end=0.05, speaker=0),
            SpeakerSegment(start=0.05, end=0.1, speaker=1),
        ]


class _FakePipeline:
    """Fake pipeline whose diarize() mirrors FullMemoryPipeline's contract."""

    def __init__(self, diarization) -> None:
        self._diarization = diarization

    async def diarize(self, audio) -> tuple[float, list[dict]]:
        if self._diarization is None:
            msg = "Diarization is not enabled on this server (CORO_BACKEND_DIARIZATION)."
            raise ValueError(msg)
        data = await audio.read_bytes()
        duration = len(data) / (16000 * 2)  # tests upload 16 kHz mono WAV bytes
        timeline = await self._diarization.diarize_pcm(data)
        segments = [
            {"start": round(s.start, 3), "end": round(s.end, 3), "speaker": s.speaker}
            for s in timeline
        ]
        return duration, segments


def _app_with_pipeline(pipeline):
    application = create_app(ServerSettings())
    runtime = RuntimeState(asr_adapter=object())  # non-None → ready=True
    runtime.pipeline = pipeline
    application.state.runtime = runtime
    return application


async def _post_diarization(app, *, data: bytes | None = None) -> tuple[int, dict]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/audio/diarization",
            # `is not None`, not truthiness: an empty body must stay empty.
            files={"file": ("clip.wav", data if data is not None else _minimal_wav_bytes(), "audio/wav")},
        )
        return resp.status_code, resp.json()


# ---------------------------------------------------------------------------
# Pipeline-level: the real FullMemoryPipeline.diarize with a fake adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_memory_pipeline_diarize_without_diarization_raises():
    """An ASR-Only Server must reject diarize() with ValueError (→ 400, not 500)."""
    from coro.audio import AudioInput
    from coro.pipelines.full_memory import FullMemoryPipeline

    pipeline = FullMemoryPipeline(asr=object(), diarization=None)
    audio = AudioInput(_minimal_wav_bytes())
    with pytest.raises(ValueError, match="not enabled"):
        await pipeline.diarize(audio)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diarization_endpoint_returns_timeline():
    diarization = _RecordingDiarization()
    status, body = await _post_diarization(_app_with_pipeline(_FakePipeline(diarization)))
    assert status == 200
    assert body["duration"] == pytest.approx(0.1, abs=0.05)
    assert body["segments"] == [
        {"start": 0.0, "end": 0.05, "speaker": 0},
        {"start": 0.05, "end": 0.1, "speaker": 1},
    ]
    # The adapter ran on decoded PCM, once.
    assert len(diarization.calls) == 1
    assert len(diarization.calls[0]) > 0


@pytest.mark.asyncio
async def test_diarization_endpoint_rejects_empty_file():
    diarization = _RecordingDiarization()
    status, body = await _post_diarization(
        _app_with_pipeline(_FakePipeline(diarization)), data=b""
    )
    assert status == 400
    assert "error" in body
    assert len(diarization.calls) == 0


@pytest.mark.asyncio
async def test_diarization_endpoint_rejects_disabled_diarization():
    status, body = await _post_diarization(_app_with_pipeline(_FakePipeline(None)))
    assert status == 400
    message = body["error"]["message"].lower()
    assert "diarization" in message
    assert "not enabled" in message
