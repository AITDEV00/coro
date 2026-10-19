"""speakrs diarization backend adapter using fake pipeline objects.

Tests verify that the SpeakrsDiarizationAdapter:
- Converts 16-bit PCM bytes into float32 samples scaled to [-1, 1).
- Resolves the community-1 model id, compute tier, and device to a speakrs
  Execution Mode, rejecting other ids, unsupported tiers, and cpu devices
  (Strict Startup Validation, not silent override).
- Dispatches round-robin across the Worker Pool of pipeline handles.
- Converts speakrs ``result.segments`` (start/end/speaker attributes) into
  1-indexed SpeakerSegment values clamped to the audio duration.
- Surfaces an informative error when the ``speakrs`` wheel is missing.
- Loads the pool with one ``prepare`` call and ``workers`` pipelines.

No real model inference, GPU, or network access is performed; the ``speakrs``
import is patched everywhere it would happen for real.
"""

from __future__ import annotations

import struct
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from coro.backends.diarization.speakrs import (
    SPEAKRS_MODEL_ID,
    SpeakrsDiarizationAdapter,
    _load_pipeline_pool,
    _resolve_mode,
    build_speakrs_diarization_adapter,
)
from coro.core.models import SpeakerSegment

# 1 second of silence at 16 kHz mono 16-bit.
_FAKE_PCM = struct.pack("<16000h", *([0] * 16000))
# 8 seconds, so multi-second segments fit inside the audio duration.
_LONG_PCM = struct.pack("<128000h", *([0] * 128000))


class _FakeResult:
    """Minimal stand-in for a speakrs DiarizationResult."""

    def __init__(self, segments):
        # segments: list of (start, end, speaker-label)
        self.segments = [SimpleNamespace(start=s, end=e, speaker=label) for s, e, label in segments]


class _FakePipeline:
    """Minimal stand-in for a speakrs.Pipeline with call recording."""

    def __init__(self, segments):
        self._segments = segments
        self.calls: list[object] = []

    async def diarize_samples_async(self, samples):
        self.calls.append(samples)
        return _FakeResult(self._segments)


# MARK: mode resolution
def test_resolve_mode_accepts_both_cuda_tiers():
    """The community-1 id resolves with either compute tier on a GPU device."""
    assert (
        _resolve_mode(SPEAKRS_MODEL_ID, "cuda-fast", "auto")
        == _resolve_mode(SPEAKRS_MODEL_ID, "cuda-fast", "cuda")
        == "cuda-fast"
    )
    assert _resolve_mode(SPEAKRS_MODEL_ID, "cuda", "auto") == "cuda"


def test_resolve_mode_rejects_other_model_ids():
    """The backend serves exactly the community-1 pipeline id, like pyannote."""
    with pytest.raises(ValueError, match="serves 'pyannote/speaker-diarization-community-1'"):
        _resolve_mode("speakrs/cuda-fast", "cuda-fast", "auto")
    with pytest.raises(ValueError, match="serves 'pyannote/speaker-diarization-community-1'"):
        _resolve_mode("", "cuda-fast", "auto")


def test_resolve_mode_rejects_unsupported_tier():
    """Only the two CUDA tiers are valid; cpu is gone."""
    with pytest.raises(ValueError, match="Unsupported speakrs mode"):
        _resolve_mode(SPEAKRS_MODEL_ID, "cpu", "auto")


def test_resolve_mode_rejects_cpu_device():
    """The backend is GPU-only; a cpu device fails Strict Startup Validation."""
    with pytest.raises(ValueError, match="GPU-only"):
        _resolve_mode(SPEAKRS_MODEL_ID, "cuda-fast", "cpu")


# MARK: adapter conversion
async def test_diarize_pcm_converts_segments_and_scales_samples():
    """PCM becomes /32768 float32 and result segments become SpeakerSegments."""
    pipeline = _FakePipeline([(0.5, 3.0, "SPEAKER_00"), (3.5, 6.0, "SPEAKER_01")])
    adapter = SpeakrsDiarizationAdapter([pipeline], mode="cuda-fast")

    segments = await adapter.diarize_pcm(_LONG_PCM)

    assert segments == [
        SpeakerSegment(start=0.5, end=3.0, speaker=1),
        SpeakerSegment(start=3.5, end=6.0, speaker=2),
    ]
    samples = pipeline.calls[0]
    assert samples.dtype.name == "float32"
    assert len(samples) == 128000
    assert float(samples.max()) == 0.0  # silence stays 0.0 after scaling


async def test_diarize_pcm_clamps_end_to_duration():
    """Segment end times beyond the audio duration clamp to it."""
    pipeline = _FakePipeline([(0.0, 99.0, "SPEAKER_00")])
    adapter = SpeakrsDiarizationAdapter([pipeline], mode="cuda-fast")

    segments = await adapter.diarize_pcm(_FAKE_PCM)

    assert segments == [SpeakerSegment(start=0.0, end=1.0, speaker=1)]


async def test_diarize_pcm_dispatches_round_robin():
    """Consecutive calls cycle across every pooled handle."""
    p1 = _FakePipeline([(0.0, 0.5, "SPEAKER_00")])
    p2 = _FakePipeline([(0.0, 0.5, "SPEAKER_01")])
    adapter = SpeakrsDiarizationAdapter([p1, p2], mode="cuda-fast")

    await adapter.diarize_pcm(_FAKE_PCM)
    await adapter.diarize_pcm(_FAKE_PCM)
    await adapter.diarize_pcm(_FAKE_PCM)

    assert len(p1.calls) == 2
    assert len(p2.calls) == 1


# MARK: builder
def test_load_pool_rejects_non_positive_workers():
    """The Worker Pool size must be a positive integer."""
    with pytest.raises(ValueError, match="at least 1"):
        _load_pipeline_pool("cuda-fast", 0)


def _fake_speakrs_module() -> types.ModuleType:
    """A minimal stand-in for the speakrs package with recorded calls."""
    module = types.ModuleType("speakrs")
    prepared = SimpleNamespace(mode="cuda-fast")
    module.prepare = lambda mode: prepared
    module.Pipeline = SimpleNamespace(from_prepared=lambda p: _FakePipeline([]))
    module._prepared_sentinel = prepared
    return module


def test_build_adapter_loads_pool_via_prepare():
    """One prepare call loads shared assets; workers pipelines are created."""
    fake = _fake_speakrs_module()
    with patch.dict(sys.modules, {"speakrs": fake}):
        adapter = build_speakrs_diarization_adapter(SPEAKRS_MODEL_ID, device="auto", workers=4)

    assert isinstance(adapter, SpeakrsDiarizationAdapter)
    assert adapter.mode == "cuda-fast"
    assert len(adapter._pipelines) == 4


def test_build_adapter_raises_without_wheel():
    """A missing speakrs wheel raises an error naming the build fix."""
    import builtins

    real_import = builtins.__import__

    def _no_speakrs(name, *args, **kwargs):
        if name == "speakrs":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=_no_speakrs),
        pytest.raises(RuntimeError, match="not on PyPI"),
    ):
        _load_pipeline_pool("cuda-fast", 1)
