"""speakrs native diarization backend (batch-only, pooled CUDA).

Wraps the ``speakrs`` Python bindings (native Rust/CUDA implementation of the
pyannote community-1 pipeline) as a batch Diarization Adapter. speakrs cannot
stream: clustering needs the whole recording, exactly like the pyannote
backend, so the Streaming Pipeline rejects this provider at startup.

The Diarization Model Selection names the speakrs Execution Mode rather than a
Hugging Face repository: ``speakrs/cuda-fast`` (2 s hop, throughput tier),
``speakrs/cuda`` (1 s step, accuracy tier), and ``speakrs/cpu`` (native Rust
CPU). A ``speakrs/`` prefix is optional. Model weights come from the pinned
``avencera/speakrs-models`` revision inside the wheel (public, no HF token
required), so an HF-style id would be meaningless here.

Adapter Concurrency Policy: pooled. The builder loads ``workers`` pipeline
handles (each its own CUDA stream, mirroring ``clone_shared`` semantics) and
``diarize_pcm`` dispatches round-robin. One handle runs one call at a time —
the binding serializes calls on a pipeline — so at most ``workers`` diarization
runs execute concurrently and further callers queue inside the binding.
"""

from __future__ import annotations

import logging
from itertools import cycle
from typing import Any

import numpy as np

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.backends.diarization.segments import convert_diarization_segments
from coro.core.models import SpeakerSegment

logger = logging.getLogger(__name__)

# Model-id token -> speakrs Execution Mode value. The binding accepts these
# spellings directly ("case, `-`, and `_` are ignored"), so no mapping layer
# beyond prefix stripping and normalization is needed.
_MODE_BY_TOKEN: dict[str, str] = {
    "cuda": "cuda",
    "cuda-fast": "cuda-fast",
    "cpu": "cpu",
}

_VALID_MODEL_IDS = ", ".join(f"'speakrs/{mode}'" for mode in _MODE_BY_TOKEN)


def _resolve_mode(model_diarization: str | None, device: str) -> str:
    """Resolve the Diarization Model Selection and device to a speakrs mode.

    Args:
        model_diarization: Model id such as ``speakrs/cuda-fast``; the
            ``speakrs/`` prefix is optional and the tier token is normalized.
        device: ``auto``/``cuda``/``cpu`` Diarization device selection. ``auto``
            keeps the mode named by the model id; an explicit device must not
            contradict it (Strict Startup Validation, not a silent override).

    Returns:
        The speakrs Execution Mode value (``cuda``, ``cuda-fast``, ``cpu``).

    Raises:
        ValueError: If the model id is not a speakrs mode, or the device
            contradicts it.

    """
    token = (model_diarization or "").strip().removeprefix("speakrs/").lower().replace("_", "-")
    mode = _MODE_BY_TOKEN.get(token)
    if mode is None:
        msg = (
            f"'{model_diarization}' is not a speakrs diarization mode. "
            f"Use one of {_VALID_MODEL_IDS}."
        )
        raise ValueError(msg)
    if device == "cpu" and mode != "cpu":
        msg = (
            f"Device 'cpu' contradicts diarization model '{model_diarization}'. "
            "Use CORO_DIARIZATION_DEVICE=auto, or select 'speakrs/cpu'."
        )
        raise ValueError(msg)
    if device == "cuda" and mode == "cpu":
        msg = (
            f"Device 'cuda' contradicts diarization model '{model_diarization}'. "
            "Use CORO_DIARIZATION_DEVICE=auto, or select a CUDA tier."
        )
        raise ValueError(msg)
    return mode


def _load_pipeline_pool(mode: str, workers: int) -> list[Any]:
    """Prepare assets once and load ``workers`` pipeline handles.

    ``prepare`` downloads or locates the pinned weight set (about 31 MB for
    CUDA modes) exactly once; every handle then loads from the same prepared
    assets, so a pool costs ``workers`` model loads, not ``workers`` downloads.
    In CUDA modes each handle runs on its own CUDA stream, which is what makes
    the pool equivalent to Rust-side ``clone_shared``.

    Args:
        mode: speakrs Execution Mode value (``cuda``, ``cuda-fast``, ``cpu``).
        workers: Worker Pool size; at least 1.

    Returns:
        ``workers`` loaded ``speakrs.Pipeline`` handles.

    Raises:
        RuntimeError: If the ``speakrs`` wheel is not installed, naming the fix.
        ValueError: If ``workers`` is not a positive integer.

    """
    if workers < 1:
        msg = f"diarization workers must be at least 1, got {workers}."
        raise ValueError(msg)
    try:
        import speakrs
    except ImportError as exc:
        msg = (
            "The 'speakrs' package is required for the 'speakrs' diarization "
            "backend but is not installed. speakrs is not on PyPI yet; build "
            "the wheel from avencera/speakrs (maturin build, Linux x86_64 "
            "ships CUDA) and pip install it, then retry."
        )
        raise RuntimeError(msg) from exc

    logger.info("Preparing speakrs assets for mode '%s' (%d workers).", mode, workers)
    prepared = speakrs.prepare(mode)
    return [speakrs.Pipeline.from_prepared(prepared) for _ in range(workers)]


class SpeakrsDiarizationAdapter:
    """DiarizationAdapter over a round-robin pool of speakrs pipelines.

    Each ``diarize_pcm`` converts 16-bit PCM to the float32 samples the binding
    reads in place, awaits ``diarize_samples_async`` on the next handle (GIL
    released for the whole run), and maps ``result.segments`` — speakrs
    speaker-turn segments with ``start``/``end``/``speaker`` attributes —
    through the shared backend-neutral normalizer.
    """

    def __init__(self, pipelines: list[Any], *, mode: str) -> None:
        self._pipelines = pipelines
        self._cycle = cycle(pipelines)
        self.mode = mode

    async def diarize_pcm(self, pcm: bytes) -> list[SpeakerSegment]:
        """Run batch diarization over full 16 kHz mono PCM audio."""
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        pipeline = next(self._cycle)
        result = await pipeline.diarize_samples_async(samples)
        duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        native_segments = [
            (segment.start, segment.end, segment.speaker) for segment in result.segments
        ]
        return convert_diarization_segments(native_segments, duration=duration)


def build_speakrs_diarization_adapter(
    model_diarization: str,
    *,
    device: str = "auto",
    workers: int = 4,
    hf_token: str | None = None,
) -> SpeakrsDiarizationAdapter:
    """Construct and return a SpeakrsDiarizationAdapter.

    Args:
        model_diarization: speakrs mode id, e.g. ``speakrs/cuda-fast``. See
            ``_resolve_mode`` for accepted spellings and device interaction.
        device: ``auto``/``cuda``/``cpu`` device selector; must not contradict
            the model id.
        workers: Worker Pool size. Each handle owns a CUDA stream; 4 is the
            measured throughput sweet spot for ``cuda-fast`` on an RTX 4090
            (~2450x aggregate RTFx at ~600x per request).
        hf_token: Hugging Face token; accepted for factory uniformity and
            ignored — the pinned speakrs weights repository is public.

    Returns:
        A ready-to-use SpeakrsDiarizationAdapter.

    Raises:
        ValueError: If the model id is not a speakrs mode, the device
            contradicts it, or ``workers`` is not a positive integer.
        RuntimeError: If the ``speakrs`` wheel is not installed.

    """
    mode = _resolve_mode(model_diarization, device)
    logger.debug("hf_token ignored by the speakrs backend (public weights).")
    pipelines = _load_pipeline_pool(mode, workers)
    logger.info("speakrs diarization loaded: mode '%s', %d pooled handle(s).", mode, len(pipelines))
    return SpeakrsDiarizationAdapter(pipelines, mode=mode)


__all__ = [
    "SpeakrsDiarizationAdapter",
    "build_speakrs_diarization_adapter",
]
