"""speakrs native diarization backend (batch-only, pooled CUDA).

Wraps the ``speakrs`` Python bindings (native Rust/CUDA implementation of the
pyannote community-1 pipeline) as a batch Diarization Adapter. speakrs cannot
stream: clustering needs the whole recording, exactly like the pyannote
backend, so the Streaming Pipeline rejects this provider at startup.

The Diarization Model Selection keeps its pipeline identity — ``pyannote/
speaker-diarization-community-1``, the same id the pyannote backend uses —
so both engines serve the same model name and a gateway can route or
fail over between them interchangeably. speakrs weights come from the pinned
public ``avencera/speakrs-models`` revision inside the wheel (no HF token
required), which is why the id names the pipeline rather than the weights
repository. The compute tier is a separate setting
(``CORO_DIARIZATION_SPEAKRS_MODE``): ``cuda-fast`` (2 s hop, throughput)
or ``cuda`` (1 s step, accuracy). The backend is GPU-only.

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

# The model id this backend serves — the same pipeline identity as the
# pyannote backend, so both engines are addressable under one model name.
SPEAKRS_MODEL_ID = "pyannote/speaker-diarization-community-1"

# speakrs Execution Mode values accepted by the ``speakrs_mode`` setting.
SUPPORTED_MODES = ("cuda-fast", "cuda")


def _resolve_mode(model_diarization: str, speakrs_mode: str, device: str) -> str:
    """Resolve the model id, compute tier, and device to a speakrs mode.

    Args:
        model_diarization: Diarization Model Selection; must be the community-1
            pipeline id (the same id the pyannote backend serves).
        speakrs_mode: ``cuda-fast`` (2 s hop, throughput) or ``cuda`` (1 s
            step, accuracy).
        device: ``auto``/``cuda``/``cpu`` Diarization device selection. This
            backend is GPU-only: ``auto`` and ``cuda`` are accepted, ``cpu``
            fails Strict Startup Validation rather than silently running on
            CPU.

    Returns:
        The speakrs Execution Mode value (``cuda`` or ``cuda-fast``).

    Raises:
        ValueError: If the model id is not the community-1 pipeline, the
            tier is unsupported, or the device is ``cpu``.

    """
    if (model_diarization or "").strip() != SPEAKRS_MODEL_ID:
        msg = (
            f"The speakrs backend serves '{SPEAKRS_MODEL_ID}' but got "
            f"'{model_diarization}'. Both diarization engines use the same "
            "model id; set CORO_MODEL_DIARIZATION accordingly (the default "
            "when CORO_BACKEND_DIARIZATION=speakrs)."
        )
        raise ValueError(msg)
    if speakrs_mode not in SUPPORTED_MODES:
        valid = " or ".join(f"'{mode}'" for mode in SUPPORTED_MODES)
        msg = f"Unsupported speakrs mode '{speakrs_mode}'. Use {valid}."
        raise ValueError(msg)
    if device == "cpu":
        msg = (
            "The speakrs backend is GPU-only: CORO_DIARIZATION_DEVICE=cpu is "
            "not supported. Use 'auto' or 'cuda', or select the pyannote "
            "backend for CPU inference."
        )
        raise ValueError(msg)
    return speakrs_mode


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
    # Prefer the baked model directory: `prepare(mode, model_dir=...)` skips
    # hub resolution entirely (no network, no HF cache layout, no refs
    # bookkeeping), which is what makes it immune to offline clusters. The
    # path is baked into the ADEO image via SPEAKRS_MODEL_DIR; when it is
    # absent (dev machines, slim images) fall back to the normal cache-or-
    # download flow, which works wherever egress exists.
    import os

    model_dir = os.environ.get("SPEAKRS_MODEL_DIR", "/home/runner/models/speakrs")
    if os.path.isdir(model_dir):
        logger.info("Loading speakrs models from baked dir '%s'.", model_dir)
        prepared = speakrs.prepare(mode, model_dir=model_dir)
    else:
        logger.info("No baked model dir at '%s'; resolving via HF cache.", model_dir)
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

    async def diarize_pcm_full(self, pcm: bytes) -> tuple[list, list]:
        """Run diarization and return native, string-labelled segments.

        The pyannoteAI-contract counterpart of ``diarize_pcm``: keeps the
        backend's native ``SPEAKER_XX`` string labels (instead of mapping to
        the internal 1-indexed ints) and also returns the exclusive view
        (at most one speaker at a time; overlaps go to the highest-activation
        speaker) for the spec's ``exclusiveDiarization`` field. Serves
        FullMemoryPipeline.diarize_full → POST /v1/diarize.
        """
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        pipeline = next(self._cycle)
        result = await pipeline.diarize_samples_async(samples)
        return result.segments, result.exclusive_segments


def build_speakrs_diarization_adapter(
    model_diarization: str,
    *,
    speakrs_mode: str = "cuda-fast",
    device: str = "auto",
    workers: int = 4,
    hf_token: str | None = None,
) -> SpeakrsDiarizationAdapter:
    """Construct and return a SpeakrsDiarizationAdapter.

    Args:
        model_diarization: must be ``pyannote/speaker-diarization-community-1``
            (the same id the pyannote backend serves, so a gateway can route
            between the engines under one model name).
        speakrs_mode: ``cuda-fast`` (2 s hop, throughput; default) or ``cuda``
            (1 s step, accuracy).
        device: ``auto``/``cuda``/``cpu`` device selector; this backend is
            GPU-only, so ``cpu`` is rejected.
        workers: Worker Pool size. Each handle owns a CUDA stream; 4 is the
            measured throughput sweet spot for ``cuda-fast`` on an RTX 4090
            (~2450x aggregate RTFx at ~600x per request).
        hf_token: Hugging Face token; accepted for factory uniformity and
            ignored — the pinned speakrs weights repository is public.

    Returns:
        A ready-to-use SpeakrsDiarizationAdapter.

    Raises:
        ValueError: If the model id is not the community-1 pipeline, the tier
            is unsupported, the device is ``cpu``, or ``workers`` is not a
            positive integer.
        RuntimeError: If the ``speakrs`` wheel is not installed.

    """
    mode = _resolve_mode(model_diarization, speakrs_mode, device)
    logger.debug("hf_token ignored by the speakrs backend (public weights).")
    pipelines = _load_pipeline_pool(mode, workers)
    logger.info("speakrs diarization loaded: mode '%s', %d pooled handle(s).", mode, len(pipelines))
    return SpeakrsDiarizationAdapter(pipelines, mode=mode)


__all__ = [
    "SpeakrsDiarizationAdapter",
    "build_speakrs_diarization_adapter",
]
