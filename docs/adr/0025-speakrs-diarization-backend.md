# 0025: speakrs diarization backend

Date: 2026-10-19

## Status

Accepted

## Context

The pyannote Diarization Backend Provider wraps `pyannote/speaker-diarization-community-1`
in Python and runs at ~32x real time on a consumer GPU. speakrs (avencera/speakrs, v0.6.0)
is a native Rust/CUDA reimplementation of the same community-1 pipeline: segmentation,
WeSpeaker embeddings, and VBx clustering run as custom CUDA kernels (TF32 tensor-core
convolutions, FP16 trunk, FFT filterbank), with weights loaded from a pinned public
`avencera/speakrs-models` revision (~31 MB, no HF token, no ONNX Runtime). Measured on the
same class of GPU as the existing pyannote datapoint (RTX 4090): 1241x real time single
stream in the `cuda` tier (1 s step) and 2423x in the `cuda-fast` tier (2 s hop), with
matching community-1 DER (7.0% / 7.4% on VoxConverse dev). VRAM is flat at ~2.8 GB
regardless of audio length or concurrency. Python bindings exist (0.7.0 wheel, abi3,
GIL released during inference) but are not published to PyPI; the deployment builds the
wheel from source (maturin, Linux x86_64 ships CUDA).

The H200 MIG gateway plan (see the speakrs repo's `SPEAKRS_HANDOFF.md`) routes
`/v1/audio/transcriptions` through LiteLLM to per-workload coro deployments, so the
deployment must be selectable by model id.

## Decision

Add `speakrs` as a fourth Diarization Backend Provider: a batch-only adapter under
`coro/backends/diarization/speakrs.py`, dispatched by the existing Backend Adapter
Factory, following the pyannote adapter's shape (runtime import inside the builder,
shared segment normalization, `SpeakerSegment` at the boundary).

- **Model id selects the Execution Mode.** The Diarization Model Selection reuses its
  existing meaning — the value a deployment is addressed by — but names a speakrs tier:
  `speakrs/cuda-fast` (throughput, 2 s hop), `speakrs/cuda` (accuracy, 1 s step),
  `speakrs/cpu` (native Rust CPU). An explicit Diarization device selection that
  contradicts the tier fails Strict Startup Validation rather than being silently
  overridden. One LiteLLM deployment maps to one `CORO_MODEL_DIARIZATION`, so
  `/v1/audio/transcriptions` model routing picks the tier without new request-surface
  parameters.
- **A pooled Adapter Concurrency Policy.** A new `diarization_workers` setting
  (default 4) makes the builder load N pipeline handles — each its own CUDA stream,
  the binding-level equivalent of the Rust `clone_shared` — and `diarize_pcm`
  dispatches round-robin. At most N diarization runs execute concurrently; further
  callers queue inside the binding. Measured basis: 4 x `cuda-fast` is the throughput
  sweet spot (~2450x aggregate at ~600x per request) and VRAM stays flat, so 4 is the
  default rather than a tuning exercise per deployment. Single-engine providers
  (nemo, pyannote) ignore the setting; the factory kwarg defaults make their existing
  call sites unchanged.
- **Batch-only, like pyannote.** speakrs clusters over the whole recording, so the
  Streaming Pipeline rejects this provider at startup (the `reject_streaming_pyannote`
  validator covers both) and `_STREAMING_CAPABLE` stays `{"nemo"}`.
- **`hf_token` is ignored.** The pinned weights repository is public; the kwarg stays in
  the factory signature for protocol uniformity.

## Consequences

- The application factory, pipelines, and request surfaces are unchanged: they already
  consume the `DiarizationAdapter` protocol and the factory's provider dispatch.
- `diar-speakrs` becomes the third gated backend extra (wheel built from
  avencera/speakrs; nothing on PyPI yet). Tests patch the `speakrs` import per the
  backend-test convention, so the suite stays green without the wheel or a GPU.
- Per-request `diarize_pcm` calls on the pool serialize per handle; throughput scales
  with `diarization_workers`, bounded by GPU saturation (~2500x on an RTX 4090-class
  card; expect a fraction of that on the H200 `1g.18gb` MIG slice and re-tune there).
- Later extensions stay out of this slice deliberately: exposing `exclusive_segments`
  in responses, a diarization-only transcription mode, and dynamic per-request tier
  selection are separate tickets once the backend proves out under real coro load.
