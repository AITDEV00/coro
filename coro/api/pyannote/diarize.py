"""pyannoteAI-compatible Diarization API — POST /v1/diarize + GET /v1/jobs/{jobId}.

Implements the exact response conventions of the pyannoteAI cloud
``POST /v1/diarize`` batch diarization endpoint (docs.pyannote.ai, OpenAPI
`DiarizeRequest` / `DiarizationJob` / `DiarizationJobOutput`), so a client
written against the pyannoteAI REST shapes — and a LiteLLM provider
targeting this route — needs no bespoke contract for the job/output/error
bodies.

What is exact:
  * Response bodies — `DiarizationJob` (jobId, status, createdAt,
    updatedAt, output), `DiarizationJobOutput` (diarization,
    exclusiveDiarization, warning), `DiarizationSegment` ({speaker:
    "SPEAKER_00", start, end}, overlaps preserved), and the validation
    error shape ({message, errors: [{field, message}]}).
  * Request FIELD names and semantics — model, exclusive, numSpeakers,
    ... (the ones the engine can honor; the rest are spec-shaped 400s).

What deliberately differs (transport, not contract):
  * Audio arrives as a multipart upload, not a fetchable ``url``. The
    cloud requires a URL because it is multi-tenant; coro is a dedicated
    engine holding the audio already (its OpenAI route takes multipart
    too), and fetching arbitrary URLs from a private cluster is an SSRF
    hole, not a feature. NOTE: this means the OFFICIAL ``pyannoteai-sdk``
    client does NOT work against coro as-is — its flow is upload() →
    presigned URL → JSON body with a url field, plus a GET /v1/test
    auth probe. A provider that already has the bytes (LiteLLM after
    multipart parsing, plain requests/curl/httpx) posts them directly.
  * Jobs are SYNCHRONOUS under the hood: the POST runs diarization at
    hundreds× real time and returns the completed job object directly
    (status "succeeded", output included). A client that polls
    GET /v1/jobs/{jobId} anyway also works — the answer is idempotent.
* Fields the speakrs engine cannot honor are REJECTED with the spec's
  validation-error shape (400 ``{message, errors: [{field, message}]}``)
  rather than silently ignored — the pyannote convention of "in rare cases
  where we cannot honor this request, a warning is added" cannot apply to
  whole-field unsupported capabilities.

Supported engine mapping (speakrs community-1):
  model: "community-1" only (the engine IS community-1).
  numSpeakers/minSpeakers/maxSpeakers: not supported (threshold-based AHC
    auto-detects; no hint input exists) → 400.
  confidence/turnLevelConfidence: not supported → 400.
  exclusive: true → speakrs ``exclusive_segments`` in ``exclusiveDiarization``.
  transcription/transcriptionConfig: STT orchestration is a cloud product;
    coro serves transcription via the OpenAI route instead → 400.
"""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from coro.api.dependencies import get_pipeline
from coro.api.openai.transcriptions import _rate_limit_error
from coro.api.rate_limit import admit_request, admit_upload
from coro.audio import AudioConversionError, AudioInput
from coro.pipelines.full_memory import DiarizeResult


# MARK: Router Configuration
router = APIRouter()
logger = logging.getLogger(__name__)


# MARK: Spec models (pyannoteAI OpenAPI: DiarizationJob / DiarizationJobOutput)
class DiarizationSegment(BaseModel):
    """One speaker turn; overlaps preserved (pyannoteAI `DiarizationSegment`)."""

    speaker: str = Field(description="Speaker label, e.g. SPEAKER_00")
    start: float = Field(description="Segment start in seconds")
    end: float = Field(description="Segment end in seconds")


class DiarizationJobOutput(BaseModel):
    """pyannoteAI `DiarizationJobOutput` — the fields the engine can produce."""

    diarization: list[DiarizationSegment]
    exclusiveDiarization: list[DiarizationSegment] | None = None
    warning: str | None = None


class DiarizationJob(BaseModel):
    """pyannoteAI `DiarizationJob` — returned by GET /v1/jobs/{jobId}."""

    jobId: str
    status: str
    createdAt: str
    updatedAt: str
    output: DiarizationJobOutput | None = None


# MARK: Unsupported-field table (spec-honest 400s)
def _unsupported(*, field: str, reason: str) -> tuple[dict, str]:
    error = {"field": field, "message": reason}
    return error, f"{field} is not supported by this engine."


_SUPPORTED_MODELS = ("community-1",)
_QUEUE: dict[str, DiarizationJob] = {}
_QUEUE_MAX = 128


def _validation_response(message: str, errors: list[dict]) -> JSONResponse:
    """Render the spec's `ValidationErrorResponse` shape with status 400."""
    return JSONResponse({"message": message, "errors": errors}, status_code=400)


# MARK: POST /v1/diarize
@router.post("/v1/diarize", response_model=None)
async def diarize(
    request: Request,
    file: UploadFile = File(..., description="Audio file (any format ffmpeg reads)."),
    model: str = Form(default="community-1"),
    numSpeakers: int | None = Form(default=None),
    minSpeakers: int | None = Form(default=None),
    maxSpeakers: int | None = Form(default=None),
    exclusive: bool = Form(default=False),
    confidence: bool = Form(default=False),
    turnLevelConfidence: bool = Form(default=False),
    transcription: bool = Form(default=False),
    transcriptionConfig: str | None = Form(default=None),
    webhook: str | None = Form(default=None),
    webhookStatusOnly: bool = Form(default=False),
    pipeline=Depends(get_pipeline),
) -> JSONResponse:
    """Diarize audio following the pyannoteAI `POST /v1/diarize` conventions.

    The response is the completed `DiarizationJob` directly (see module
    docstring): status is "succeeded" and `output` is present, so both
    direct callers and the SDK's poll-once flow finish in one call.
    """
    started = time.perf_counter()
    request_id = uuid4().hex[:8]

    if (limited := admit_request(request)) is not None:
        raise _rate_limit_error(limited)

    # Reject unsupported spec fields with the spec's validation shape (see
    # module docstring: silent acceptance would break the convention the
    # route exists to serve).
    errors: list[dict] = []
    if model not in _SUPPORTED_MODELS:
        errors.append({
            "field": "model",
            "message": f"Model '{model}' is not supported by this engine. "
                       f"Supported: {', '.join(_SUPPORTED_MODELS)}.",
        })
    if numSpeakers is not None:
        errors.append({
            "field": "numSpeakers",
            "message": "numSpeakers is not supported: the engine's threshold-based "
                       "clustering detects the speaker count automatically.",
        })
    if minSpeakers is not None:
        errors.append({
            "field": "minSpeakers",
            "message": "minSpeakers is not supported: the engine's threshold-based "
                       "clustering detects the speaker count automatically.",
        })
    if maxSpeakers is not None:
        errors.append({
            "field": "maxSpeakers",
            "message": "maxSpeakers is not supported: the engine's threshold-based "
                       "clustering detects the speaker count automatically.",
        })
    if confidence:
        errors.append({
            "field": "confidence",
            "message": "confidence is not supported: the engine does not expose "
                       "per-segment confidence scores.",
        })
    if turnLevelConfidence:
        errors.append({
            "field": "turnLevelConfidence",
            "message": "turnLevelConfidence is not supported: the engine does not "
                       "expose per-turn confidence scores.",
        })
    if transcription:
        errors.append({
            "field": "transcription",
            "message": "transcription (STT orchestration) is not available on this "
                       "endpoint; use POST /v1/audio/transcriptions instead.",
        })
    if transcriptionConfig is not None:
        errors.append({
            "field": "transcriptionConfig",
            "message": "transcriptionConfig is not available on this endpoint; use "
                       "POST /v1/audio/transcriptions instead.",
        })
    if errors:
        return _validation_response("Invalid request", errors)

    audio = await AudioInput.from_upload(file)
    try:
        if not audio.size:
            return _validation_response(
                "Invalid request",
                [{"field": "file", "message": "Empty audio file."}],
            )
        if (limited := await admit_upload(request, audio)) is not None:
            raise _rate_limit_error(limited)

        try:
            result = await pipeline.diarize_full(audio, exclusive=exclusive)
        except ValueError as exc:
            return _validation_response(
                "Invalid request",
                [{"field": "model", "message": str(exc)}],
            )
        except AudioConversionError:
            return _validation_response(
                "Invalid request",
                [{"field": "file", "message": "Audio file could not be decoded."}],
            )
        except Exception:
            logger.exception("diarize[%s] pipeline failed after %.3fs",
                             request_id, time.perf_counter() - started)
            raise

        job_id = uuid4().hex
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        output = DiarizationJobOutput(
            diarization=[
                DiarizationSegment(speaker=s.speaker, start=s.start, end=s.end)
                for s in result.segments
            ],
            exclusiveDiarization=(
                [DiarizationSegment(speaker=s.speaker, start=s.start, end=s.end)
                 for s in result.exclusive_segments]
                if exclusive else None
            ),
        )
        job = DiarizationJob(
            jobId=job_id,
            status="succeeded",
            createdAt=now,
            updatedAt=now,
            output=output,
        )
        # Keep the last jobs answerable for SDK-style retrieve() calls.
        if len(_QUEUE) >= _QUEUE_MAX:
            _QUEUE.pop(next(iter(_QUEUE)))
        _QUEUE[job_id] = job

        logger.info(
            "diarize[%s] complete in %.3fs duration=%.3fs segments=%d exclusive=%s",
            request_id,
            time.perf_counter() - started,
            result.duration,
            len(result.segments),
            exclusive,
        )
        # exclude_none: the pyannoteAI spec omits absent optional fields
        # (exclusiveDiarization/warning only appear when requested/needed),
        # rather than serializing them as null.
        return JSONResponse(job.model_dump(exclude_none=True))
    finally:
        await audio.cleanup()


# MARK: GET /v1/jobs/{jobId}
@router.get("/v1/jobs/{job_id}", response_model=None)
async def get_job(job_id: str) -> JSONResponse:
    """Return a finished diarization job by id (pyannoteAI `Get Job`).

    Jobs live in memory only (this is a dedicated engine, not a multi-tenant
    queue): the POST response already carries the full output, and this route
    exists so the SDK's ``retrieve()`` poll loop terminates on its first
    check. Unknown ids return the spec's 404-ish validation shape.
    """
    job = _QUEUE.get(job_id)
    if job is None:
        return _validation_response(
            "Job not found",
            [{"field": "jobId", "message": f"Unknown job id {job_id!r}. "
                       "Jobs are kept in memory for a limited number of "
                       "recent completions; results are in the POST response."}],
        )
    return JSONResponse(job.model_dump(exclude_none=True))
