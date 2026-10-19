"""Diarization Endpoint router — /v1/audio/diarization.

A transcription-free variant of the OpenAI transcriptions route: accepts the
same upload mechanics (multipart file, spooled to disk), then runs ONLY the
configured Diarization Adapter and returns the speaker timeline as JSON.
The Transcription Endpoint stays the combined pipeline; this route exists
because diarization-only is a first-class use case for the speakrs backend —
the ASR windowing that dominates its request cost is skipped entirely, so a
60 s clip answers in milliseconds instead of seconds.

The route stays thin (ADR 0018): orchestration lives in the pipeline's
``diarize()`` (FullMemoryPipeline), errors keep the Transcription Endpoint's
OpenAI-style contract so a client that handles one route handles both.
"""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import JSONResponse

from coro.api.dependencies import get_pipeline
from coro.api.exceptions import (
    PROCESSING_FAILED_MESSAGE,
    UNDECODABLE_MEDIA_MESSAGE,
    TranscriptionProcessingError,
    TranscriptionValidationError,
)
from coro.api.openai.transcriptions import _rate_limit_error
from coro.api.rate_limit import admit_request, admit_upload
from coro.audio import AudioConversionError, AudioInput


# MARK: Router Configuration
router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


# MARK: Diarization Endpoint
@router.post("/audio/diarization", response_model=None)
async def create_diarization(
    request: Request,
    file: UploadFile = File(..., description="Audio file to diarize (any format ffmpeg reads)."),
    pipeline=Depends(get_pipeline),
) -> JSONResponse:
    """Accept audio and return the speaker timeline without transcribing.

    The response is coro's own minimal contract (no OpenAI API returns only
    speakers):

        {"duration": 58.0, "segments": [{"start": 0.0, "end": 1.36, "speaker": 0}, ...]}
    """
    request_id = uuid4().hex[:8]
    started = time.perf_counter()
    logger.info(
        "diarization[%s] request start filename=%s content_type=%s",
        request_id,
        file.filename,
        file.content_type,
    )

    if (limited := admit_request(request)) is not None:
        raise _rate_limit_error(limited)

    audio = await AudioInput.from_upload(file)
    try:
        if not audio.size:
            raise TranscriptionValidationError("Empty audio file.", param="file")
        if (limited := await admit_upload(request, audio)) is not None:
            raise _rate_limit_error(limited)

        try:
            duration, segments = await pipeline.diarize(audio)
        except AudioConversionError as exc:
            raise TranscriptionValidationError(UNDECODABLE_MEDIA_MESSAGE, param="file") from exc
        except ValueError as exc:
            raise TranscriptionValidationError(str(exc), param="backend_diarization") from exc
        except Exception as exc:
            logger.exception(
                "diarization[%s] pipeline failed after %.3fs",
                request_id,
                time.perf_counter() - started,
            )
            raise TranscriptionProcessingError(PROCESSING_FAILED_MESSAGE) from exc

        elapsed = time.perf_counter() - started
        logger.info(
            "diarization[%s] complete in %.3fs duration=%.3fs segments=%d",
            request_id,
            elapsed,
            duration,
            len(segments),
        )
        return JSONResponse({"duration": round(duration, 3), "segments": segments})
    finally:
        await audio.cleanup()
