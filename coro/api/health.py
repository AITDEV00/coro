"""Health check endpoint for coro."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


# MARK: Router Configuration
router = APIRouter()


# MARK: Health Endpoint
# "/health-check" is an alias kept for cluster healthcheck conventions (the
# omnivoice-server layout probes it); both names share one handler.
@router.get("/health")
@router.get("/health-check")
async def health(request: Request) -> JSONResponse:
    """Return startup selection and capability readiness."""
    runtime = request.app.state.runtime
    return JSONResponse(
        {
            "status": "ok",
            "ready": runtime.ready and runtime.warmup_ready,
            "warmup_ready": runtime.warmup_ready,
            "startup_selection": {
                "pipeline": runtime.pipeline_selector,
                "asr_provider": runtime.asr_provider,
                "asr_model": runtime.asr_model,
                "diarization_provider": runtime.diarization_provider,
                "diarization_model": runtime.diarization_model,
                **(
                    {"diarization_latency": runtime.diarization_latency}
                    if runtime.diarization_latency is not None
                    else {}
                ),
            },
            "capability_readiness": {
                "asr": runtime.asr_adapter is not None,
                "diarization": (
                    "disabled"
                    if runtime.diarization_provider == "none"
                    else runtime.diarization_adapter is not None
                ),
                "transcription": runtime.ready,
            },
        }
    )
