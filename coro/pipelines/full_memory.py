"""Full-Memory Pipeline implementation."""

from __future__ import annotations

import json
from dataclasses import asdict

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, AudioInput, convert_path_to_pcm_bytes
from coro.core.response import build_transcription_response
from coro.core.protocols import ASRAdapter, DiarizationAdapter
from coro.core.models import TokenBatchEvent, TranscriptDoneEvent, TranscriptionResult
from coro.pipelines.windowing import ASRWindowing, LanguageState


# MARK: Full-Memory Pipeline
class FullMemoryPipeline:
    """Decode the full upload into memory before shared ASR Windowing."""

    def __init__(
        self,
        *,
        asr: ASRAdapter,
        diarization: DiarizationAdapter | None = None,
        windowing: ASRWindowing | None = None,
    ) -> None:
        self._asr = asr
        self._diarization = diarization
        self._windowing = windowing or ASRWindowing()

    # PCM Decoding ----------------------------------------------------------
    async def _pcm(self, audio: AudioInput) -> bytes:
        """Decode the upload to PCM, letting ffmpeg read it from disk.

        Decoding from the path rather than from ``read_bytes()`` keeps the
        encoded upload out of resident memory; the decoded PCM is still fully
        materialised, which is what makes this the Full-Memory Pipeline.
        """
        return await convert_path_to_pcm_bytes(await audio.temp_path())

    # Batch Transcription ---------------------------------------------------
    async def transcribe(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        # The pipeline consuming the audio owns its temp file, mirroring the
        # Streaming Pipeline: uploads are now spooled eagerly, so skipping this
        # would leak a file the size of the upload on every request.
        try:
            pcm = await self._pcm(audio)
            duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            result = await self._windowing.transcribe_pcm(
                pcm,
                asr=self._asr,
                language=language,
                prompt=prompt,
            )
            timeline = []
            if self._diarization is not None:
                timeline = await self._diarization.diarize_pcm(pcm)
            return build_transcription_response(
                result.tokens, timeline, duration, detected_language=result.detected_language
            )
        finally:
            await audio.cleanup()

    # Diarization-Only -------------------------------------------------------
    async def diarize(self, audio: AudioInput) -> tuple[float, list[dict]]:
        """Decode the upload and return (duration_s, speaker segments).

        The diarization-only counterpart of ``transcribe()``: same decode path
        and temp-file ownership, no ASR windowing. Serves POST /v1/audio/
        diarization — first-class for the speakrs backend, where skipping ASR
        turns a seconds-scale request into milliseconds. Raises ``ValueError``
        when no diarization adapter is configured; the route maps that to the
        OpenAI-style 400 (an ASR-Only Server must reject, not 500).
        """
        try:
            if self._diarization is None:
                msg = (
                    "Diarization is not enabled on this server "
                    "(CORO_BACKEND_DIARIZATION)."
                )
                raise ValueError(msg)
            pcm = await self._pcm(audio)
            duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            timeline = await self._diarization.diarize_pcm(pcm)
            segments = [
                {"start": round(s.start, 3), "end": round(s.end, 3), "speaker": s.speaker}
                for s in timeline
            ]
            return duration, segments
        finally:
            await audio.cleanup()

    # Streaming Transcription ----------------------------------------------
    async def stream(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ):
        try:
            pcm = await self._pcm(audio)
            duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            tokens = []
            state = LanguageState()
            async for event in self._windowing.stream_pcm(
                pcm,
                asr=self._asr,
                language=language,
                prompt=prompt,
                language_state=state,
            ):
                if isinstance(event, TokenBatchEvent):
                    tokens.extend(event.tokens)
                    continue
                yield event
            timeline = []
            if self._diarization is not None:
                timeline = await self._diarization.diarize_pcm(pcm)
            yield TranscriptDoneEvent(
                text=json.dumps(
                    asdict(
                        build_transcription_response(
                            tokens, timeline, duration, detected_language=state.resolved
                        )
                    )
                ),
            )
        finally:
            await audio.cleanup()
