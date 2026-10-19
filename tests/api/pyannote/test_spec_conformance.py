"""Spec-sync conformance: coro's pyannoteAI models vs the pinned official OpenAPI.

The SDK (``pyannoteai-sdk``) ships no importable shape classes — its client
is plain dicts — so the authoritative shape source is the published OpenAPI
spec itself. This suite pins that spec as a fixture
(``pyannoteai-openapi-v1.json``, fetched from docs.pyannote.ai/openapi.json)
and fails on ANY drift between it and coro's models:

* every ``DiarizeRequest`` property has a matching route parameter,
* every model field name/type matches the spec schema,
* required fields are required, optional fields optional,
* unknown-model rejection covers every model the spec defines.

If pyannoteAI adds a field, the coverage assertion here fails until coro
either supports it or adds an explicit unsupported-entry — the sync is
enforced, not hoped for. To refresh the pin after reviewing upstream
changes: curl -sL https://docs.pyannote.ai/openapi.json > <fixture>.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from coro.api.pyannote.diarize import (
    DiarizationJob,
    DiarizationJobOutput,
    DiarizationSegment,
    _SUPPORTED_MODELS,
    _UNSUPPORTED_FIELDS,
)

SPEC_PATH = Path(__file__).parent / "pyannoteai-openapi-v1.json"


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(SPEC_PATH.read_text())


def _schema(spec: dict, name: str) -> dict:
    schema = spec["components"]["schemas"].get(name)
    assert schema is not None, f"schema {name} missing from pinned spec"
    return schema


# ---------------------------------------------------------------------------
# DiarizeRequest coverage — every spec field is handled (support or reject)
# ---------------------------------------------------------------------------


def test_route_covers_every_diarize_request_field(spec):
    """Every `DiarizeRequest` property must be a route param, supported or 400'd.

    This is the sync guarantee: an upstream spec addition fails here until
    coro handles it explicitly (a new Form param, plus an entry in either
    the supported path or _UNSUPPORTED_FIELDS).
    """
    import inspect

    from coro.api.pyannote import diarize as module

    request_schema = _schema(spec, "DiarizeRequest")
    spec_fields = set(request_schema["properties"])

    handler = module.diarize
    params = inspect.signature(handler).parameters
    route_fields = set(params) - {"request", "pipeline"}  # fastapi plumbing
    # Transport mapping: the spec's `url` (fetchable remote reference) is
    # coro's multipart `file` upload — same field, different transport
    # (see the route module docstring).
    route_fields.add("url") if "file" in route_fields else None

    unhandled = spec_fields - route_fields
    assert not unhandled, (
        f"pyannoteAI spec defines fields the route does not handle: {sorted(unhandled)}. "
        "Add a Form parameter and either support the field or add it to "
        "_UNSUPPORTED_FIELDS (never ignore it silently)."
    )


def test_supported_plus_unsupported_covers_non_file_fields(spec):
    """Non-file spec fields split exactly into supported (model/exclusive) and _UNSUPPORTED_FIELDS."""
    request_schema = _schema(spec, "DiarizeRequest")
    spec_fields = set(request_schema["properties"])
    handled = {"url", "model", "exclusive", *_UNSUPPORTED_FIELDS}
    missing = spec_fields - handled
    assert not missing, f"Spec fields neither supported nor rejected: {sorted(missing)}"


def test_model_must_be_exactly_community_1(spec):
    """community-1 is the only supported model id (the engine IS community-1)."""
    request_schema = _schema(spec, "DiarizeRequest")
    spec_models = set(request_schema["properties"]["model"]["enum"])
    assert _SUPPORTED_MODELS == ("community-1",)
    assert "community-1" in spec_models  # sanity: still a valid upstream value
    rejected = spec_models - set(_SUPPORTED_MODELS)
    assert rejected, "upstream dropped all other models? update the rejection test"
    assert rejected == {"precision-2", "precision-3"}, (
        f"Upstream model enum changed: {sorted(spec_models)} — update the route's "
        "model handling and this pin."
    )


# ---------------------------------------------------------------------------
# Response schema conformance — exact shape sync
# ---------------------------------------------------------------------------


def test_diarization_segment_matches_spec(spec):
    schema = _schema(spec, "DiarizationSegment")
    assert set(schema["required"]) == {"speaker", "start", "end"}
    adapter = TypeAdapter(DiarizationSegment)
    seg = adapter.validate_python({"speaker": "SPEAKER_00", "start": 0.5, "end": 3.2})
    assert seg.speaker == "SPEAKER_00" and seg.start == 0.5 and seg.end == 3.2
    # Spec field names round-trip exactly (no renames).
    dumped = adapter.dump_python(seg)
    assert set(dumped) == {"speaker", "start", "end"}


def test_diarization_job_output_matches_spec(spec):
    schema = _schema(spec, "DiarizationJobOutput")
    assert set(schema["required"]) == {"diarization"}
    optional = set(schema["properties"]) - set(schema["required"])
    assert optional == {"confidence", "exclusiveDiarization", "speechProbability",
                       "crosstalkProbability", "speakerProbability",
                       "wordLevelTranscription", "turnLevelTranscription",
                       "error", "warning"}, (
        f"Upstream DiarizationJobOutput properties changed: {sorted(schema['properties'])}"
    )
    # coro implements the required field + exclusive + warning; the rest are
    # cloud-only (rejected at request time), which the coverage test above
    # guards from the request side.
    adapter = TypeAdapter(DiarizationJobOutput)
    out = adapter.validate_python({"diarization": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]})
    assert out.exclusiveDiarization is None and out.warning is None
    dumped = adapter.dump_python(out, exclude_none=True)
    assert set(dumped) == {"diarization"}  # absent optionals omitted, per spec


def test_diarization_job_matches_spec(spec):
    schema = _schema(spec, "DiarizationJob")
    # NOTE: the spec declares no `required` array for DiarizationJob — all
    # five properties are documented as present in practice; coro's model
    # makes exactly the documented five fields, output optional (pending
    # jobs carry no output).
    assert set(schema["properties"]) == {"jobId", "status", "createdAt", "updatedAt", "output"}
    assert schema.get("required") is None
    adapter = TypeAdapter(DiarizationJob)
    job = adapter.validate_python({
        "jobId": "abc", "status": "succeeded",
        "createdAt": "2026-10-19T12:00:00Z", "updatedAt": "2026-10-19T12:00:00Z",
        "output": {"diarization": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]},
    })
    assert job.status == "succeeded" and job.output is not None
    # coro's model must expose exactly the spec's five fields.
    assert set(DiarizationJob.model_fields) == {"jobId", "status", "createdAt", "updatedAt", "output"}


def test_job_status_enum_matches_spec(spec):
    schema = _schema(spec, "JobStatus")
    assert set(schema["enum"]) == {"pending", "created", "succeeded", "canceled", "failed", "running"}


def test_validation_error_shape_matches_spec(spec):
    schema = _schema(spec, "ValidationError")
    assert set(schema["required"]) == {"field", "message"}
    err_schema = _schema(spec, "ValidationErrorResponse")
    assert set(err_schema["required"]) == {"message", "errors"}
