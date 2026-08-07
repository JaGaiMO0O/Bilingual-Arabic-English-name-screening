"""FastAPI service.

The index is loaded once at startup, not per request — embedding model load is
seconds and the FAISS index is memory-resident, so a per-request load turns a
millisecond lookup into an unusable endpoint.

Endpoints:
    GET  /health        liveness plus whether an index is actually loaded
    POST /screen        screen one name
    POST /screen/batch  screen many in one embedding pass
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScreenRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Name to screen, Arabic or Latin script")
    top_k: int | None = Field(None, ge=1, le=100)
    threshold: float | None = Field(None, ge=-1.0, le=1.0)


class BatchScreenRequest(BaseModel):
    names: list[str] = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=100)
    threshold: float | None = Field(None, ge=-1.0, le=1.0)


class CandidateResponse(BaseModel):
    record_id: str
    name: str
    score: float
    matched_form: str
    matched_via_alias: bool
    is_match: bool


class ScreenResponse(BaseModel):
    query: str
    normalized_query: str
    detected_script: str
    has_match: bool
    threshold: float
    candidates: list[CandidateResponse]


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    index_size: int
    model_name: str


def create_app() -> Any:
    """Build the FastAPI app.

    A factory rather than a module-level ``app`` so that tests can construct an
    instance against a small fixture index without touching the real artifacts
    directory.
    """
    raise NotImplementedError
