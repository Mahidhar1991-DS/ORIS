from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class JobRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_km: float = Field(gt=0)
    max_usd_cost: float | None = Field(default=None, gt=0)


class Job(BaseModel):
    job_id: str
    status: str = "queued"
    progress: float = Field(default=0.0, ge=0, le=1)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    error: str | None = None
    result: dict[str, Any] | None = None


class Quarry(BaseModel):
    site_id: str
    name: str | None = None
    municipality: str | None = None
    latitude: float
    longitude: float
    distance_km: float
    website: str | None = None
    source: str = "openstreetmap"


class Document(BaseModel):
    source_id: str
    url: str
    text: str


class Evidence(BaseModel):
    source_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    quote: str


class FieldValue(BaseModel):
    value: Any | None = None
    confidence: float = Field(ge=0, le=1)
    abstain_reason: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class LocationVerification(BaseModel):
    is_verified: bool
    confidence: float = Field(ge=0, le=1)
    extracted_city: str | None = None
    method: str


class ExtractionResult(BaseModel):
    official_name: FieldValue
    site_type: FieldValue
    description: FieldValue
    materials_produced: list[FieldValue] = Field(default_factory=list)
    certifications: list[FieldValue] = Field(default_factory=list)
    operational_status: FieldValue
    location_verification: LocationVerification


class Source(BaseModel):
    source_id: str
    url: str
    source_type: str = "web"
    title: str | None = None
    authority: str | None = None
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    content_hash: str
    trust_tier: str = "unknown"


class ReconciliationCandidate(BaseModel):
    value: Any
    source_id: str
    score: float


class Reconciliation(BaseModel):
    field: str
    candidates: list[ReconciliationCandidate]
    winner_source_id: str
    reason: str


class Provenance(BaseModel):
    sources: list[Source] = Field(default_factory=list)
    reconciliations: list[Reconciliation] = Field(default_factory=list)


class ModelCall(BaseModel):
    model: str
    purpose: str
    tokens_in: int
    tokens_out: int
    usd_cost: float


class Metrics(BaseModel):
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    usd_cost: float = 0.0
    latency_ms: int = 0
    model_calls: list[ModelCall] = Field(default_factory=list)


class RunMetadata(BaseModel):
    run_id: str
    prompt_hash: str
    scraper_version: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SiteRecord(BaseModel):
    site_id: str
    schema_version: str = "2.0.0"
    input: JobRequest
    extraction: ExtractionResult
    provenance: Provenance
    metrics: Metrics
    run_metadata: RunMetadata