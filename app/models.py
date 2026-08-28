from datetime import datetime
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
    created_at: datetime = Field(default_factory=datetime.utcnow)
    error: str | None = None


class Quarry(BaseModel):
    site_id: str
    name: str | None = None
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
    char_start: int
    char_end: int
    quote: str


class FieldValue(BaseModel):
    value: Any | None = None
    confidence: float = Field(ge=0, le=1)
    abstain_reason: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    
class ExtractionResult(BaseModel):
    name: FieldValue
    material: FieldValue
    operator: FieldValue
    operational_status: FieldValue

class Source(BaseModel):
    source_id: str
    url: str
    source_type: str = "web"
    title: str | None = None
    authority: str | None = None