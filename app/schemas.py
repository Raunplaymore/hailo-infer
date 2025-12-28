from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    filename: str
    videoPath: str
    metaPath: str


class JobOptions(BaseModel):
    force: bool = False
    tailFramesForLive: int = 30


class JobCreateRequest(BaseModel):
    mode: Literal["coach_from_meta", "infer_meta_from_video"]
    jobId: str
    source: Source
    options: JobOptions = Field(default_factory=JobOptions)


class JobCreateResponse(BaseModel):
    ok: bool
    jobId: str
    status: Literal["queued", "running", "succeeded", "failed", "canceled"]


class JobStatusResponse(BaseModel):
    ok: bool
    jobId: str
    status: Literal["queued", "running", "succeeded", "failed", "canceled"]
    errorCode: Optional[str] = None
    errorMessage: Optional[str] = None


class HealthResponse(BaseModel):
    ok: bool
    version: str
    hailoAvailable: bool


class Events(BaseModel):
    addressMs: Optional[int]
    topMs: Optional[int]
    impactMs: Optional[int]
    finishMs: Optional[int]


class SwingPlane(BaseModel):
    label: str
    confidence: float


class Tempo(BaseModel):
    backswingMs: int
    downswingMs: int
    ratio: float


class ImpactStability(BaseModel):
    label: str
    score: float


class Metrics(BaseModel):
    swingPlane: SwingPlane
    tempo: Tempo
    impactStability: ImpactStability


class ResultMeta(BaseModel):
    fps: int
    width: Optional[int]
    height: Optional[int]
    durationMs: int
    analysisVersion: str


class JobResult(BaseModel):
    ok: bool
    jobId: str
    status: Literal["done", "failed", "running", "pending"]
    errorCode: Optional[str]
    errorMessage: Optional[str]
    events: Events
    metrics: Optional[Metrics]
    summary: Optional[str]
    meta: ResultMeta
    debug: Optional[Dict[str, float]] = None
