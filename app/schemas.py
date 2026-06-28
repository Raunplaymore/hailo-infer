from typing import Dict, List, Literal, Optional

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


class ShaftPlane(BaseModel):
    label: str
    confidence: float
    angleDeg: Optional[float] = None
    addressAngleDeg: Optional[float] = None
    sampleCount: Optional[int] = None
    comment: Optional[str] = None


class Backswing(BaseModel):
    label: str
    score: float
    clubTravelRatio: Optional[float] = None
    topHeightRatio: Optional[float] = None
    comment: Optional[str] = None


class Readiness(BaseModel):
    label: str
    confidence: float
    readyFrames: int = 0
    notReadyFrames: int = 0


class TrackingQuality(BaseModel):
    label: str
    score: float
    frames: int
    clubHeadFrames: int = 0
    clubHandleFrames: int = 0
    clubFrames: int = 0
    ballFrames: int = 0
    personFrames: int = 0
    clubHeadConfidence: Optional[float] = None
    clubHandleConfidence: Optional[float] = None
    clubConfidence: Optional[float] = None
    ballConfidence: Optional[float] = None
    personConfidence: Optional[float] = None


class BallMetric(BaseModel):
    launchDirection: str = "unknown"
    launchAngle: Optional[float] = None
    speedRelative: str = "unknown"
    confidence: Optional[float] = None


class EventTimingMetric(BaseModel):
    address: Optional[int] = None
    top: Optional[int] = None
    impact: Optional[int] = None
    finish: Optional[int] = None


class Metrics(BaseModel):
    swingPlane: SwingPlane
    tempo: Tempo
    impactStability: ImpactStability
    shaftPlane: Optional[ShaftPlane] = None
    backswing: Optional[Backswing] = None
    readiness: Optional[Readiness] = None
    trackingQuality: Optional[TrackingQuality] = None
    ball: Optional[BallMetric] = None
    eventTiming: Optional[EventTimingMetric] = None


class ResultMeta(BaseModel):
    fps: int
    width: Optional[int]
    height: Optional[int]
    durationMs: int
    analysisVersion: str
    modelLabels: Optional[Dict[int, str]] = None


class JobResult(BaseModel):
    ok: bool
    jobId: str
    status: Literal["done", "failed", "running", "pending"]
    errorCode: Optional[str]
    errorMessage: Optional[str]
    events: Events
    metrics: Optional[Metrics]
    summary: Optional[str]
    coachSummary: Optional[List[str]] = None
    confidence: Optional[float] = None
    meta: ResultMeta
    debug: Optional[Dict[str, float]] = None
