from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    filename: str
    videoPath: str
    metaPath: str
    bodyPath: Optional[str] = None


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
    takeawayMs: Optional[int] = None
    topMs: Optional[int]
    impactMs: Optional[int]
    finishMs: Optional[int]


class SwingPlane(BaseModel):
    label: str
    confidence: float


class Tempo(BaseModel):
    backswingMs: Optional[int] = None
    downswingMs: Optional[int] = None
    ratio: Optional[float] = None


class ImpactStability(BaseModel):
    label: str
    score: Optional[float] = None


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
    takeaway: Optional[int] = None
    top: Optional[int] = None
    impact: Optional[int] = None
    finish: Optional[int] = None


class GenericMetricPayload(BaseModel):
    label: Optional[str] = None
    confidence: Optional[float] = None
    score: Optional[float] = None
    comment: Optional[str] = None

    model_config = {"extra": "allow"}


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
    body: Optional[Dict[str, GenericMetricPayload]] = None
    club: Optional[Dict[str, GenericMetricPayload]] = None
    fusion: Optional[Dict[str, GenericMetricPayload]] = None


class ProgressPayload(BaseModel):
    stage: str
    stageLabel: Optional[str] = None
    message: Optional[str] = None
    analysisPath: Optional[str] = None
    metaPath: Optional[str] = None
    bodyPath: Optional[str] = None
    clubPath: Optional[str] = None
    fusionPath: Optional[str] = None
    detail: Optional[Dict[str, Any]] = None


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
    analysisQuality: Optional[Dict[str, Any]] = None
    eventValidation: Optional[Dict[str, Any]] = None
    metricQuality: Optional[Dict[str, Dict[str, Any]]] = None
    meta: ResultMeta
    debug: Optional[Dict[str, Any]] = None
    progress: Optional[ProgressPayload] = None


class BodyVideoMeta(BaseModel):
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    durationMs: Optional[int] = None


class BodyVideoRequest(BaseModel):
    jobId: str
    filename: str
    inputPath: str
    force: bool = False
    videoMeta: Optional[BodyVideoMeta] = None


class BodyVideoResponse(BaseModel):
    ok: bool
    jobId: str
    status: Literal["queued", "running", "succeeded", "failed"]
    path: Optional[str] = None
    bodyPath: Optional[str] = None
    metrics: Optional[Dict[str, GenericMetricPayload]] = None
    summary: Optional[str] = None
    errorCode: Optional[str] = None
    errorMessage: Optional[str] = None


class ClubPreprocessLabRequest(BaseModel):
    jobId: str
    inputPath: str
    bodyPath: Optional[str] = None


class ClubPreprocessLabResponse(BaseModel):
    ok: bool
    jobId: str
    labOnly: bool = True
    report: Dict[str, Any]
    scorePath: str


class ClubSidecarRequest(BaseModel):
    """A separate club-only pass that must not replace the primary job result."""

    jobId: str
    metaPath: str
    bodyPath: Optional[str] = None
    takeawayProfile: Optional[str] = None


class ClubSidecarResponse(BaseModel):
    ok: bool
    jobId: str
    result: Dict[str, Any]
