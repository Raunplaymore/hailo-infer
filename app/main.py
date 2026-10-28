import subprocess

from fastapi import BackgroundTasks, FastAPI, HTTPException

from app.core.config import Settings
from app.schemas import (
    BodyVideoRequest,
    BodyVideoResponse,
    ClubSidecarRequest,
    ClubSidecarResponse,
    ClubPreprocessLabRequest,
    ClubPreprocessLabResponse,
    HealthResponse,
    ImpactReviewRequest,
    PracticeReviewRequest,
    PracticeFramesRequest,
    JobCreateRequest,
    JobCreateResponse,
    JobStatusResponse,
)
from app.services.body_pipeline import BodyPipelineError, analyze_body_video
from app.services.club_preprocess_lab import ClubPreprocessLabError, run_club_preprocess_lab
from app.services.coach_pipeline import COACH_ANALYSIS_VERSION, CoachError, analyze_meta
from app.services.impact_review import REVIEW_LOCK, review_impact
from app.services.job_store import JobStore
from app.services.club_observation import observe
from app.services.meta_loader import MetaLoadError, load_meta
from app.services.result_completion import is_complete_coach_result
from app.services.practice_review import review_saved_body, preview_saved_body
from app.services.coaching_review import CoachingBusy, review_coaching

APP_VERSION = "0.1.0"

settings = Settings()
store = JobStore(settings)
app = FastAPI(title="hailo-infer", version=APP_VERSION)


def _is_complete_coach_result(result: dict | None) -> bool:
    return is_complete_coach_result(result, COACH_ANALYSIS_VERSION)


def _run_job(job_id: str, payload: JobCreateRequest) -> None:
    if store.is_canceled(job_id):
        return
    store.set_status(job_id, "running")
    try:
        if payload.mode == "coach_from_meta":
            meta = load_meta(payload.source.metaPath)
            # Capture geometry belongs to the request contract.  Older metadata
            # remains compatible, while an explicit request value takes priority.
            meta = dict(meta)
            if payload.options.viewpoint != "unknown":
                meta["viewpoint"] = payload.options.viewpoint
            if payload.options.handedness != "unknown":
                meta["handedness"] = payload.options.handedness
            if payload.options.club:
                meta["club"] = payload.options.club
            result = analyze_meta(
                meta,
                job_id=job_id,
                force=payload.options.force,
                body_path=payload.source.bodyPath,
            )
        else:
            raise CoachError("MODE_DISABLED", "infer_meta_from_video disabled on this host")
        store.save_result(job_id, result)
        store.set_status(job_id, "succeeded")
    except (CoachError, MetaLoadError) as exc:
        error_code = exc.code if isinstance(exc, CoachError) else "META_LOAD_FAILED"
        error_message = str(exc)
        failed = {
            "ok": False,
            "jobId": job_id,
            "status": "failed",
            "errorCode": error_code,
            "errorMessage": error_message,
            "events": {
                "addressMs": None,
                "takeawayMs": None,
                "topMs": None,
                "impactMs": None,
                "finishMs": None,
            },
            "metrics": None,
            "summary": None,
            "meta": {
                "fps": 0,
                "width": None,
                "height": None,
                "durationMs": 0,
                "analysisVersion": "hailo-coach-v1",
            },
        }
        store.save_result(job_id, failed)
        store.set_status(job_id, "failed", error_code=error_code, error_message=error_message)
    except Exception as exc:
        error_message = f"unexpected error: {exc}"
        failed = {
            "ok": False,
            "jobId": job_id,
            "status": "failed",
            "errorCode": "UNEXPECTED",
            "errorMessage": error_message,
            "events": {
                "addressMs": None,
                "takeawayMs": None,
                "topMs": None,
                "impactMs": None,
                "finishMs": None,
            },
            "metrics": None,
            "summary": None,
            "meta": {
                "fps": 0,
                "width": None,
                "height": None,
                "durationMs": 0,
                "analysisVersion": "hailo-coach-v1",
            },
        }
        store.save_result(job_id, failed)
        store.set_status(job_id, "failed", error_code="UNEXPECTED", error_message=error_message)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True, version=APP_VERSION, hailoAvailable=settings.hailo_available)


@app.post("/v1/body/from-video", response_model=BodyVideoResponse)
def create_body_artifact(payload: BodyVideoRequest) -> BodyVideoResponse:
    try:
        result = analyze_body_video(
            settings=settings,
            job_id=payload.jobId,
            filename=payload.filename,
            input_path=payload.inputPath,
            force=payload.force,
            video_meta=payload.videoMeta.model_dump() if payload.videoMeta else None,
        )
        return BodyVideoResponse(**result)
    except BodyPipelineError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "jobId": payload.jobId,
                "status": "failed",
                "errorCode": exc.code,
                "errorMessage": str(exc),
            },
        ) from exc


@app.post("/v1/labs/club-preprocess", response_model=ClubPreprocessLabResponse)
def create_club_preprocess_lab(payload: ClubPreprocessLabRequest) -> ClubPreprocessLabResponse:
    """Run only derived-video detection passes; never create an analysis job."""
    try:
        return ClubPreprocessLabResponse(
            **run_club_preprocess_lab(
                settings=settings,
                job_id=payload.jobId,
                input_path=payload.inputPath,
                body_path=payload.bodyPath,
            )
        )
    except ClubPreprocessLabError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)}) from exc


@app.post("/v1/club-sidecar/from-meta", response_model=ClubSidecarResponse)
def analyze_club_sidecar(payload: ClubSidecarRequest) -> ClubSidecarResponse:
    """Analyze V2 club-only metadata without writing or replacing a primary job."""
    try:
        result = analyze_meta(
            load_meta(payload.metaPath), job_id=payload.jobId, force=False,
            body_path=payload.bodyPath, takeaway_profile=payload.takeawayProfile,
        )
        if payload.impactMs is not None:
            try:
                observation = observe(result, payload.impactMs)
                result["clubObservations"] = {
                    "status": observation["status"], "automaticCoachingReady": False,
                    "revision": 1, "levels": observation["matchedGripHeights"],
                }
            except ValueError as exc:
                result["clubObservations"] = {"status": "withheld", "revision": 1,
                    "automaticCoachingReady": False, "reason": str(exc), "levels": []}
        return ClubSidecarResponse(ok=True, jobId=payload.jobId, result=result)
    except (CoachError, MetaLoadError) as exc:
        error_code = exc.code if isinstance(exc, CoachError) else "META_LOAD_FAILED"
        raise HTTPException(
            status_code=400,
            detail={"ok": False, "jobId": payload.jobId, "errorCode": error_code, "error": str(exc)},
        ) from exc


@app.post("/v1/jobs", response_model=JobCreateResponse)
def create_job(payload: JobCreateRequest, background_tasks: BackgroundTasks) -> JobCreateResponse:
    if payload.mode == "infer_meta_from_video" and not settings.hailo_available:
        raise HTTPException(status_code=400, detail="infer_meta_from_video disabled on this host")

    existing = store.get(payload.jobId)
    if existing and existing.status in {"queued", "running"} and not payload.options.force:
        return JobCreateResponse(ok=True, jobId=payload.jobId, status=existing.status)

    cached = store.load_cached_result(payload.jobId)
    if cached and _is_complete_coach_result(cached) and not payload.options.force:
        store.init_job(payload.jobId, payload.mode, "succeeded")
        return JobCreateResponse(ok=True, jobId=payload.jobId, status="succeeded")

    store.init_job(payload.jobId, payload.mode, "queued")
    background_tasks.add_task(_run_job, payload.jobId, payload)
    return JobCreateResponse(ok=True, jobId=payload.jobId, status="queued")


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    info = store.get(job_id)
    if not info:
        cached = store.load_cached_result(job_id)
        if _is_complete_coach_result(cached):
            status = "succeeded" if cached.get("ok") is True else "failed"
            return JobStatusResponse(
                ok=True,
                jobId=job_id,
                status=status,
                errorCode=cached.get("errorCode"),
                errorMessage=cached.get("errorMessage"),
            )
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatusResponse(
        ok=True,
        jobId=job_id,
        status=info.status,
        errorCode=info.error_code,
        errorMessage=info.error_message,
    )


@app.post("/v1/jobs/{job_id}/cancel", response_model=JobStatusResponse)
def cancel(job_id: str) -> JobStatusResponse:
    info = store.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="job not found")
    store.mark_canceled(job_id)
    return JobStatusResponse(ok=True, jobId=job_id, status="canceled")


@app.get("/v1/jobs/{job_id}/result")
def result(job_id: str) -> dict:
    cached = store.load_cached_result(job_id)
    if _is_complete_coach_result(cached):
        return cached
    info = store.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="job not found")
    status = "running" if info.status in {"queued", "running"} else "pending"
    return {
        "ok": True,
        "jobId": job_id,
        "status": status,
        "errorCode": info.error_code,
        "errorMessage": info.error_message,
        "events": {
            "addressMs": None,
            "takeawayMs": None,
            "topMs": None,
            "impactMs": None,
            "finishMs": None,
        },
        "metrics": None,
        "summary": None,
        "meta": {
            "fps": 0,
            "width": None,
            "height": None,
            "durationMs": 0,
            "analysisVersion": "hailo-coach-v1",
        },
    }


@app.post("/v1/impact-review")
def impact_review(payload: ImpactReviewRequest) -> dict:
    if not REVIEW_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="다른 임팩트 검수가 진행 중입니다. 잠시 후 다시 시도해 주세요.")
    try:
        return review_impact(settings, payload.jobId, payload.inputPath, payload.bodyPath, payload.confirmedInput)
    except (ValueError, OSError, BodyPipelineError, subprocess.SubprocessError) as exc:
        raise HTTPException(status_code=400, detail="임팩트 후보를 만들 수 없습니다. 영상과 자세 입력을 확인해 주세요.") from exc
    finally:
        REVIEW_LOCK.release()


@app.post("/v1/practice-review")
def practice_review(payload: PracticeReviewRequest) -> dict:
    try:
        return review_saved_body(settings, payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="저장된 자세 분석이 없습니다. 영상을 먼저 분석해 주세요.") from exc
    except (ValueError, OSError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        raise HTTPException(status_code=422, detail="선택 구간 또는 영상과 자세 분석의 연결을 확인할 수 없습니다. 영상을 다시 분석하고 장면을 선택해 주세요.") from exc


@app.post("/v1/practice-preview")
def practice_preview(payload: PracticeFramesRequest) -> dict:
    try:
        return preview_saved_body(settings, payload)
    except (ValueError, OSError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        raise HTTPException(status_code=422, detail="저장된 영상의 자세 비교를 준비하지 못했습니다.") from exc


@app.post("/v1/coaching-review")
def coaching_review(payload: PracticeFramesRequest) -> dict:
    try:
        return review_coaching(settings, payload)
    except CoachingBusy as exc:
        raise HTTPException(status_code=409, detail="다른 영상의 근거를 준비 중입니다. 잠시 후 다시 시도해 주세요.") from exc
    except (ValueError, OSError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        raise HTTPException(status_code=422, detail="영상과 분석의 연결을 확인하지 못했습니다. 이 영상을 다시 분석해 주세요.") from exc
