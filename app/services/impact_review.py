"""Human review only; never writes coach results or training labels."""
import copy
import hashlib
import json
import math
import subprocess
import threading
from pathlib import Path

import cv2

from app.services.body_pipeline import analyze_body_video
from scripts.pose_event_model import predict

MODEL = Path(__file__).resolve().parents[1] / "models/pose-event-tcn-v1.npz"
MODEL_SHA256 = "70b338b519565f870ed1fce044a0453f828773e53b9fa9d024391b3b4cf57733"
REVIEW_LOCK = threading.Lock()


def file_sha256(source):
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_frames(source):
    capture = cv2.VideoCapture(str(source))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or not 0 < fps <= 240 or not 4 <= count <= fps * 30:
            raise ValueError("검수는 30초 이하의 읽을 수 있는 단일 스윙 영상만 지원합니다.")
    finally:
        capture.release()
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(source),
    ], capture_output=True, text=True, check=True, timeout=30)
    times = [float(f["best_effort_timestamp_time"]) * 1000 for f in json.loads(result.stdout)["frames"]]
    if len(times) < 4 or not all(math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("영상 프레임 시간축을 확인할 수 없습니다.")
    return fps, [round(t - times[0], 3) for t in times]


def review_impact(settings, job_id, input_path, body_path=None, input_confirmed=True):
    source = Path(input_path).resolve(strict=True)
    roots = (settings.upload_dir.resolve(), settings.data_dir.resolve())
    if not any(source.is_relative_to(root) for root in roots) or not source.is_file():
        raise ValueError("허용된 업로드 영상이 아닙니다.")
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("검증된 임팩트 후보 모델과 일치하지 않습니다.")
    _, times = source_frames(source)
    digest = file_sha256(source)
    isolated = copy.copy(settings)
    isolated.data_dir = settings.data_dir / "impact-review" / digest
    isolated.body_pose_target_fps = 30
    body = None
    if body_path:
        cached_path = Path(body_path).resolve(strict=True)
        if not cached_path.is_relative_to(settings.data_dir.resolve()):
            raise ValueError("허용된 자세 입력이 아닙니다.")
        cached = json.loads(cached_path.read_text())
        meta = cached.get("meta") or {}
        if (Path(cached.get("source", {}).get("inputPath", "")).resolve() == source
                and meta.get("sampleMode") == "target_fps"
                and meta.get("sampleTargetFps") == 30
                and cached_path.stat().st_mtime_ns >= source.stat().st_mtime_ns):
            body = cached
    if body is None:
        body_result = analyze_body_video(
            settings=isolated, job_id="pose", filename=source.name,
            input_path=str(source), force=False,
        )
        body = json.loads(Path(body_result["bodyPath"]).read_text())
    prediction = predict(body, MODEL)
    impact_ms = prediction["events"]["impactMs"]
    if not math.isfinite(impact_ms) or not 0 <= impact_ms <= times[-1]:
        raise ValueError("임팩트 후보가 영상 범위를 벗어났습니다.")
    frame = min(range(len(times)), key=lambda index: abs(times[index] - impact_ms))
    # shortcut: automatic candidates assume one right-handed DTL swing; validate input rejection before coaching use.
    return {
        "ok": True, "jobId": job_id, "status": "reference_candidate",
        "model": "pose-tcn-v1", "modelSha256": MODEL_SHA256, "videoSha256": digest, "timingRevision": 2,
        "requiresHumanConfirmation": True, "inputScopeConfirmed": bool(input_confirmed), "publicValue": None,
        "automaticCoachingReady": False, "appliedToPrimary": False,
        "candidate": {"frameIndex": frame, "timeMs": times[frame]},
        "frameTimesMs": times,
        "withheld": ["address", "takeaway", "top", "finish", "tempo", "bodyMovement", "ballFlight"],
    }
