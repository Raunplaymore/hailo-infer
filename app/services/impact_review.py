"""Human review only; never writes coach results or training labels."""
import copy
import hashlib
import json
import math
import subprocess
import threading
from functools import lru_cache
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


def stat_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def pose_stat_token(identity):
    values = [str(value) for value in identity]
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode('ascii')).hexdigest()


def read_stable_pose(path):
    before = path.stat()
    body = json.loads(path.read_text())
    if stat_identity(before) != stat_identity(path.stat()):
        raise ValueError("자세 분석을 읽는 동안 입력이 변경됐습니다.")
    return body, stat_identity(before)


def source_frames(source):
    before = stat_identity(source.stat())
    fps, times = _source_frames(source, before)
    if before != stat_identity(source.stat()):
        raise ValueError("시간축을 읽는 동안 영상이 변경됐습니다.")
    return fps, list(times)


@lru_cache(maxsize=16)
def _source_frames(source, identity):
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
    try:
        times = [float(f["best_effort_timestamp_time"]) * 1000 for f in json.loads(result.stdout)["frames"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("영상 프레임 시간축을 확인할 수 없습니다.") from exc
    if (len(times) != count or len(times) < 4 or not all(math.isfinite(t) for t in times)
            or any(b <= a for a, b in zip(times, times[1:])) or times[-1] - times[0] > 30000):
        raise ValueError("영상 프레임 시간축을 확인할 수 없습니다.")
    if identity != stat_identity(source.stat()):
        raise ValueError("시간축을 읽는 동안 영상이 변경됐습니다.")
    return fps, tuple(round(t - times[0], 3) for t in times)


def align_body_times(body, times):
    meta = body.get("meta") if isinstance(body, dict) else None
    frames = body.get("frames") if isinstance(body, dict) else None
    stride = meta.get("sampleStride") if isinstance(meta, dict) else None
    if (not isinstance(frames, list) or not isinstance(stride, int) or isinstance(stride, bool) or stride < 1
            or meta.get("sampledFrames") != len(frames)):
        raise ValueError("자세 프레임과 원본 영상의 대응을 확인할 수 없습니다.")
    indices = list(range(0, len(times), stride))
    if len(frames) != len(indices) or any(
        not isinstance(frame, dict) or type(frame.get("frameIndex")) is not int or frame["frameIndex"] != index
        for frame, index in zip(frames, indices)
    ):
        raise ValueError("자세 프레임과 원본 영상의 대응을 확인할 수 없습니다.")
    return {**body, "frames": [{**frame, "timeMs": times[index]} for frame, index in zip(frames, indices)]}


def review_impact(settings, job_id, input_path, body_path=None, input_confirmed=True):
    source = Path(input_path).resolve(strict=True)
    roots = (settings.upload_dir.resolve(), settings.data_dir.resolve())
    if not any(source.is_relative_to(root) for root in roots) or not source.is_file():
        raise ValueError("허용된 업로드 영상이 아닙니다.")
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("검증된 임팩트 후보 모델과 일치하지 않습니다.")
    source_identity = stat_identity(source.stat())
    _, times = source_frames(source)
    digest = file_sha256(source)
    if source_identity != stat_identity(source.stat()):
        raise ValueError("임팩트 분석 중 영상이 변경됐습니다.")
    isolated = copy.copy(settings)
    isolated.data_dir = settings.data_dir / "impact-review" / digest
    isolated.body_pose_target_fps = 30
    body = None
    pose_path, pose_identity, pose_kind = None, None, None
    if body_path:
        cached_path = Path(body_path).resolve()
        if not cached_path.is_relative_to(settings.data_dir.resolve()):
            raise ValueError("허용된 자세 입력이 아닙니다.")
        try:
            cached_stat = cached_path.stat()
            cached, cached_identity = read_stable_pose(cached_path)
            meta = cached.get("meta") if isinstance(cached, dict) else None
            cached_source = cached.get("source") if isinstance(cached, dict) else None
            input_path = cached_source.get("inputPath") if isinstance(cached_source, dict) else None
            if (isinstance(meta, dict) and isinstance(input_path, str) and input_path
                    and Path(input_path).resolve() == source and meta.get("sampleMode") == "target_fps"
                    and meta.get("sampleTargetFps") == 30
                    and cached_stat.st_mtime_ns >= source.stat().st_mtime_ns
                    and cached_stat.st_ctime_ns > source.stat().st_ctime_ns
                    and stat_identity(cached_stat) == cached_identity):
                body = align_body_times(cached, times)
                pose_path, pose_identity, pose_kind = cached_path, cached_identity, "provided"
        except (OSError, ValueError, TypeError):
            pass  # A stale or incomplete cache must not prevent fresh extraction.
    if body is None:
        isolated_cache = isolated.data_dir / "body" / "pose.json"
        expected_cache = settings.data_dir.resolve() / "impact-review" / digest / "body" / "pose.json"
        if isolated_cache.resolve() != expected_cache:
            raise ValueError("격리 자세 입력의 경로를 확인할 수 없습니다.")
        rebuild = False
        if isolated_cache.exists():
            try:
                cached, cached_identity = read_stable_pose(isolated_cache)
                body = align_body_times(cached, times)
                pose_path, pose_identity, pose_kind = isolated_cache, cached_identity, "isolated"
            except (OSError, ValueError, TypeError):
                rebuild = True
        if body is None:
            body_result = analyze_body_video(
                settings=isolated, job_id="pose", filename=source.name,
                input_path=str(source), force=rebuild,
            )
            pose_path = Path(body_result["bodyPath"]).resolve()
            if pose_path != expected_cache:
                raise ValueError("격리 자세 입력의 경로를 확인할 수 없습니다.")
            cached, pose_identity = read_stable_pose(pose_path)
            body = align_body_times(cached, times)
            pose_kind = "isolated"
    prediction = predict(body, MODEL)
    if source_identity != stat_identity(source.stat()):
        raise ValueError("임팩트 분석 중 영상이 변경됐습니다.")
    try:
        current_path = Path(body_path).resolve(strict=True) if pose_kind == "provided" else isolated_cache.resolve(strict=True)
        expected_path = pose_path if pose_kind == "provided" else expected_cache
        if current_path != expected_path or pose_identity != stat_identity(pose_path.stat()):
            raise ValueError("임팩트 분석 중 자세 입력이 변경됐습니다.")
    except OSError as exc:
        raise ValueError("임팩트 분석 중 자세 입력이 변경됐습니다.") from exc
    impact_ms = prediction["events"]["impactMs"]
    if not math.isfinite(impact_ms) or not 0 <= impact_ms <= times[-1]:
        raise ValueError("임팩트 후보가 영상 범위를 벗어났습니다.")
    frame = min(range(len(times)), key=lambda index: abs(times[index] - impact_ms))
    # shortcut: automatic candidates assume one right-handed DTL swing; validate input rejection before coaching use.
    return {
        "ok": True, "jobId": job_id, "status": "reference_candidate",
        "model": "pose-tcn-v1", "modelSha256": MODEL_SHA256, "videoSha256": digest, "timingRevision": 3,
        "poseProvenance": {"version": 1, "kind": pose_kind, "statToken": pose_stat_token(pose_identity)},
        "requiresHumanConfirmation": True, "inputScopeConfirmed": bool(input_confirmed), "publicValue": None,
        "automaticCoachingReady": False, "appliedToPrimary": False,
        "candidate": {"frameIndex": frame, "timeMs": times[frame]},
        "frameTimesMs": times,
        "withheld": ["address", "takeaway", "top", "finish", "tempo", "bodyMovement", "ballFlight"],
    }
