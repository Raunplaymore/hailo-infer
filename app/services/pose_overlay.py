"""Read a video-aligned pose overlay without changing the stored analysis."""
import math
from pathlib import Path

from app.services.body_pipeline import POSE_KEYPOINTS
from app.services.impact_review import stat_identity
from app.services.practice_review import load_practice_body


def read_pose_overlay(settings, payload):
    source = Path(payload.inputPath).resolve(strict=True)
    body_path = Path(payload.bodyPath).resolve(strict=True)
    if not any(source.is_relative_to(root.resolve()) for root in (settings.upload_dir, settings.data_dir)) or not body_path.is_relative_to(settings.data_dir.resolve()):
        raise ValueError("저장된 영상과 자세 입력만 사용할 수 있습니다.")
    identity = (stat_identity(source.stat()), stat_identity(body_path.stat()))
    body = load_practice_body(settings, payload)
    frames = body["frames"]
    if not 1 <= len(frames) <= 7200:
        raise ValueError("자세 프레임 수를 확인할 수 없습니다.")
    result = []
    for frame in frames:
        points = frame.get("keypoints") if frame.get("keypointSource") in (None, "observed") else None
        points = points if isinstance(points, dict) else {}
        valid = {}
        for name in POSE_KEYPOINTS:
            point = points.get(name)
            if (isinstance(point, (list, tuple)) and len(point) >= 3
                    and all(type(v) in (int, float) and math.isfinite(v) for v in point[:3])
                    and 0 <= point[2] <= 1):
                valid[name] = list(point[:3])
        result.append({"frame": frame["frameIndex"], "timeMs": frame["timeMs"], "keypoints": valid})
    if identity != (stat_identity(source.stat()), stat_identity(body_path.stat())):
        raise ValueError("자세 오버레이를 준비하는 동안 영상이나 자세 분석이 변경됐습니다.")
    available = any(frame["keypoints"] for frame in result)
    return {"ok": True, "jobId": payload.jobId, "status": "ready" if available else "unavailable",
            "timeSource": "video_pts", "poseFrames": result if available else []}
