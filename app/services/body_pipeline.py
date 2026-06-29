import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from app.core.config import Settings


class BodyPipelineError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _body_dir(settings: Settings) -> Path:
    return settings.data_dir / "body"


def _body_path(settings: Settings, job_id: str) -> Path:
    return _body_dir(settings) / f"{job_id}.json"


def _create_detector() -> cv2.HOGDescriptor:
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return detector


def _pick_best_detection(rects, weights) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
    if rects is None or len(rects) == 0:
        return None
    best_idx = 0
    best_weight = float(weights[0]) if len(weights) > 0 else 0.0
    for idx in range(1, len(rects)):
        weight = float(weights[idx]) if idx < len(weights) else 0.0
        if weight > best_weight:
            best_idx = idx
            best_weight = weight
    return tuple(rects[best_idx]), best_weight


def _normalize_box(box: Tuple[int, int, int, int], width: int, height: int) -> Dict[str, float]:
    x, y, w, h = box
    safe_w = max(width, 1)
    safe_h = max(height, 1)
    return {
        "x": round(x / safe_w, 6),
        "y": round(y / safe_h, 6),
        "w": round(w / safe_w, 6),
        "h": round(h / safe_h, 6),
    }


def _sample_stride(frame_count: int) -> int:
    if frame_count <= 0:
        return 1
    return max(1, frame_count // 48)


def analyze_body_video(
    *,
    settings: Settings,
    job_id: str,
    filename: str,
    input_path: str,
    force: bool = False,
    video_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source = Path(input_path)
    if not source.exists():
        raise BodyPipelineError("BODY_INPUT_MISSING", f"video not found: {source}")

    out_path = _body_path(settings, job_id)
    if out_path.exists() and not force:
        cached = json.loads(out_path.read_text())
        return {
            "ok": True,
            "jobId": job_id,
            "status": "succeeded",
            "path": str(out_path),
            "bodyPath": str(out_path),
            "metrics": cached.get("metrics"),
            "summary": cached.get("summary"),
        }

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise BodyPipelineError("BODY_VIDEO_OPEN_FAILED", f"failed to open video: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_ms = int(round(frame_count * 1000 / fps)) if fps > 0 and frame_count > 0 else 0
    stride = _sample_stride(frame_count)
    detector = _create_detector()

    sampled_frames: List[Dict[str, Any]] = []
    detected_frames = 0
    best_conf = 0.0
    processed = 0
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue

            processed += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects, weights = detector.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )
            best = _pick_best_detection(rects, weights)
            time_ms = int(round(frame_index * 1000 / fps)) if fps > 0 else None
            if best:
                box, confidence = best
                detected_frames += 1
                best_conf = max(best_conf, float(confidence))
                sampled_frames.append(
                    {
                        "frameIndex": frame_index,
                        "timeMs": time_ms,
                        "personBox": {
                            **_normalize_box(box, width, height),
                            "confidence": round(float(confidence), 6),
                        },
                    }
                )
            else:
                sampled_frames.append(
                    {
                        "frameIndex": frame_index,
                        "timeMs": time_ms,
                        "personBox": None,
                    }
                )
            frame_index += 1
    finally:
        capture.release()

    coverage = float(detected_frames / processed) if processed > 0 else 0.0
    label = "available" if detected_frames > 0 else "missing"
    summary = (
        f"body bootstrap complete: sampled {processed} frames, "
        f"person detected in {detected_frames} frames."
    )

    result = {
        "ok": True,
        "jobId": job_id,
        "status": "succeeded",
        "analysisVersion": "body-bootstrap-v1",
        "source": {
            "filename": filename,
            "inputPath": str(source),
        },
        "meta": {
            "fps": int(round(fps)) if fps > 0 else int(video_meta.get("fps") or 0) if video_meta else 0,
            "width": width or (int(video_meta.get("width")) if video_meta and video_meta.get("width") is not None else None),
            "height": height or (int(video_meta.get("height")) if video_meta and video_meta.get("height") is not None else None),
            "durationMs": duration_ms or (int(video_meta.get("durationMs")) if video_meta and video_meta.get("durationMs") is not None else 0),
            "sampleStride": stride,
            "sampledFrames": processed,
        },
        "frames": sampled_frames,
        "metrics": {
            "bodyPresence": {
                "label": label,
                "confidence": round(min(1.0, max(best_conf / 2.0, coverage)), 6),
                "score": round(coverage, 6),
                "detectedFrames": detected_frames,
                "sampledFrames": processed,
                "comment": "OpenCV HOG person bootstrap. Replace with pose pipeline for production-grade body metrics.",
            },
            "bodyCoverage": {
                "label": "sampled",
                "score": round(coverage, 6),
                "confidence": round(coverage, 6),
                "detectedFrames": detected_frames,
                "sampledFrames": processed,
            },
        },
        "summary": summary,
        "debug": {
            "frameCount": frame_count,
            "processedFrames": processed,
            "detectedFrames": detected_frames,
            "bestConfidence": round(best_conf, 6),
        },
    }

    _body_dir(settings).mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=True, indent=2))
    return {
        "ok": True,
        "jobId": job_id,
        "status": "succeeded",
        "path": str(out_path),
        "bodyPath": str(out_path),
        "metrics": result["metrics"],
        "summary": summary,
    }
