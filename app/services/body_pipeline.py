import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from app.core.config import Settings

BODY_SAMPLE_TARGET = 24
BODY_FULL_SAMPLE_MAX_FRAMES = 120
BODY_DETECT_MAX_SIDE = 480
POSE_DETECT_MAX_SIDE = 768

POSE_KEYPOINTS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


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


def _create_pose_detector():
    try:
        import mediapipe as mp  # type: ignore
    except Exception:
        return None

    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    )


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


def _sample_stride(frame_count: int, fps: float = 0.0, target_fps: float = 0.0) -> int:
    if target_fps > 0 and fps > 0:
        return max(1, int(round(fps / min(target_fps, fps))))
    if frame_count <= 0:
        return 1
    if frame_count <= BODY_FULL_SAMPLE_MAX_FRAMES:
        return 1
    return max(1, frame_count // BODY_SAMPLE_TARGET)


def _effective_sample_fps(fps: float, stride: int) -> float:
    if fps <= 0:
        return 0.0
    return fps / max(1, stride)


def _pose_track_quality(
    frames: List[Dict[str, Any]], effective_fps: float, visibility_threshold: float = 0.25
) -> Dict[str, Any]:
    frame_count = len(frames)
    frame_ms = 1000.0 / effective_fps if effective_fps > 0 else 0.0
    joints: Dict[str, Dict[str, Any]] = {}
    for name in POSE_KEYPOINTS:
        usable = []
        for frame in frames:
            keypoints = frame.get("keypoints") if isinstance(frame, dict) else None
            point = keypoints.get(name) if isinstance(keypoints, dict) else None
            usable.append(
                isinstance(point, list)
                and len(point) >= 3
                and float(point[2]) >= visibility_threshold
            )

        longest_gap = 0
        current_gap = 0
        for is_usable in usable:
            if is_usable:
                longest_gap = max(longest_gap, current_gap)
                current_gap = 0
            else:
                current_gap += 1
        longest_gap = max(longest_gap, current_gap)
        usable_count = sum(usable)
        joints[name] = {
            "usableFrames": usable_count,
            "sampledFrames": frame_count,
            "usableCoverage": round(usable_count / frame_count, 6) if frame_count else 0.0,
            "maxGapFrames": longest_gap,
            "maxGapMs": int(round(longest_gap * frame_ms)) if frame_ms > 0 else None,
        }

    core_names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_wrist", "right_wrist")
    core_coverage = min((joints[name]["usableCoverage"] for name in core_names), default=0.0)
    return {
        "label": "usable" if frame_count >= 90 and core_coverage >= 0.8 else "limited",
        "score": round(core_coverage, 6),
        "confidence": round(core_coverage, 6),
        "visibilityThreshold": visibility_threshold,
        "sampledFrames": frame_count,
        "source": "mediapipe_pose_legacy",
        "joints": joints,
    }


def _resize_for_detection(frame) -> Tuple[Any, float]:
    height, width = frame.shape[:2]
    longest_side = max(width, height)
    if longest_side <= BODY_DETECT_MAX_SIDE:
        return frame, 1.0
    scale = BODY_DETECT_MAX_SIDE / float(longest_side)
    resized = cv2.resize(
        frame,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _resize_for_pose(frame) -> Any:
    height, width = frame.shape[:2]
    longest_side = max(width, height)
    if longest_side <= POSE_DETECT_MAX_SIDE:
        return frame
    scale = POSE_DETECT_MAX_SIDE / float(longest_side)
    return cv2.resize(
        frame,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _extract_pose_keypoints(pose_detector, frame) -> Optional[Dict[str, List[float]]]:
    if pose_detector is None:
        return None
    pose_frame = _resize_for_pose(frame)
    rgb = cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGB)
    try:
        result = pose_detector.process(rgb)
    except Exception:
        return None
    landmarks = getattr(result, "pose_landmarks", None)
    if not landmarks:
        return None

    keypoints: Dict[str, List[float]] = {}
    for name, idx in POSE_KEYPOINTS.items():
        if idx >= len(landmarks.landmark):
            continue
        landmark = landmarks.landmark[idx]
        visibility = float(getattr(landmark, "visibility", 0.0) or 0.0)
        keypoints[name] = [
            round(float(landmark.x), 6),
            round(float(landmark.y), 6),
            round(visibility, 6),
        ]
    return keypoints or None


def _roi_bounds(
    keypoints: Optional[Dict[str, List[float]]], names: Tuple[str, ...], margin: float
) -> Optional[Tuple[float, float, float, float]]:
    if not keypoints:
        return None
    points = [keypoints.get(name) for name in names]
    valid = [point for point in points if isinstance(point, list) and len(point) >= 3 and float(point[2]) >= 0.15]
    if len(valid) < 2:
        return None
    xs = [float(point[0]) for point in valid]
    ys = [float(point[1]) for point in valid]
    return (
        max(0.0, min(xs) - margin),
        max(0.0, min(ys) - margin),
        min(1.0, max(xs) + margin),
        min(1.0, max(ys) + margin),
    )


def _roi_flow_motion(previous_gray, gray, keypoints: Optional[Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    """Compute pose-anchored Farneback flow summaries for the sampled frame."""
    if previous_gray is None or previous_gray.shape != gray.shape:
        return {}
    flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    height, width = gray.shape[:2]
    roi_specs = {
        "upper": ("nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"),
        "torso": ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
        "lower": ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"),
    }
    result: Dict[str, Dict[str, float]] = {}
    for name, points in roi_specs.items():
        bounds = _roi_bounds(keypoints, points, 0.06)
        if not bounds:
            continue
        x0, y0, x1, y1 = bounds
        left, top = int(x0 * width), int(y0 * height)
        right, bottom = max(left + 1, int(x1 * width)), max(top + 1, int(y1 * height))
        roi = flow[top:bottom, left:right]
        if roi.size == 0:
            continue
        mean_dx = float(roi[..., 0].mean()) / float(max(width, 1))
        mean_dy = float(roi[..., 1].mean()) / float(max(height, 1))
        magnitude = float(cv2.magnitude(roi[..., 0], roi[..., 1]).mean()) / float(max(width, height, 1))
        result[name] = {
            "magnitude": round(magnitude, 7),
            "dx": round(mean_dx, 7),
            "dy": round(mean_dy, 7),
        }
    return result


def _scale_box(box: Tuple[int, int, int, int], scale: float) -> Tuple[int, int, int, int]:
    if scale <= 0 or scale == 1.0:
        return box
    x, y, w, h = box
    return (
        int(round(x / scale)),
        int(round(y / scale)),
        int(round(w / scale)),
        int(round(h / scale)),
    )


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
    target_fps = settings.body_pose_target_fps
    stride = _sample_stride(frame_count, fps, target_fps)
    detector = _create_detector()
    pose_detector = _create_pose_detector()

    sampled_frames: List[Dict[str, Any]] = []
    detected_frames = 0
    pose_frames = 0
    wrist_frames = 0
    best_conf = 0.0
    processed = 0
    frame_index = 0
    previous_flow_gray = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue

            processed += 1
            keypoints = _extract_pose_keypoints(pose_detector, frame)
            if keypoints:
                pose_frames += 1
                left_wrist = keypoints.get("left_wrist")
                right_wrist = keypoints.get("right_wrist")
                if (left_wrist and left_wrist[2] >= 0.25) or (right_wrist and right_wrist[2] >= 0.25):
                    wrist_frames += 1
            detection_frame, detection_scale = _resize_for_detection(frame)
            gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
            roi_motion = _roi_flow_motion(previous_flow_gray, gray, keypoints)
            previous_flow_gray = gray
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
                box = _scale_box(box, detection_scale)
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
                        "keypoints": keypoints,
                        "keypointSource": "observed" if keypoints else "missing",
                        "roiMotion": roi_motion,
                    }
                )
            else:
                sampled_frames.append(
                    {
                        "frameIndex": frame_index,
                        "timeMs": time_ms,
                        "personBox": None,
                        "keypoints": keypoints,
                        "keypointSource": "observed" if keypoints else "missing",
                        "roiMotion": roi_motion,
                    }
                )
            frame_index += 1
    finally:
        capture.release()
        if pose_detector is not None:
            pose_detector.close()

    coverage = float(detected_frames / processed) if processed > 0 else 0.0
    pose_coverage = float(pose_frames / processed) if processed > 0 else 0.0
    wrist_coverage = float(wrist_frames / processed) if processed > 0 else 0.0
    effective_sample_fps = _effective_sample_fps(fps, stride)
    pose_track_quality = _pose_track_quality(sampled_frames, effective_sample_fps)
    label = "available" if detected_frames > 0 else "missing"
    summary = (
        f"body bootstrap complete: sampled {processed} frames, "
        f"person detected in {detected_frames} frames."
    )

    result = {
        "ok": True,
        "jobId": job_id,
        "status": "succeeded",
        "analysisVersion": "body-bootstrap-v2",
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
            "sampleTargetFrames": BODY_SAMPLE_TARGET,
            "sampleMode": "target_fps" if target_fps > 0 else "legacy_target_count",
            "sampleTargetFps": round(target_fps, 3) if target_fps > 0 else None,
            "effectiveSampleFps": round(effective_sample_fps, 3),
            "sampledFrames": processed,
            "detectMaxSide": BODY_DETECT_MAX_SIDE,
            "poseMaxSide": POSE_DETECT_MAX_SIDE,
            "poseAvailable": pose_detector is not None,
            "roiMotion": {
                "method": "farneback-pose-anchored-v1",
                "regions": ["upper", "torso", "lower"],
            },
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
            "poseCoverage": {
                "label": "available" if pose_frames > 0 else "missing",
                "score": round(pose_coverage, 6),
                "confidence": round(pose_coverage, 6),
                "detectedFrames": pose_frames,
                "sampledFrames": processed,
            },
            "wristCoverage": {
                "label": "available" if wrist_frames > 0 else "missing",
                "score": round(wrist_coverage, 6),
                "confidence": round(wrist_coverage, 6),
                "detectedFrames": wrist_frames,
                "sampledFrames": processed,
            },
            "poseTrackQuality": pose_track_quality,
        },
        "summary": summary,
        "debug": {
            "frameCount": frame_count,
            "processedFrames": processed,
            "detectedFrames": detected_frames,
            "poseFrames": pose_frames,
            "wristFrames": wrist_frames,
            "poseAvailable": pose_detector is not None,
            "bestConfidence": round(best_conf, 6),
            "detectMaxSide": BODY_DETECT_MAX_SIDE,
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
