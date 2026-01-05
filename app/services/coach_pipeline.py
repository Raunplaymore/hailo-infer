import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class CoachError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _normalize_times(frames: List[dict], fps: int) -> List[Optional[float]]:
    times = []
    for frame in frames:
        t = frame.get("t")
        if t is None and "frame" in frame:
            t = frame["frame"] / float(fps)
        times.append(t)
    if not times or all(t is None for t in times):
        return [None for _ in frames]
    max_t = max(t for t in times if t is not None)
    if max_t < 1000:
        return [(t * 1000.0 if t is not None else None) for t in times]
    return [t for t in times]


def _select_clubhead(frames: List[dict]) -> Tuple[List[Tuple[float, float]], List[float], List[Tuple[float, float]], List[float]]:
    centers = []
    times = []
    sizes = []
    confs = []
    for frame in frames:
        detections = frame.get("detections", [])
        candidates = [d for d in detections if d.get("label") == "clubhead"]
        if not candidates:
            continue
        best = max(candidates, key=lambda d: d.get("conf", 0.0))
        bbox = best.get("bbox", [0, 0, 0, 0])
        x, y, w, h = bbox
        centers.append((x + w / 2.0, y + h / 2.0))
        sizes.append((w, h))
        times.append(frame.get("_t_ms"))
        confs.append(best.get("conf", 0.0))
    return centers, times, sizes, confs


def _speeds(points: List[Tuple[float, float]]) -> List[float]:
    speeds = [0.0]
    for idx in range(1, len(points)):
        dx = points[idx][0] - points[idx - 1][0]
        dy = points[idx][1] - points[idx - 1][1]
        speeds.append(math.hypot(dx, dy))
    return speeds


def _find_stable_index(speeds: List[float], start: int, direction: int) -> Optional[int]:
    if not speeds:
        return None
    median_speed = np.median(speeds)
    threshold = max(1.0, median_speed * 0.35)
    streak = 0
    idx = start
    while 0 <= idx < len(speeds):
        if speeds[idx] <= threshold:
            streak += 1
            if streak >= 3:
                return idx
        else:
            streak = 0
        idx += direction
    return None


def _find_top(points: List[Tuple[float, float]], impact_idx: int) -> Optional[int]:
    if impact_idx <= 1:
        return None
    sample = points[: max(2, min(6, impact_idx))]
    dx = sample[-1][0] - sample[0][0]
    direction = 1 if dx >= 0 else -1
    xs = [p[0] for p in points[:impact_idx]]
    if not xs:
        return None
    if direction >= 0:
        return int(np.argmax(xs))
    return int(np.argmin(xs))


def _tempo(address_ms: int, top_ms: int, impact_ms: int) -> Tuple[int, int, float]:
    backswing = max(0, top_ms - address_ms)
    downswing = max(1, impact_ms - top_ms)
    ratio = round(backswing / float(downswing), 2)
    return backswing, downswing, ratio


def _impact_stability(points: List[Tuple[float, float]], sizes: List[Tuple[float, float]], impact_idx: int) -> Tuple[str, float]:
    start = max(0, impact_idx - 3)
    end = min(len(points), impact_idx + 4)
    window = points[start:end]
    if len(window) < 2:
        return "unstable", 0.0
    arr = np.array(window)
    std = float(np.linalg.norm(arr.std(axis=0)))
    if sizes:
        size_arr = np.array(sizes[start:end])
        scale = float(np.mean(size_arr)) if size_arr.size else 1.0
    else:
        scale = 1.0
    score = max(0.0, min(1.0, 1.0 - std / (scale + 1e-6)))
    label = "stable" if score >= 0.6 else "unstable"
    return label, round(score, 2)


def analyze_meta(meta: Dict[str, object], job_id: str, force: bool) -> Dict[str, object]:
    frames = meta.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise CoachError("NOT_SWING", "meta frames missing")

    fps = int(meta.get("fps", 60))
    times_ms = _normalize_times(frames, fps)
    for idx, frame in enumerate(frames):
        frame["_t_ms"] = times_ms[idx] if times_ms[idx] is not None else idx * (1000.0 / fps)

    centers, times, sizes, confs = _select_clubhead(frames)
    if len(centers) < 15:
        if not force:
            raise CoachError("NOT_SWING", "insufficient clubhead detections")
    if confs and np.mean(confs) < 0.25 and not force:
        raise CoachError("NOT_SWING", "clubhead detections too weak")

    speeds = _speeds(centers)
    if not speeds or max(speeds) <= 0:
        if not force:
            raise CoachError("NOT_SWING", "insufficient motion")

    impact_idx = int(np.argmax(speeds)) if speeds else None
    top_idx = _find_top(centers, impact_idx or 0) if impact_idx is not None else None
    address_idx = _find_stable_index(speeds, 0, 1)
    finish_idx = _find_stable_index(speeds, len(speeds) - 1, -1)

    if impact_idx is None or top_idx is None or address_idx is None or finish_idx is None:
        if not force:
            raise CoachError("NOT_SWING", "event segmentation failed")

    address_idx = address_idx or 0
    top_idx = top_idx or min(len(centers) - 1, address_idx + 1)
    impact_idx = impact_idx or min(len(centers) - 1, top_idx + 1)
    finish_idx = finish_idx or (len(centers) - 1)

    address_ms = int(times[address_idx])
    top_ms = int(times[top_idx])
    impact_ms = int(times[impact_idx])
    finish_ms = int(times[finish_idx])

    dx = centers[impact_idx][0] - centers[top_idx][0]
    dy = centers[impact_idx][1] - centers[top_idx][1]
    swing_label = "inside-out" if dx >= 0 else "outside-in"
    swing_conf = round(min(1.0, abs(dx) / (abs(dx) + abs(dy) + 1e-6)), 2)

    backswing_ms, downswing_ms, ratio = _tempo(address_ms, top_ms, impact_ms)
    tempo = {
        "backswingMs": backswing_ms,
        "downswingMs": downswing_ms,
        "ratio": ratio,
    }

    impact_label, impact_score = _impact_stability(centers, sizes, impact_idx)

    summary = f"Swing plane {swing_label}. Impact stability {impact_label}. Tempo {ratio}:1."

    duration_ms = int(frames[-1]["_t_ms"]) if frames else 0

    return {
        "ok": True,
        "jobId": job_id,
        "status": "done",
        "errorCode": None,
        "errorMessage": None,
        "events": {
            "addressMs": address_ms,
            "topMs": top_ms,
            "impactMs": impact_ms,
            "finishMs": finish_ms,
        },
        "metrics": {
            "swingPlane": {
                "label": swing_label,
                "confidence": swing_conf,
            },
            "tempo": tempo,
            "impactStability": {
                "label": impact_label,
                "score": impact_score,
            },
        },
        "summary": summary,
        "meta": {
            "fps": fps,
            "width": meta.get("width"),
            "height": meta.get("height"),
            "durationMs": duration_ms,
            "analysisVersion": "hailo-coach-v1",
        },
        "debug": {
            "points": float(len(centers)),
            "speedMax": float(max(speeds)) if speeds else 0.0,
        },
    }
