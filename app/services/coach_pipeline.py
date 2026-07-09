import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.services.body_event_selector import select_body_events

COACH_ANALYSIS_VERSION = "hailo-coach-service7-v3"

SERVICE7_LABELS = {
    0: "person",
    1: "player_ready",
    2: "player_not_ready",
    3: "golf_ball",
    4: "club_head",
    5: "club",
    6: "club_handle",
}

PERSON_LABELS = {"person", "player", "golfer", "player_ready", "player_not_ready", "ready", "not_ready", "player_unready"}
READY_LABELS = {"player_ready", "ready"}
NOT_READY_LABELS = {"player_not_ready", "not_ready", "player_unready"}
BALL_LABELS = {"golf_ball", "golfball", "ball"}
CLUBHEAD_LABELS = {"club_head", "clubhead", "club_head_center", "golf_club_head"}
CLUB_LABELS = {"club", "golf_club", "baseball_bat", "bat"}
HANDLE_LABELS = {"club_handle", "handle", "grip", "club_grip"}
MOTION_SOURCE_PRIORITY = {"club_head": 0, "club_handle": 1, "club_box_endpoint": 2, "club": 3}
WRIST_CONFIDENCE_MIN = 0.25
WRIST_PERSON_HEIGHT_NORM = 0.72
WRIST_MIN_EVENT_TRAVEL_RATIO = 0.08
WRIST_MIN_EVENT_RISE_RATIO = 0.035
EVENT_FALLBACK_ENABLED = False


class CoachError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _normalize_viewpoint(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _body_selector_is_operational(viewpoint: str, selector_result: Dict[str, Any]) -> bool:
    viewpoint = viewpoint or "unknown"
    if not selector_result.get("available"):
        return False
    # The offset fallback improved some weak cases but is not stable enough for
    # production event timing yet.
    if selector_result.get("method") == "feature-vote-offset-fallback":
        return False
    if viewpoint == "down_the_line":
        return True
    if viewpoint == "unknown":
        debug = selector_result.get("debug") if isinstance(selector_result.get("debug"), dict) else {}
        return (
            selector_result.get("method") == "feature-vote-early-top-cluster"
            or debug.get("feature") == "shoulder_width/local_max"
        )
    return False


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    avg = _mean(values)
    return math.sqrt(_mean([(value - avg) ** 2 for value in values]))


def _argmax(values: List[float]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])


def _argmin(values: List[float]) -> int:
    return min(range(len(values)), key=lambda idx: values[idx])


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        num = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return num if math.isfinite(num) else default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        num = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return num


def _normalize_label(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _detection_label(det: dict) -> str:
    for key in ("label", "name", "class_name", "className", "category"):
        if det.get(key) is not None:
            return _normalize_label(det.get(key))
    for key in ("class_id", "classId", "cls", "id"):
        if det.get(key) is None:
            continue
        try:
            class_id = int(det.get(key))
        except (TypeError, ValueError):
            continue
        if class_id in SERVICE7_LABELS:
            return SERVICE7_LABELS[class_id]
    return ""


def _detection_conf(det: dict) -> float:
    for key in ("conf", "confidence", "score", "prob"):
        if det.get(key) is not None:
            return _clamp(_safe_float(det.get(key)))
    return 0.0


def _bbox_xywh(det: dict) -> Optional[Tuple[float, float, float, float]]:
    bbox = det.get("bbox") or det.get("box") or det.get("rect")
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x", "y", "w", "h")):
            x = _safe_float(bbox.get("x"))
            y = _safe_float(bbox.get("y"))
            w = abs(_safe_float(bbox.get("w")))
            h = abs(_safe_float(bbox.get("h")))
            return (x, y, w, h)
        if all(k in bbox for k in ("x", "y", "width", "height")):
            x = _safe_float(bbox.get("x"))
            y = _safe_float(bbox.get("y"))
            w = abs(_safe_float(bbox.get("width")))
            h = abs(_safe_float(bbox.get("height")))
            return (x, y, w, h)
        x1_key = "x1" if "x1" in bbox else "xmin"
        y1_key = "y1" if "y1" in bbox else "ymin"
        x2_key = "x2" if "x2" in bbox else "xmax"
        y2_key = "y2" if "y2" in bbox else "ymax"
        if all(k in bbox for k in (x1_key, y1_key, x2_key, y2_key)):
            x1 = _safe_float(bbox.get(x1_key))
            y1 = _safe_float(bbox.get(y1_key))
            x2 = _safe_float(bbox.get(x2_key))
            y2 = _safe_float(bbox.get(y2_key))
            return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x = _safe_float(bbox[0])
        y = _safe_float(bbox[1])
        a = _safe_float(bbox[2])
        b = _safe_float(bbox[3])
        fmt = _normalize_label(det.get("bboxFormat") or det.get("box_format"))
        if fmt in {"xyxy", "x1y1x2y2"}:
            return (min(x, a), min(y, b), abs(a - x), abs(b - y))
        return (x, y, abs(a), abs(b))

    return None


def _frame_detections(frame: dict) -> List[dict]:
    detections = frame.get("detections")
    if detections is None:
        detections = frame.get("objects") or frame.get("predictions") or []
    return detections if isinstance(detections, list) else []


def _normalize_times(frames: List[dict], fps: int) -> List[Optional[float]]:
    times: List[Optional[float]] = []
    explicit_ms = False
    for frame in frames:
        t = frame.get("t")
        if t is None:
            for key in ("timeMs", "time_ms", "timestampMs"):
                if frame.get(key) is not None:
                    t = frame.get(key)
                    explicit_ms = True
                    break
        if t is None and "frame" in frame:
            t = _safe_float(frame["frame"]) / float(fps)
        times.append(None if t is None else _safe_float(t))
    if not times or all(t is None for t in times):
        return [None for _ in frames]
    max_t = max(t for t in times if t is not None)
    if not explicit_ms and max_t < 1000:
        expected_seconds = len(frames) / float(max(1, fps))
        if max_t > expected_seconds * 4.0:
            return times
        return [(t * 1000.0 if t is not None else None) for t in times]
    return times


def _select_best_track(frames: List[dict], labels: Iterable[str]) -> List[dict]:
    wanted = {_normalize_label(label) for label in labels}
    track: List[dict] = []
    for frame_idx, frame in enumerate(frames):
        candidates = [
            det
            for det in _frame_detections(frame)
            if _detection_label(det) in wanted and _bbox_xywh(det) is not None
        ]
        if not candidates:
            continue
        best = max(candidates, key=_detection_conf)
        bbox = _bbox_xywh(best)
        if bbox is None:
            continue
        x, y, w, h = bbox
        t_ms = frame.get("_t_ms")
        track.append(
            {
                "x": x + w / 2.0,
                "y": y + h / 2.0,
                "w": w,
                "h": h,
                "t": _safe_float(t_ms, frame_idx * 16.67),
                "frame": _safe_int(frame.get("frame", frame_idx), frame_idx),
                "conf": _detection_conf(best),
                "label": _detection_label(best),
            }
        )
    return track


def _club_box_endpoint_track(club_track: List[dict]) -> List[dict]:
    if not club_track:
        return []

    candidate_tracks = ([], [])
    for point in club_track:
        x = _safe_float(point.get("x"), 0.0)
        y = _safe_float(point.get("y"), 0.0)
        w = _safe_float(point.get("w"), 0.0)
        h = _safe_float(point.get("h"), 0.0)
        if w >= h:
            endpoints = ((x - w / 2.0, y), (x + w / 2.0, y))
        else:
            endpoints = ((x, y - h / 2.0), (x, y + h / 2.0))

        for idx, (endpoint_x, endpoint_y) in enumerate(endpoints):
            candidate_tracks[idx].append(
                {
                    **point,
                    "x": endpoint_x,
                    "y": endpoint_y,
                    "source": "club_box_endpoint",
                    "proxySource": "club_box_endpoint",
                }
            )

    def endpoint_score(track: List[dict]) -> float:
        speeds = _speeds(track) if len(track) >= 2 else []
        xs = [_safe_float(point.get("x"), 0.0) for point in track]
        ys = [_safe_float(point.get("y"), 0.0) for point in track]
        x_range = max(xs) - min(xs) if xs else 0.0
        y_range = max(ys) - min(ys) if ys else 0.0
        return sum(speeds) + math.hypot(x_range, y_range)

    return max(candidate_tracks, key=endpoint_score)


def _merge_motion_track(named_tracks: List[Tuple[str, List[dict]]]) -> List[dict]:
    by_frame: Dict[int, dict] = {}
    for source, track in named_tracks:
        priority = MOTION_SOURCE_PRIORITY.get(source, 99)
        for point in track:
            frame = _safe_int(point.get("frame"), -1)
            if frame < 0:
                continue
            candidate = {**point, "source": source}
            existing = by_frame.get(frame)
            if not existing:
                by_frame[frame] = candidate
                continue
            existing_priority = MOTION_SOURCE_PRIORITY.get(existing.get("source", ""), 99)
            if priority < existing_priority or (priority == existing_priority and candidate.get("conf", 0.0) >= existing.get("conf", 0.0)):
                by_frame[frame] = candidate
    merged = list(by_frame.values())
    merged.sort(key=lambda point: (_safe_int(point.get("frame"), 0), _safe_float(point.get("t"), 0.0)))
    return merged


def _choose_motion_track(club_head_track: List[dict], handle_track: List[dict], club_track: List[dict]) -> Tuple[List[dict], str]:
    named_tracks: List[Tuple[str, List[dict]]] = []
    club_endpoint_track = _club_box_endpoint_track(club_track)
    if club_head_track:
        named_tracks.append(("club_head", club_head_track))
    if handle_track:
        named_tracks.append(("club_handle", handle_track))
    if club_endpoint_track:
        named_tracks.append(("club_box_endpoint", club_endpoint_track))
    if club_track:
        named_tracks.append(("club", club_track))
    if not named_tracks:
        return [], "club_head"

    hybrid_track = _merge_motion_track(named_tracks)
    candidates: List[Tuple[str, List[dict]]] = named_tracks + [("hybrid", hybrid_track)]

    def rank(item: Tuple[str, List[dict]]) -> Tuple[int, float, int]:
        source, track = item
        preferred = 1 if source != "hybrid" else 0
        return (len(track), _mean([_safe_float(point.get("conf"), 0.0) for point in track]), preferred)

    source, track = max(candidates, key=rank)
    return track, source


def _label_conf_stats(frames: List[dict], labels: Iterable[str]) -> Tuple[int, float]:
    wanted = {_normalize_label(label) for label in labels}
    confs: List[float] = []
    for frame in frames:
        frame_confs = [
            _detection_conf(det)
            for det in _frame_detections(frame)
            if _detection_label(det) in wanted
        ]
        if frame_confs:
            confs.append(max(frame_confs))
    avg = _mean(confs)
    return len(confs), avg


def _speeds(track: List[dict]) -> List[float]:
    speeds = [0.0]
    for idx in range(1, len(track)):
        dx = track[idx]["x"] - track[idx - 1]["x"]
        dy = track[idx]["y"] - track[idx - 1]["y"]
        speeds.append(math.hypot(dx, dy))
    return speeds


def _coord_scale(track: List[dict]) -> float:
    if not track:
        return 1.0
    xs = [_safe_float(point.get("x"), 0.0) for point in track]
    ys = [_safe_float(point.get("y"), 0.0) for point in track]
    return max(1e-6, max(xs) - min(xs), max(ys) - min(ys))


def _smooth_speeds(speeds: List[float]) -> List[float]:
    if not speeds:
        return []
    smoothed: List[float] = []
    for idx in range(len(speeds)):
        start = max(0, idx - 1)
        end = min(len(speeds), idx + 2)
        smoothed.append(_mean(speeds[start:end]))
    return smoothed


def _load_body_artifact(job_id: str, body_path: Optional[str]) -> Optional[Dict[str, Any]]:
    candidates: List[Path] = []
    if body_path:
        candidates.append(Path(body_path))
    data_dir = Path(os.getenv("DATA_DIR", "/home/ray/data"))
    candidates.append(data_dir / "body" / f"{job_id}.json")
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _keypoint_xyc(frame: dict, name: str) -> Optional[Tuple[float, float, float]]:
    keypoints = frame.get("keypoints")
    if not isinstance(keypoints, dict):
        return None
    raw = keypoints.get(name)
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    conf = _safe_float(raw[2], 0.0) if len(raw) >= 3 else 1.0
    return _safe_float(raw[0], 0.0), _safe_float(raw[1], 0.0), conf


def _wrist_track_from_body(body: Optional[Dict[str, Any]]) -> List[dict]:
    frames = body.get("frames") if isinstance(body, dict) else None
    if not isinstance(frames, list):
        return []
    track: List[dict] = []
    previous: Optional[dict] = None
    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        left = _keypoint_xyc(frame, "left_wrist")
        right = _keypoint_xyc(frame, "right_wrist")
        candidates: List[dict] = []
        for name, point in (("left_wrist", left), ("right_wrist", right)):
            if point is None:
                continue
            x, y, conf = point
            if conf < 0.02:
                continue
            candidates.append({"x": x, "y": y, "conf": conf, "wristSource": name})
        wrist_gap = math.hypot(left[0] - right[0], left[1] - right[1]) if left and right else float("inf")
        if left and right and left[2] >= 0.02 and right[2] >= 0.02 and wrist_gap <= 0.16:
            total_conf = max(1e-6, left[2] + right[2])
            candidates.append(
                {
                    "x": (left[0] * left[2] + right[0] * right[2]) / total_conf,
                    "y": (left[1] * left[2] + right[1] * right[2]) / total_conf,
                    "conf": _mean([left[2], right[2]]),
                    "wristSource": "weighted_midpoint",
                }
            )
        if not candidates:
            continue
        if previous:
            max_jump = 0.32

            def score(candidate: dict) -> float:
                jump = math.hypot(candidate["x"] - previous["x"], candidate["y"] - previous["y"])
                continuity = 1.0 - min(1.0, jump / max_jump)
                height_bias = 1.0 - _clamp(candidate["y"])
                conf_score = _clamp(candidate["conf"])
                return continuity * 0.52 + conf_score * 0.28 + height_bias * 0.2

            selected = max(candidates, key=score)
        else:
            selected = max(candidates, key=lambda candidate: _clamp(candidate["conf"]) * 0.68 + (1.0 - _clamp(candidate["y"])) * 0.32)
        point = {
            "x": selected["x"],
            "y": selected["y"],
            "t": _safe_float(frame.get("timeMs"), idx * 33.33),
            "frame": _safe_int(frame.get("frameIndex"), idx),
            "conf": selected["conf"],
            "source": "pose_wrist",
            "wristSource": selected["wristSource"],
        }
        track.append(point)
        previous = point
    track.sort(key=lambda point: (_safe_int(point.get("frame"), 0), _safe_float(point.get("t"), 0.0)))
    return track


def _nearest_track_index_by_time(track: List[dict], time_ms: float) -> Optional[int]:
    if not track:
        return None
    return min(
        range(len(track)),
        key=lambda idx: abs(_safe_float(track[idx].get("t"), 0.0) - time_ms),
    )


def _find_address_index(track: List[dict], speeds: List[float]) -> Optional[int]:
    if not track or not speeds:
        return None
    search_end = max(1, min(len(track) - 1, len(speeds) - 1, max(3, len(track) // 5)))
    speed_ref = max(_median(speeds), _mean(speeds) * 0.35, 1e-6)
    best_idx = 0
    best_score = float("-inf")
    for idx in range(0, search_end + 1):
        speed_score = 1.0 - min(1.0, speeds[idx] / (speed_ref * 2.5))
        early_score = 1.0 - idx / max(1, search_end)
        conf_score = _safe_float(track[idx].get("conf"), 0.0)
        score = speed_score * 0.55 + early_score * 0.25 + conf_score * 0.2
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _find_top_after_address(track: List[dict], address_idx: int) -> Optional[int]:
    if len(track) < 3:
        return None
    search_start = min(len(track) - 2, max(0, address_idx + 1))
    search_end = max(search_start + 1, min(len(track) - 2, int(len(track) * 0.68)))
    address = track[address_idx]
    scale = _coord_scale(track)
    best_idx = search_start
    best_score = float("-inf")
    for idx in range(search_start, search_end + 1):
        point = track[idx]
        height_gain = max(0.0, address["y"] - point["y"]) / scale
        displacement = math.hypot(point["x"] - address["x"], point["y"] - address["y"]) / scale
        time_score = 1.0 - abs((idx / max(1, len(track) - 1)) - 0.42) / 0.35
        score = height_gain * 0.42 + displacement * 0.42 + max(0.0, time_score) * 0.16
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _find_top_from_wrist_track(wrist_track: List[dict], address_time_ms: float) -> Optional[dict]:
    if len(wrist_track) < 3:
        return None
    search = [
        point
        for point in wrist_track
        if _safe_float(point.get("t"), 0.0) >= address_time_ms
    ]
    if len(search) < 3:
        search = wrist_track
    address = search[0]
    last_t = _safe_float(search[-1].get("t"), address_time_ms)
    duration = max(1.0, last_t - address_time_ms)
    window_start_ms = address_time_ms + min(150.0, duration * 0.12)
    window_end_ms = address_time_ms + min(900.0, duration * 0.62)
    candidates = [
        point
        for point in search
        if window_start_ms <= _safe_float(point.get("t"), 0.0) <= window_end_ms
    ]
    if len(candidates) < 3:
        fallback_end = max(2, min(len(search) - 1, int(len(search) * 0.45)))
        candidates = search[1 : fallback_end + 1]
    if not candidates:
        return None

    xs = [_safe_float(point.get("x"), 0.0) for point in wrist_track]
    ys = [_safe_float(point.get("y"), 0.0) for point in wrist_track]
    scale = max(1e-6, max(xs) - min(xs), max(ys) - min(ys))
    target_ms = address_time_ms + min(520.0, duration * 0.36)
    candidate_metrics: List[Dict[str, Any]] = []
    for point in candidates:
        point_t = _safe_float(point.get("t"), 0.0)
        point_y = _safe_float(point.get("y"), 0.0)
        future = [
            future_point
            for future_point in search
            if point_t + 80.0 <= _safe_float(future_point.get("t"), 0.0) <= point_t + 320.0
        ]
        future_drop = 0.0
        if future:
            future_drop = max(
                0.0,
                max(_safe_float(future_point.get("y"), 0.0) for future_point in future) - point_y,
            ) / scale
        height_gain = max(0.0, _safe_float(address.get("y"), 0.0) - _safe_float(point.get("y"), 0.0)) / scale
        displacement = math.hypot(
            _safe_float(point.get("x"), 0.0) - _safe_float(address.get("x"), 0.0),
            _safe_float(point.get("y"), 0.0) - _safe_float(address.get("y"), 0.0),
        ) / scale
        time_score = 1.0 - min(1.0, abs(point_t - target_ms) / 300.0)
        conf_score = _safe_float(point.get("conf"), 0.0)
        candidate_metrics.append(
            {
                "point": point,
                "t": point_t,
                "y": point_y,
                "heightGain": height_gain,
                "displacement": displacement,
                "futureDrop": future_drop,
                "timeScore": max(0.0, time_score),
                "confScore": conf_score,
            }
        )

    max_height_gain = max((metric["heightGain"] for metric in candidate_metrics), default=0.0)
    min_peak_gain = max(0.08, max_height_gain * 0.35)
    best_rise_metric: Optional[Dict[str, Any]] = None
    for metric in sorted(candidate_metrics, key=lambda item: _safe_float(item.get("t"), 0.0)):
        if metric["heightGain"] >= min_peak_gain:
            if best_rise_metric is None:
                best_rise_metric = metric
            else:
                metric_t = _safe_float(metric.get("t"), 0.0)
                best_t = _safe_float(best_rise_metric.get("t"), 0.0)
                near_peak_plateau = (
                    metric_t <= best_t + 120.0
                    and metric["heightGain"] >= best_rise_metric["heightGain"] * 0.94
                )
                if metric["heightGain"] > best_rise_metric["heightGain"] or near_peak_plateau:
                    best_rise_metric = metric
        if best_rise_metric is None:
            continue
        metric_t = _safe_float(metric.get("t"), 0.0)
        best_t = _safe_float(best_rise_metric.get("t"), 0.0)
        if metric_t <= best_t + 45.0:
            continue
        drop_from_peak = (
            max(
                0.0,
                _safe_float(metric.get("y"), 0.0) - _safe_float(best_rise_metric.get("y"), 0.0),
            )
            / scale
        )
        if drop_from_peak >= max(0.10, best_rise_metric["heightGain"] * 0.28):
            return best_rise_metric["point"]

    transition_metrics = [
        metric
        for metric in candidate_metrics
        if metric["heightGain"] >= min_peak_gain
        and metric["futureDrop"] >= max(0.16, metric["heightGain"] * 0.35)
    ]
    if transition_metrics:
        first_transition_t = _safe_float(transition_metrics[0]["point"].get("t"), 0.0)
        early_transition_metrics = [
            metric
            for metric in transition_metrics
            if _safe_float(metric["point"].get("t"), 0.0) <= first_transition_t + 260.0
        ]
        candidate_metrics = early_transition_metrics or transition_metrics

    best_metric = max(
        candidate_metrics,
        key=lambda metric: (
            metric["heightGain"] * 0.42
            + metric["futureDrop"] * 0.32
            + metric["displacement"] * 0.12
            + metric["timeScore"] * 0.08
            + metric["confScore"] * 0.06
        ),
    )
    best = best_metric["point"]
    return best


def _find_impact_from_wrist_track(
    wrist_track: List[dict],
    top: dict,
    club_head_track: Optional[List[dict]] = None,
) -> Optional[dict]:
    if len(wrist_track) < 3:
        return None
    top_t = _safe_float(top.get("t"), 0.0)
    window_start_ms = top_t + 110.0
    window_end_ms = top_t + 300.0
    candidates = [
        point
        for point in wrist_track
        if window_start_ms <= _safe_float(point.get("t"), 0.0) <= window_end_ms
    ]
    if len(candidates) < 2:
        return None

    xs = [_safe_float(point.get("x"), 0.0) for point in wrist_track]
    ys = [_safe_float(point.get("y"), 0.0) for point in wrist_track]
    scale = max(1e-6, max(xs) - min(xs), max(ys) - min(ys))
    target_ms = top_t + 180.0
    candidate_metrics: List[Dict[str, Any]] = []
    for point in candidates:
        descent = max(0.0, _safe_float(point.get("y"), 0.0) - _safe_float(top.get("y"), 0.0)) / scale
        travel = math.hypot(
            _safe_float(point.get("x"), 0.0) - _safe_float(top.get("x"), 0.0),
            _safe_float(point.get("y"), 0.0) - _safe_float(top.get("y"), 0.0),
        ) / scale
        candidate_metrics.append({"point": point, "descent": descent, "travel": travel})

    max_descent = max((metric["descent"] for metric in candidate_metrics), default=0.0)
    for metric in candidate_metrics:
        if (
            metric["descent"] >= max(0.12, max_descent * 0.75)
            and metric["travel"] >= max(0.12, max_descent * 0.55)
        ):
            return metric["point"]

    best = candidates[0]
    best_score = float("-inf")
    for point in candidates:
        point_t = _safe_float(point.get("t"), 0.0)
        descent = max(0.0, _safe_float(point.get("y"), 0.0) - _safe_float(top.get("y"), 0.0)) / scale
        travel = math.hypot(
            _safe_float(point.get("x"), 0.0) - _safe_float(top.get("x"), 0.0),
            _safe_float(point.get("y"), 0.0) - _safe_float(top.get("y"), 0.0),
        ) / scale
        head = _nearest_track_point(club_head_track or [], point_t, 70.0)
        head_score = _safe_float(head.get("conf"), 0.0) if head else 0.0
        time_score = 1.0 - min(1.0, abs(point_t - target_ms) / 150.0)
        score = descent * 0.34 + travel * 0.22 + max(0.0, time_score) * 0.34 + head_score * 0.1
        if score > best_score:
            best_score = score
            best = point
    return best


def _find_finish_from_wrist_track(wrist_track: List[dict], impact: dict) -> Optional[dict]:
    if len(wrist_track) < 3:
        return None
    impact_t = _safe_float(impact.get("t"), 0.0)
    window_start_ms = impact_t + 180.0
    window_end_ms = impact_t + 470.0
    candidates = [
        point
        for point in wrist_track
        if window_start_ms <= _safe_float(point.get("t"), 0.0) <= window_end_ms
    ]
    if len(candidates) < 2:
        return None

    xs = [_safe_float(point.get("x"), 0.0) for point in wrist_track]
    ys = [_safe_float(point.get("y"), 0.0) for point in wrist_track]
    scale = max(1e-6, max(xs) - min(xs), max(ys) - min(ys))
    target_ms = impact_t + 340.0
    best = candidates[0]
    best_score = float("-inf")
    for point in candidates:
        point_t = _safe_float(point.get("t"), 0.0)
        travel = math.hypot(
            _safe_float(point.get("x"), 0.0) - _safe_float(impact.get("x"), 0.0),
            _safe_float(point.get("y"), 0.0) - _safe_float(impact.get("y"), 0.0),
        ) / scale
        height_gain = max(0.0, _safe_float(impact.get("y"), 0.0) - _safe_float(point.get("y"), 0.0)) / scale
        time_score = 1.0 - min(1.0, abs(point_t - target_ms) / 180.0)
        score = travel * 0.36 + height_gain * 0.28 + max(0.0, time_score) * 0.36
        if score > best_score:
            best_score = score
            best = point
    return best


def _nearest_track_point(track: List[dict], time_ms: float, max_delta_ms: float) -> Optional[dict]:
    idx = _nearest_track_index_by_time(track, time_ms)
    if idx is None:
        return None
    point = track[idx]
    return point if abs(_safe_float(point.get("t"), 0.0) - time_ms) <= max_delta_ms else None


def _wrist_motion_between(wrist_track: List[dict], start_ms: float, end_ms: float) -> Optional[Dict[str, float]]:
    start = _nearest_track_point(wrist_track, start_ms, 85.0)
    end = _nearest_track_point(wrist_track, end_ms, 85.0)
    if not start or not end:
        return None
    dx = _safe_float(end.get("x"), 0.0) - _safe_float(start.get("x"), 0.0)
    dy = _safe_float(end.get("y"), 0.0) - _safe_float(start.get("y"), 0.0)
    travel = math.hypot(dx, dy)
    rise = max(0.0, -dy)
    drop = max(0.0, dy)
    return {
        "travelRatio": travel / WRIST_PERSON_HEIGHT_NORM,
        "riseRatio": rise / WRIST_PERSON_HEIGHT_NORM,
        "dropRatio": drop / WRIST_PERSON_HEIGHT_NORM,
    }


def _wrist_events_are_usable(
    wrist_track: List[dict],
    address_time_ms: float,
    wrist_top: Optional[dict],
    wrist_impact: Optional[dict],
) -> bool:
    if not wrist_top or not wrist_impact:
        return False
    top_motion = _wrist_motion_between(wrist_track, address_time_ms, _safe_float(wrist_top.get("t"), 0.0))
    down_motion = _wrist_motion_between(
        wrist_track,
        _safe_float(wrist_top.get("t"), 0.0),
        _safe_float(wrist_impact.get("t"), 0.0),
    )
    if not top_motion or not down_motion:
        return False
    top_has_motion = (
        top_motion["travelRatio"] >= WRIST_MIN_EVENT_TRAVEL_RATIO
        or top_motion["riseRatio"] >= WRIST_MIN_EVENT_RISE_RATIO
    )
    down_has_motion = down_motion["travelRatio"] >= WRIST_MIN_EVENT_TRAVEL_RATIO
    return top_has_motion and down_has_motion


def _find_impact_after_top(
    track: List[dict],
    speeds: List[float],
    address_idx: int,
    top_idx: int,
    club_head_track: Optional[List[dict]] = None,
) -> Optional[int]:
    if len(track) < 3:
        return None
    top_t = _safe_float(track[top_idx].get("t"), 0.0)
    window_start_ms = top_t + 110.0
    window_end_ms = top_t + 280.0
    candidates = [
        idx
        for idx in range(top_idx + 1, len(track))
        if window_start_ms <= _safe_float(track[idx].get("t"), 0.0) <= window_end_ms
    ]
    if not candidates:
        search_start = min(len(track) - 1, max(top_idx + 1, int(len(track) * 0.45)))
        search_end = max(search_start, min(len(track) - 1, int(len(track) * 0.72)))
        candidates = list(range(search_start, search_end + 1))
    address = track[address_idx]
    top = track[top_idx]
    scale = _coord_scale(track)
    max_speed = max(speeds) if speeds else 1.0
    target_ms = top_t + 175.0
    best_idx = candidates[0]
    best_score = float("-inf")
    for idx in candidates:
        point = track[idx]
        head = _nearest_track_point(club_head_track or [], _safe_float(point.get("t"), 0.0), 65.0)
        speed_score = speeds[idx] / max(max_speed, 1e-6)
        address_y_score = 1.0 - min(1.0, abs(point["y"] - address["y"]) / scale)
        address_dist_score = 1.0 - min(1.0, math.hypot(point["x"] - address["x"], point["y"] - address["y"]) / (scale * 1.35))
        descent = max(0.0, point["y"] - top["y"]) / scale
        head_score = _safe_float(head.get("conf"), 0.0) if head else 0.0
        time_score = 1.0 - min(1.0, abs(_safe_float(point.get("t"), 0.0) - target_ms) / 150.0)
        score = (
            speed_score * 0.24
            + address_y_score * 0.19
            + address_dist_score * 0.15
            + descent * 0.13
            + max(0.0, time_score) * 0.23
            + head_score * 0.06
        )
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _find_finish_after_impact(track: List[dict], speeds: List[float], impact_idx: int) -> Optional[int]:
    if len(track) < 2:
        return None
    impact_t = _safe_float(track[impact_idx].get("t"), 0.0)
    window_start_ms = impact_t + 120.0
    window_end_ms = impact_t + 340.0
    candidates = [
        idx
        for idx in range(impact_idx + 1, len(track))
        if window_start_ms <= _safe_float(track[idx].get("t"), 0.0) <= window_end_ms
    ]
    if not candidates:
        search_start = min(len(track) - 1, max(impact_idx + 1, int(len(track) * 0.58)))
        search_end = max(search_start, min(len(track) - 1, int(len(track) * 0.82)))
        candidates = list(range(search_start, search_end + 1))
    address = track[0]
    impact = track[impact_idx]
    scale = _coord_scale(track)
    target_ms = impact_t + 230.0
    best_idx = candidates[0]
    best_score = float("-inf")
    for idx in candidates:
        point = track[idx]
        displacement = math.hypot(point["x"] - address["x"], point["y"] - address["y"]) / scale
        post_impact_move = math.hypot(point["x"] - impact["x"], point["y"] - impact["y"]) / scale
        height_gain = max(0.0, impact["y"] - point["y"]) / scale
        time_score = 1.0 - min(1.0, abs(_safe_float(point.get("t"), 0.0) - target_ms) / 220.0)
        score = displacement * 0.34 + post_impact_move * 0.26 + height_gain * 0.22 + max(0.0, time_score) * 0.18
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _segment_events(
    track: List[dict],
    speeds: List[float],
    wrist_track: Optional[List[dict]] = None,
    club_head_track: Optional[List[dict]] = None,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], str, Optional[dict]]:
    if not track or not speeds:
        return None, None, None, None, "empty", None
    smoothed = _smooth_speeds(speeds)
    address_idx = _find_address_index(track, smoothed)
    if address_idx is None:
        return None, None, None, None, "empty", None
    address_time_ms = _safe_float(track[address_idx].get("t"), 0.0)
    wrist_top = _find_top_from_wrist_track(wrist_track or [], address_time_ms)
    wrist_impact = _find_impact_from_wrist_track(wrist_track or [], wrist_top, club_head_track) if wrist_top else None
    wrist_finish = _find_finish_from_wrist_track(wrist_track or [], wrist_impact) if wrist_impact else None
    wrist_usable = _wrist_events_are_usable(wrist_track or [], address_time_ms, wrist_top, wrist_impact)
    if not EVENT_FALLBACK_ENABLED and wrist_top:
        top_idx = _nearest_track_index_by_time(track, _safe_float(wrist_top.get("t"), 0.0))
        impact_idx = _nearest_track_index_by_time(track, _safe_float(wrist_impact.get("t"), 0.0)) if wrist_impact else None
        finish_idx = _nearest_track_index_by_time(track, _safe_float(wrist_finish.get("t"), 0.0)) if wrist_finish else None
        event_source = "pose_wrist_no_fallback" if wrist_usable else "pose_wrist_unvalidated_no_fallback"
        return address_idx, top_idx, impact_idx, finish_idx, event_source, wrist_top
    top_idx = _nearest_track_index_by_time(track, _safe_float(wrist_top.get("t"), 0.0)) if wrist_usable and wrist_top else None
    impact_idx = _nearest_track_index_by_time(track, _safe_float(wrist_impact.get("t"), 0.0)) if wrist_usable and wrist_impact else None
    finish_idx = _nearest_track_index_by_time(track, _safe_float(wrist_finish.get("t"), 0.0)) if wrist_usable and wrist_finish else None
    event_source = "pose_wrist_fusion" if top_idx is not None and impact_idx is not None else "trajectory_score"
    if wrist_top and wrist_impact and not wrist_usable:
        event_source = "trajectory_score_wrist_rejected"
    if top_idx is None and EVENT_FALLBACK_ENABLED:
        top_idx = _find_top_after_address(track, address_idx)
    if top_idx is None:
        return address_idx, None, None, None, "no_top", wrist_top
    if impact_idx is None and EVENT_FALLBACK_ENABLED:
        impact_idx = _find_impact_after_top(track, smoothed, address_idx, top_idx, club_head_track)
    if impact_idx is None:
        return address_idx, top_idx, None, None, "no_impact", wrist_top
    if finish_idx is None and EVENT_FALLBACK_ENABLED:
        finish_idx = _find_finish_after_impact(track, smoothed, impact_idx)
    return address_idx, top_idx, impact_idx, finish_idx, event_source, wrist_top


def _find_stable_index(speeds: List[float], start: int, direction: int) -> Optional[int]:
    if not speeds:
        return None
    median_speed = _median(speeds)
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


def _find_top(track: List[dict], impact_idx: int) -> Optional[int]:
    if impact_idx <= 1:
        return None
    pre_impact = track[:impact_idx]
    if len(pre_impact) < 2:
        return None

    ys = [p["y"] for p in pre_impact]
    xs = [p["x"] for p in pre_impact]
    if max(ys) - min(ys) >= 12.0:
        return _argmin(ys)

    sample = pre_impact[: max(2, min(6, len(pre_impact)))]
    dx = sample[-1]["x"] - sample[0]["x"]
    return _argmax(xs) if dx >= 0 else _argmin(xs)


def _tempo(address_ms: int, top_ms: int, impact_ms: int) -> Tuple[int, int, float]:
    backswing = max(0, top_ms - address_ms)
    downswing = max(1, impact_ms - top_ms)
    ratio = round(backswing / float(downswing), 2)
    return backswing, downswing, ratio


def _impact_stability(track: List[dict], impact_idx: int) -> Tuple[str, float]:
    start = max(0, impact_idx - 3)
    end = min(len(track), impact_idx + 4)
    window = track[start:end]
    if len(window) < 2:
        return "unstable", 0.0
    xs = [p["x"] for p in window]
    ys = [p["y"] for p in window]
    deviation = math.hypot(_std(xs), _std(ys))
    sizes = [(p["w"] + p["h"]) / 2.0 for p in window]
    scale = _mean(sizes) if sizes else 1.0
    score = _clamp(1.0 - deviation / (scale + 1e-6))
    label = "stable" if score >= 0.6 else "unstable"
    return label, round(score, 2)


def _nearest_track_point(track: List[dict], target_ms: float, max_delta_ms: Optional[float] = None) -> Optional[dict]:
    if not track:
        return None
    point = min(track, key=lambda p: abs(p["t"] - target_ms))
    if max_delta_ms is not None and abs(point["t"] - target_ms) > max_delta_ms:
        return None
    return point


def _shaft_samples(head_track: List[dict], handle_track: List[dict], fps: int) -> List[dict]:
    if not head_track or not handle_track:
        return []
    max_delta = max(60.0, 3.0 * (1000.0 / max(1, fps)))
    samples: List[dict] = []
    for head in head_track:
        handle = _nearest_track_point(handle_track, head["t"], max_delta)
        if not handle:
            continue
        dx = head["x"] - handle["x"]
        dy = head["y"] - handle["y"]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        angle = math.degrees(math.atan2(abs(dy), abs(dx) + 1e-6))
        samples.append(
            {
                "t": head["t"],
                "frame": head["frame"],
                "angleDeg": angle,
                "length": length,
                "confidence": min(head["conf"], handle["conf"]),
            }
        )
    return samples


def _club_box_shaft_samples(club_track: List[dict]) -> List[dict]:
    samples: List[dict] = []
    for point in club_track:
        w = _safe_float(point.get("w"), 0.0)
        h = _safe_float(point.get("h"), 0.0)
        if w <= 1e-6 and h <= 1e-6:
            continue

        angle = math.degrees(math.atan2(abs(h), abs(w) + 1e-6))
        # A club bbox is only a coarse proxy for the shaft line, especially when
        # motion blur or tight crops shrink the box. Keep confidence conservative.
        confidence = _clamp(_safe_float(point.get("conf"), 0.0) * 0.45)
        samples.append(
            {
                "t": point["t"],
                "frame": point["frame"],
                "angleDeg": angle,
                "length": math.hypot(w, h),
                "confidence": confidence,
                "source": "club_box_proxy",
            }
        )
    return samples


def _shaft_plane(
    samples: List[dict],
    address_ms: int,
    top_ms: int,
    impact_ms: int,
) -> Dict[str, object]:
    if not samples:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "angleDeg": None,
            "addressAngleDeg": None,
            "comment": "club_head와 club_handle을 동시에 충분히 추적하지 못했습니다.",
        }

    target_ms = top_ms + (impact_ms - top_ms) * 0.55
    downswing_sample = _nearest_track_point(samples, target_ms)
    address_sample = _nearest_track_point(samples, address_ms)
    angle = downswing_sample["angleDeg"] if downswing_sample else samples[-1]["angleDeg"]
    address_angle = address_sample["angleDeg"] if address_sample else None
    source = str((downswing_sample or samples[-1]).get("source") or "head_handle")

    if angle >= 62:
        label = "steep"
        comment = "다운스윙 샤프트가 2D 기준으로 세워진 편입니다. 전환 이후 손이 앞서가며 클럽이 늦게 내려오는지 확인하세요."
    elif angle <= 38:
        label = "flat"
        comment = "다운스윙 샤프트가 2D 기준으로 눕는 편입니다. 클럽이 몸 뒤에 남아 임팩트 타이밍이 늦어질 수 있습니다."
    else:
        label = "neutral"
        comment = "다운스윙 샤프트 각도는 단일 카메라 2D 기준에서 중립 범위입니다."

    if source == "club_box_proxy":
        comment = f"{comment} club_handle 검출이 없어 club bbox 장축으로 근사한 값입니다."

    avg_conf = _mean([s["confidence"] for s in samples])
    confidence = _clamp(avg_conf * min(1.0, len(samples) / 8.0))
    return {
        "label": label,
        "confidence": round(confidence, 2),
        "angleDeg": round(angle, 1),
        "addressAngleDeg": round(address_angle, 1) if address_angle is not None else None,
        "sampleCount": len(samples),
        "source": source,
        "comment": comment,
    }


def _track_uses_normalized_coords(track: List[dict]) -> bool:
    if not track:
        return False
    samples = track[: min(12, len(track))]
    return all(
        -0.05 <= _safe_float(point.get("x"), 0.0) <= 1.05
        and -0.05 <= _safe_float(point.get("y"), 0.0) <= 1.05
        and 0.0 <= _safe_float(point.get("w"), 0.0) <= 1.05
        and 0.0 <= _safe_float(point.get("h"), 0.0) <= 1.05
        for point in samples
    )


def _person_height(
    person_track: List[dict],
    target_ms: int,
    image_height: Optional[float],
    normalized_coords: bool,
) -> Tuple[Optional[dict], float]:
    person = _nearest_track_point(person_track, target_ms, None)
    if person:
        min_height = 0.01 if normalized_coords else 1.0
        return person, max(min_height, person["h"])
    if normalized_coords:
        return None, 0.72
    if image_height:
        return None, max(1.0, image_height * 0.72)
    return None, 1.0


def _backswing_metric(
    motion_track: List[dict],
    person_track: List[dict],
    address_idx: int,
    top_idx: int,
    image_height: Optional[float],
    wrist_track: Optional[List[dict]] = None,
    force_wrist: bool = False,
) -> Dict[str, object]:
    address = motion_track[address_idx]
    top = motion_track[top_idx]
    normalized_coords = _track_uses_normalized_coords(motion_track)
    person, person_h = _person_height(person_track, _safe_int(top.get("t"), 0), image_height, normalized_coords)

    pre_top = motion_track[address_idx : top_idx + 1] or [address, top]
    highest_y = min(p["y"] for p in pre_top)
    vertical_travel = max(0.0, address["y"] - highest_y)
    travel_ratio = vertical_travel / person_h
    source = "club_motion"

    wrist_address = _nearest_track_point(wrist_track or [], _safe_float(address.get("t"), 0.0), 80.0)
    wrist_top = _nearest_track_point(wrist_track or [], _safe_float(top.get("t"), 0.0), 80.0)
    if wrist_address and wrist_top:
        wrist_motion = _wrist_motion_between(
            wrist_track or [],
            _safe_float(address.get("t"), 0.0),
            _safe_float(top.get("t"), 0.0),
        )
        wrist_motion_usable = (
            wrist_motion
            and (
                force_wrist
                or wrist_motion["travelRatio"] >= WRIST_MIN_EVENT_TRAVEL_RATIO
                or wrist_motion["riseRatio"] >= WRIST_MIN_EVENT_RISE_RATIO
            )
        )
        if wrist_motion_usable:
            travel_ratio = max(wrist_motion["riseRatio"], wrist_motion["travelRatio"] * 0.75)
            source = "pose_wrist"

    top_height_ratio = None
    if person:
        person_top = person["y"] - person["h"] / 2.0
        top_height_ratio = (top["y"] - person_top) / person_h

    target_ratio = 0.22 if source == "pose_wrist" else 0.42
    short_threshold = 0.08 if source == "pose_wrist" else 0.18
    score = _clamp(travel_ratio / target_ratio)
    if travel_ratio < short_threshold:
        label = "short"
        comment = (
            "백스윙 탑까지 손목 이동량이 작습니다. 어깨 회전과 손 위치가 충분히 올라가는지 확인하세요."
            if source == "pose_wrist"
            else "백스윙 탑까지 클럽 이동량이 작습니다. 어깨 회전과 손 위치가 충분히 올라가는지 확인하세요."
        )
    elif top_height_ratio is not None and top_height_ratio > 0.62:
        label = "low_top"
        comment = "백스윙 탑 위치가 낮게 잡힙니다. 전신 프레임에서 왼팔과 어깨 회전 폭을 함께 확인하세요."
    elif top_height_ratio is not None and top_height_ratio < -0.08:
        label = "high_top"
        comment = "백스윙 탑이 매우 높게 잡힙니다. 오버스윙이거나 카메라 각도 영향일 수 있습니다."
    else:
        label = "adequate"
        comment = (
            "백스윙 크기는 pose 손목 추적 기준에서 적정 범위입니다."
            if source == "pose_wrist"
            else "백스윙 크기는 service7 클럽 추적 기준에서 적정 범위입니다."
        )

    return {
        "label": label,
        "score": round(score, 2),
        "clubTravelRatio": round(travel_ratio, 2),
        "handTravelRatio": round(travel_ratio, 2) if source == "pose_wrist" else None,
        "topHeightRatio": round(top_height_ratio, 2) if top_height_ratio is not None else None,
        "source": source,
        "comment": comment,
    }


def _readiness_metric(frames: List[dict]) -> Dict[str, object]:
    ready_count, ready_conf = _label_conf_stats(frames, READY_LABELS)
    not_ready_count, not_ready_conf = _label_conf_stats(frames, NOT_READY_LABELS)
    if ready_count == 0 and not_ready_count == 0:
        return {"label": "unknown", "confidence": 0.0, "readyFrames": 0, "notReadyFrames": 0}

    ready_score = ready_count * max(ready_conf, 0.01)
    not_ready_score = not_ready_count * max(not_ready_conf, 0.01)
    if ready_score >= not_ready_score:
        label = "ready"
        confidence = ready_conf
    else:
        label = "not_ready"
        confidence = not_ready_conf
    return {
        "label": label,
        "confidence": round(_clamp(confidence), 2),
        "readyFrames": ready_count,
        "notReadyFrames": not_ready_count,
    }


def _tracking_quality(
    frames: List[dict],
    club_head: List[dict],
    handle: List[dict],
    club: List[dict],
    ball: List[dict],
    person: List[dict],
) -> Dict[str, object]:
    total = max(1, len(frames))

    def coverage(track: List[dict]) -> float:
        return _clamp(len(track) / float(total))

    def avg_conf(track: List[dict]) -> float:
        return _mean([p["conf"] for p in track])

    quality = (
        coverage(club_head) * 0.45
        + coverage(handle) * 0.2
        + coverage(club) * 0.1
        + coverage(ball) * 0.15
        + coverage(person) * 0.1
    )
    if quality >= 0.65:
        label = "good"
    elif quality >= 0.35:
        label = "fair"
    else:
        label = "weak"

    return {
        "label": label,
        "score": round(quality, 2),
        "frames": total,
        "clubHeadFrames": len(club_head),
        "clubHandleFrames": len(handle),
        "clubFrames": len(club),
        "ballFrames": len(ball),
        "personFrames": len(person),
        "clubHeadConfidence": round(avg_conf(club_head), 2),
        "clubHandleConfidence": round(avg_conf(handle), 2),
        "clubConfidence": round(avg_conf(club), 2),
        "ballConfidence": round(avg_conf(ball), 2),
        "personConfidence": round(avg_conf(person), 2),
    }


def _ball_metric(ball_track: List[dict], impact_ms: int, width: Optional[float], height: Optional[float]) -> Dict[str, object]:
    if len(ball_track) < 2:
        return {
            "launchDirection": "unknown",
            "launchAngle": None,
            "speedRelative": "unknown",
            "confidence": round(_mean([p["conf"] for p in ball_track]), 2),
        }

    after = [p for p in ball_track if p["t"] >= impact_ms - 50.0]
    points = after if len(after) >= 2 else ball_track
    first = points[0]
    last = points[-1]
    dx = last["x"] - first["x"]
    dy = last["y"] - first["y"]
    image_diag = math.hypot(width or 0.0, height or 0.0) or 1.0
    move_ratio = math.hypot(dx, dy) / image_diag

    x_threshold = max(3.0, (width or 640.0) * 0.01)
    if dx > x_threshold:
        direction = "right"
    elif dx < -x_threshold:
        direction = "left"
    else:
        direction = "center"

    angle = math.degrees(math.atan2(-dy, abs(dx) + 1e-6))
    if move_ratio >= 0.12:
        speed = "fast"
    elif move_ratio >= 0.04:
        speed = "medium"
    else:
        speed = "slow"

    return {
        "launchDirection": direction,
        "launchAngle": round(angle, 1),
        "speedRelative": speed,
        "confidence": round(_mean([p["conf"] for p in points]), 2),
    }


def _coach_comments(
    tempo: Dict[str, object],
    shaft_plane: Dict[str, object],
    backswing: Dict[str, object],
    impact_stability: Dict[str, object],
    readiness: Dict[str, object],
    tracking: Dict[str, object],
) -> List[str]:
    comments: List[str] = []

    ratio = _safe_float(tempo.get("ratio"), 0.0)
    if ratio > 0:
        if ratio < 2.0:
            comments.append(f"템포가 {ratio}:1로 빠른 편입니다. 백스윙 탑에서 전환을 조금 더 분리해 보세요.")
        elif ratio > 4.0:
            comments.append(f"템포가 {ratio}:1로 긴 편입니다. 백스윙 길이보다 다운스윙 리듬을 일정하게 만드는 쪽을 우선하세요.")
        else:
            comments.append(f"템포는 {ratio}:1로 전신 스윙 분석 기준에서 사용할 수 있는 범위입니다.")

    for metric in (shaft_plane, backswing):
        comment = metric.get("comment")
        if isinstance(comment, str) and comment:
            comments.append(comment)

    if impact_stability.get("label") == "unstable":
        comments.append("임팩트 구간 클럽 헤드 위치 변동이 큽니다. 공 주변 3~4프레임에서 손목 릴리스 타이밍을 확인하세요.")
    elif impact_stability.get("label") == "stable":
        comments.append("임팩트 구간 클럽 헤드 추적은 비교적 안정적입니다.")

    if readiness.get("label") == "not_ready":
        comments.append("어드레스 준비 상태가 불안정하게 감지되었습니다. 분석 시작 프레임을 어드레스 이후로 맞추는 편이 좋습니다.")

    if tracking.get("personFrames", 0) == 0:
        comments.append("person 검출이 없어 전신 기준 보정이 약합니다. 몸 전체가 화면에 크게 보이도록 거리와 구도를 먼저 조정하세요.")

    if tracking.get("ballFrames", 0) == 0:
        comments.append("golf_ball 검출이 없어 발사 방향/출발 조건 해석은 현재 신뢰할 수 없습니다.")

    if tracking.get("label") == "weak":
        comments.append("추적 품질이 낮습니다. 전신이 프레임 안에 들어오고 클럽 헤드가 배경과 분리되도록 조명과 거리부터 조정하세요.")

    return comments[:6]


def analyze_meta(meta: Dict[str, object], job_id: str, force: bool, body_path: Optional[str] = None) -> Dict[str, object]:
    frames = meta.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise CoachError("NOT_SWING", "meta frames missing")

    min_points = 4
    min_conf = 0.12
    fps = max(1, _safe_int(meta.get("fps"), 60))
    width = _safe_float(meta.get("width"), 0.0) or None
    height = _safe_float(meta.get("height"), 0.0) or None
    times_ms = _normalize_times(frames, fps)
    for idx, frame in enumerate(frames):
        frame["_t_ms"] = times_ms[idx] if times_ms[idx] is not None else idx * (1000.0 / fps)

    club_head_track = _select_best_track(frames, CLUBHEAD_LABELS)
    handle_track = _select_best_track(frames, HANDLE_LABELS)
    club_track = _select_best_track(frames, CLUB_LABELS)
    ball_track = _select_best_track(frames, BALL_LABELS)
    person_track = _select_best_track(frames, PERSON_LABELS)
    body_artifact = _load_body_artifact(job_id, body_path)
    wrist_track = _wrist_track_from_body(body_artifact)
    viewpoint = _normalize_viewpoint(meta.get("viewpoint") or (body_artifact or {}).get("viewpoint"))
    body_selector = select_body_events(body_artifact)
    use_body_selector = _body_selector_is_operational(viewpoint, body_selector)

    motion_track, motion_source = _choose_motion_track(club_head_track, handle_track, club_track)

    if len(motion_track) < min_points:
        if not force:
            raise CoachError("NOT_SWING", f"insufficient service7 club detections (source={motion_source}, motionFrames={len(motion_track)}, clubHeadFrames={len(club_head_track)}, clubHandleFrames={len(handle_track)}, clubFrames={len(club_track)}, personFrames={len(person_track)})")
    confs = [p["conf"] for p in motion_track]
    if confs and _mean(confs) < min_conf and not force:
        raise CoachError("NOT_SWING", f"service7 club detections too weak (source={motion_source}, motionFrames={len(motion_track)}, avgConf={round(_mean(confs), 3)})")

    speeds = _speeds(motion_track)
    if not speeds or max(speeds) <= 0:
        if not force:
            raise CoachError("NOT_SWING", f"insufficient motion (source={motion_source}, motionFrames={len(motion_track)})")

    address_idx, top_idx, impact_idx, finish_idx, event_source, wrist_top = _segment_events(
        motion_track,
        speeds,
        wrist_track,
        club_head_track,
    )
    body_selector_events = body_selector.get("events") if isinstance(body_selector.get("events"), dict) else {}
    if use_body_selector:
        address_idx = _nearest_track_index_by_time(motion_track, _safe_float(body_selector_events.get("addressMs"), 0.0))
        top_idx = _nearest_track_index_by_time(motion_track, _safe_float(body_selector_events.get("topMs"), 0.0))
        impact_idx = _nearest_track_index_by_time(motion_track, _safe_float(body_selector_events.get("impactMs"), 0.0))
        finish_idx = _nearest_track_index_by_time(motion_track, _safe_float(body_selector_events.get("finishMs"), 0.0))
        event_source = f"body_selector_down_the_line:{body_selector.get('method')}"
    wrist_impact = _find_impact_from_wrist_track(wrist_track, wrist_top, club_head_track) if wrist_top else None
    wrist_finish = _find_finish_from_wrist_track(wrist_track, wrist_impact) if wrist_impact else None
    if impact_idx is None and EVENT_FALLBACK_ENABLED:
        impact_idx = _argmax(speeds) if speeds else None
        event_source = "speed_fallback"
    if top_idx is None and impact_idx is not None and EVENT_FALLBACK_ENABLED:
        top_idx = _find_top(motion_track, impact_idx or 0)
    if address_idx is None and EVENT_FALLBACK_ENABLED:
        address_idx = _find_stable_index(speeds, 0, 1)
    if finish_idx is None and EVENT_FALLBACK_ENABLED:
        finish_idx = _find_stable_index(speeds, len(speeds) - 1, -1)

    if impact_idx is None or top_idx is None or address_idx is None or finish_idx is None:
        if not force:
            raise CoachError("NOT_SWING", f"event segmentation failed (source={motion_source}, motionFrames={len(motion_track)})")

    address_idx = address_idx if address_idx is not None else 0
    top_idx = top_idx if top_idx is not None else min(len(motion_track) - 1, address_idx + 1)
    impact_idx = impact_idx if impact_idx is not None else min(len(motion_track) - 1, top_idx + 1)
    finish_idx = finish_idx if finish_idx is not None else (len(motion_track) - 1)

    if address_idx >= top_idx:
        address_idx = 0
    if top_idx >= impact_idx:
        top_idx = max(0, impact_idx - 1)
    if finish_idx <= impact_idx:
        finish_idx = len(motion_track) - 1

    address_ms = _safe_int(motion_track[address_idx].get("t"), address_idx * (1000 // fps))
    top_ms = _safe_int(motion_track[top_idx].get("t"), top_idx * (1000 // fps))
    impact_ms = _safe_int(motion_track[impact_idx].get("t"), impact_idx * (1000 // fps))
    finish_ms = _safe_int(motion_track[finish_idx].get("t"), finish_idx * (1000 // fps))
    if use_body_selector:
        address_ms = _safe_int(body_selector_events.get("addressMs"), address_ms)
        top_ms = _safe_int(body_selector_events.get("topMs"), top_ms)
        impact_ms = _safe_int(body_selector_events.get("impactMs"), impact_ms)
        finish_ms = _safe_int(body_selector_events.get("finishMs"), finish_ms)

    dx = motion_track[impact_idx]["x"] - motion_track[top_idx]["x"]
    dy = motion_track[impact_idx]["y"] - motion_track[top_idx]["y"]
    swing_label = "inside-out" if dx >= 0 else "outside-in"
    swing_conf = round(_clamp(abs(dx) / (abs(dx) + abs(dy) + 1e-6)), 2)

    backswing_ms, downswing_ms, ratio = _tempo(address_ms, top_ms, impact_ms)
    tempo = {
        "backswingMs": backswing_ms,
        "downswingMs": downswing_ms,
        "ratio": ratio,
    }

    impact_label, impact_score = _impact_stability(motion_track, impact_idx)
    impact_stability = {
        "label": impact_label,
        "score": impact_score,
    }

    shaft_samples = _shaft_samples(club_head_track, handle_track, fps)
    if not shaft_samples:
        shaft_samples = _club_box_shaft_samples(club_track)
    shaft = _shaft_plane(shaft_samples, address_ms, top_ms, impact_ms)
    backswing = _backswing_metric(
        motion_track,
        person_track,
        address_idx,
        top_idx,
        height,
        wrist_track,
        str(event_source).startswith("pose_wrist"),
    )
    readiness = _readiness_metric(frames)
    tracking = _tracking_quality(frames, club_head_track, handle_track, club_track, ball_track, person_track)
    ball = _ball_metric(ball_track, impact_ms, width, height)
    wrist_sources: Dict[str, int] = {}
    for point in wrist_track:
        source_name = str(point.get("wristSource") or "unknown")
        wrist_sources[source_name] = wrist_sources.get(source_name, 0) + 1

    confidence = round(
        _clamp(
            tracking["score"] * 0.42
            + swing_conf * 0.16
            + shaft["confidence"] * 0.16
            + impact_score * 0.14
            + readiness["confidence"] * 0.12
        ),
        2,
    )

    coach_summary = _coach_comments(tempo, shaft, backswing, impact_stability, readiness, tracking)
    summary = f"service7 분석 완료: tempo {ratio}:1, shaft {shaft['label']}, backswing {backswing['label']}."
    duration_ms = _safe_int(meta.get("durationMs"), 0) or (_safe_int(frames[-1].get("_t_ms"), 0) if frames else 0)

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
                "source": motion_source,
            },
            "tempo": tempo,
            "impactStability": impact_stability,
            "shaftPlane": shaft,
            "backswing": backswing,
            "readiness": readiness,
            "trackingQuality": tracking,
            "ball": ball,
            "eventTiming": {
                "address": address_ms,
                "top": top_ms,
                "impact": impact_ms,
                "finish": finish_ms,
            },
        },
        "summary": summary,
        "coachSummary": coach_summary,
        "confidence": confidence,
        "analysisVersion": COACH_ANALYSIS_VERSION,
        "meta": {
            "fps": fps,
            "width": meta.get("width"),
            "height": meta.get("height"),
            "durationMs": duration_ms,
            "analysisVersion": COACH_ANALYSIS_VERSION,
            "modelLabels": SERVICE7_LABELS,
        },
        "debug": {
            "points": float(len(motion_track)),
            "motionSource": motion_source,
            "eventSource": event_source,
            "viewpoint": viewpoint or "unknown",
            "bodySelectorMethod": body_selector.get("method") if body_selector.get("available") else None,
            "bodySelectorUsed": bool(use_body_selector),
            "bodySelectorEvents": body_selector_events if body_selector.get("available") else None,
            "bodySelectorDebug": body_selector.get("debug") if body_selector.get("available") else None,
            "wristPoints": float(len(wrist_track)),
            "wristSources": wrist_sources,
            "wristTopMs": float(wrist_top.get("t")) if wrist_top else None,
            "wristImpactMs": float(wrist_impact.get("t")) if wrist_impact else None,
            "wristFinishMs": float(wrist_finish.get("t")) if wrist_finish else None,
            "eventIndices": {
                "address": int(address_idx),
                "top": int(top_idx),
                "impact": int(impact_idx),
                "finish": int(finish_idx),
            },
            "speedMax": float(max(speeds)) if speeds else 0.0,
            "shaftSamples": float(shaft.get("sampleCount") or 0),
            "trackingScore": float(tracking["score"]),
        },
    }
