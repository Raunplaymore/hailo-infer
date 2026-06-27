import math
from typing import Dict, Iterable, List, Optional, Tuple

SERVICE7_LABELS = {
    0: "person",
    1: "player_ready",
    2: "player_not_ready",
    3: "golf_ball",
    4: "club_head",
    5: "club",
    6: "club_handle",
}

PERSON_LABELS = {"person", "player", "golfer"}
READY_LABELS = {"player_ready", "ready"}
NOT_READY_LABELS = {"player_not_ready", "not_ready", "player_unready"}
BALL_LABELS = {"golf_ball", "golfball", "ball"}
CLUBHEAD_LABELS = {"club_head", "clubhead", "club_head_center", "golf_club_head"}
CLUB_LABELS = {"club", "golf_club", "baseball_bat", "bat"}
HANDLE_LABELS = {"club_handle", "handle", "grip", "club_grip"}


class CoachError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


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
                "frame": int(frame.get("frame", frame_idx)),
                "conf": _detection_conf(best),
                "label": _detection_label(best),
            }
        )
    return track


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

    if angle >= 62:
        label = "steep"
        comment = "다운스윙 샤프트가 2D 기준으로 세워진 편입니다. 전환 이후 손이 앞서가며 클럽이 늦게 내려오는지 확인하세요."
    elif angle <= 38:
        label = "flat"
        comment = "다운스윙 샤프트가 2D 기준으로 눕는 편입니다. 클럽이 몸 뒤에 남아 임팩트 타이밍이 늦어질 수 있습니다."
    else:
        label = "neutral"
        comment = "다운스윙 샤프트 각도는 단일 카메라 2D 기준에서 중립 범위입니다."

    avg_conf = _mean([s["confidence"] for s in samples])
    confidence = _clamp(avg_conf * min(1.0, len(samples) / 8.0))
    return {
        "label": label,
        "confidence": round(confidence, 2),
        "angleDeg": round(angle, 1),
        "addressAngleDeg": round(address_angle, 1) if address_angle is not None else None,
        "sampleCount": len(samples),
        "comment": comment,
    }


def _person_height(person_track: List[dict], target_ms: int, image_height: Optional[float]) -> Tuple[Optional[dict], float]:
    person = _nearest_track_point(person_track, target_ms, None)
    if person:
        return person, max(1.0, person["h"])
    if image_height:
        return None, max(1.0, image_height * 0.72)
    return None, 1.0


def _backswing_metric(
    motion_track: List[dict],
    person_track: List[dict],
    address_idx: int,
    top_idx: int,
    image_height: Optional[float],
) -> Dict[str, object]:
    address = motion_track[address_idx]
    top = motion_track[top_idx]
    person, person_h = _person_height(person_track, int(top["t"]), image_height)

    pre_top = motion_track[address_idx : top_idx + 1] or [address, top]
    highest_y = min(p["y"] for p in pre_top)
    vertical_travel = max(0.0, address["y"] - highest_y)
    travel_ratio = vertical_travel / person_h

    top_height_ratio = None
    if person:
        person_top = person["y"] - person["h"] / 2.0
        top_height_ratio = (top["y"] - person_top) / person_h

    score = _clamp(travel_ratio / 0.42)
    if travel_ratio < 0.18:
        label = "short"
        comment = "백스윙 탑까지 클럽 이동량이 작습니다. 어깨 회전과 손 위치가 충분히 올라가는지 확인하세요."
    elif top_height_ratio is not None and top_height_ratio > 0.62:
        label = "low_top"
        comment = "백스윙 탑 위치가 낮게 잡힙니다. 전신 프레임에서 왼팔과 어깨 회전 폭을 함께 확인하세요."
    elif top_height_ratio is not None and top_height_ratio < -0.08:
        label = "high_top"
        comment = "백스윙 탑이 매우 높게 잡힙니다. 오버스윙이거나 카메라 각도 영향일 수 있습니다."
    else:
        label = "adequate"
        comment = "백스윙 크기는 service7 클럽 추적 기준에서 적정 범위입니다."

    return {
        "label": label,
        "score": round(score, 2),
        "clubTravelRatio": round(travel_ratio, 2),
        "topHeightRatio": round(top_height_ratio, 2) if top_height_ratio is not None else None,
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
        + coverage(ball) * 0.15
        + coverage(person) * 0.2
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
        "ballFrames": len(ball),
        "personFrames": len(person),
        "clubHeadConfidence": round(avg_conf(club_head), 2),
        "clubHandleConfidence": round(avg_conf(handle), 2),
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

    if tracking.get("label") == "weak":
        comments.append("추적 품질이 낮습니다. 전신이 프레임 안에 들어오고 클럽 헤드가 배경과 분리되도록 조명과 거리부터 조정하세요.")

    return comments[:6]


def analyze_meta(meta: Dict[str, object], job_id: str, force: bool) -> Dict[str, object]:
    frames = meta.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise CoachError("NOT_SWING", "meta frames missing")

    min_points = 6
    min_conf = 0.15
    fps = int(meta.get("fps", 60))
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

    motion_track = club_head_track or handle_track or club_track
    motion_source = "club_head" if club_head_track else "club_handle" if handle_track else "club"

    if len(motion_track) < min_points:
        if not force:
            raise CoachError("NOT_SWING", "insufficient service7 club detections")
    confs = [p["conf"] for p in motion_track]
    if confs and _mean(confs) < min_conf and not force:
        raise CoachError("NOT_SWING", "service7 club detections too weak")

    speeds = _speeds(motion_track)
    if not speeds or max(speeds) <= 0:
        if not force:
            raise CoachError("NOT_SWING", "insufficient motion")

    impact_idx = _argmax(speeds) if speeds else None
    top_idx = _find_top(motion_track, impact_idx or 0) if impact_idx is not None else None
    address_idx = _find_stable_index(speeds, 0, 1)
    finish_idx = _find_stable_index(speeds, len(speeds) - 1, -1)

    if impact_idx is None or top_idx is None or address_idx is None or finish_idx is None:
        if not force:
            raise CoachError("NOT_SWING", "event segmentation failed")

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

    address_ms = int(motion_track[address_idx]["t"])
    top_ms = int(motion_track[top_idx]["t"])
    impact_ms = int(motion_track[impact_idx]["t"])
    finish_ms = int(motion_track[finish_idx]["t"])

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

    shaft = _shaft_plane(_shaft_samples(club_head_track, handle_track, fps), address_ms, top_ms, impact_ms)
    backswing = _backswing_metric(motion_track, person_track, address_idx, top_idx, height)
    readiness = _readiness_metric(frames)
    tracking = _tracking_quality(frames, club_head_track, handle_track, ball_track, person_track)
    ball = _ball_metric(ball_track, impact_ms, width, height)

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
        "meta": {
            "fps": fps,
            "width": meta.get("width"),
            "height": meta.get("height"),
            "durationMs": duration_ms,
            "analysisVersion": "hailo-coach-service7-v1",
            "modelLabels": SERVICE7_LABELS,
        },
        "debug": {
            "points": float(len(motion_track)),
            "speedMax": float(max(speeds)) if speeds else 0.0,
            "shaftSamples": float(shaft.get("sampleCount") or 0),
            "trackingScore": float(tracking["score"]),
        },
    }
