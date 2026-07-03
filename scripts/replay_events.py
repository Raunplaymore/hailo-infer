#!/usr/bin/env python3
"""Replay swing event segmentation against hand-labeled fixtures.

This is intentionally a local experiment harness. It does not change service
output; it lets us score event logic against saved body/meta artifacts before
deploying another heuristic.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.coach_pipeline import (  # noqa: E402
    CLUBHEAD_LABELS,
    CLUB_LABELS,
    HANDLE_LABELS,
    _choose_motion_track,
    _find_finish_from_wrist_track,
    _find_impact_from_wrist_track,
    _find_top_from_wrist_track,
    _keypoint_xyc,
    _nearest_track_index_by_time,
    _normalize_times,
    _safe_float,
    _safe_int,
    _select_best_track,
    _wrist_track_from_body,
    analyze_meta,
)


EVENT_KEYS = ("addressMs", "topMs", "impactMs", "finishMs")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _resolve_path(raw: Optional[str], base_dir: Path) -> Optional[Path]:
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _event_errors(predicted: Dict[str, Any], labels: Dict[str, Any]) -> Dict[str, Optional[float]]:
    errors: Dict[str, Optional[float]] = {}
    for key in EVENT_KEYS:
        if key not in predicted or key not in labels:
            errors[key] = None
            continue
        errors[key] = abs(_safe_float(predicted.get(key), 0.0) - _safe_float(labels.get(key), 0.0))
    return errors


def _status(errors: Dict[str, Optional[float]], tolerance_ms: float) -> str:
    comparable = [value for value in errors.values() if value is not None]
    if not comparable:
        return "missing"
    return "pass" if all(value <= tolerance_ms for value in comparable) else "fail"


def _wrist_events(body_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    wrist_track = _wrist_track_from_body(body_payload)
    return _events_from_track(wrist_track)


def _events_from_track(wrist_track: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not wrist_track:
        return {"available": False, "wristPoints": 0}

    address = wrist_track[0]
    top = _find_top_from_wrist_track(wrist_track, _safe_float(address.get("t"), 0.0))
    impact = _find_impact_from_wrist_track(wrist_track, top) if top else None
    finish = _find_finish_from_wrist_track(wrist_track, impact) if impact else None
    return {
        "available": True,
        "wristPoints": len(wrist_track),
        "events": {
            "addressMs": round(_safe_float(address.get("t"), 0.0)),
            "topMs": round(_safe_float(top.get("t"), 0.0)) if top else None,
            "impactMs": round(_safe_float(impact.get("t"), 0.0)) if impact else None,
            "finishMs": round(_safe_float(finish.get("t"), 0.0)) if finish else None,
        },
        "sources": _source_counts(wrist_track),
    }


def _track_scale(track: list[Dict[str, Any]]) -> float:
    if not track:
        return 1.0
    xs = [_safe_float(point.get("x"), 0.0) for point in track]
    ys = [_safe_float(point.get("y"), 0.0) for point in track]
    return max(1e-6, max(xs) - min(xs), max(ys) - min(ys))


def _first_backswing_local_events(wrist_track: list[Dict[str, Any]]) -> Dict[str, Any]:
    if len(wrist_track) < 5:
        return {"available": False, "wristPoints": len(wrist_track)}

    address = wrist_track[0]
    address_t = _safe_float(address.get("t"), 0.0)
    address_y = _safe_float(address.get("y"), 0.0)
    last_t = _safe_float(wrist_track[-1].get("t"), address_t)
    duration = max(1.0, last_t - address_t)
    scale = _track_scale(wrist_track)

    top_search_start = address_t + min(140.0, duration * 0.1)
    top_search_end = address_t + min(700.0, duration * 0.45)
    candidates = [
        point
        for point in wrist_track
        if top_search_start <= _safe_float(point.get("t"), 0.0) <= top_search_end
    ]
    if len(candidates) < 3:
        return {"available": False, "wristPoints": len(wrist_track)}

    min_rise = max(0.055, scale * 0.18)
    plateau_band = max(0.012, scale * 0.08)
    reversal_drop = max(0.018, scale * 0.11)
    peak = candidates[0]
    min_y = _safe_float(peak.get("y"), 0.0)
    top = None
    for point in candidates[1:]:
        point_y = _safe_float(point.get("y"), 0.0)
        if point_y < min_y:
            min_y = point_y
            peak = point
            continue
        if point_y <= min_y + plateau_band:
            peak = point
            continue
        peak_t = _safe_float(peak.get("t"), 0.0)
        height_gain = address_y - min_y
        if height_gain >= min_rise and _safe_float(point.get("t"), 0.0) >= peak_t + 30.0:
            if point_y - min_y >= reversal_drop:
                top = peak
                break

    if top is None:
        top = min(candidates, key=lambda point: _safe_float(point.get("y"), 0.0))

    top_t = _safe_float(top.get("t"), 0.0)
    impact_candidates = [
        point
        for point in wrist_track
        if top_t + 80.0 <= _safe_float(point.get("t"), 0.0) <= top_t + 420.0
    ]
    impact = None
    for point in impact_candidates:
        point_y = _safe_float(point.get("y"), 0.0)
        if point_y >= address_y - max(0.045, scale * 0.22):
            impact = point
            break
    if impact is None and impact_candidates:
        impact = max(impact_candidates, key=lambda point: _safe_float(point.get("y"), 0.0))

    finish = None
    if impact:
        impact_t = _safe_float(impact.get("t"), 0.0)
        finish_candidates = [
            point
            for point in wrist_track
            if impact_t + 180.0 <= _safe_float(point.get("t"), 0.0) <= impact_t + 520.0
        ]
        if finish_candidates:
            target_t = impact_t + 330.0
            finish = min(finish_candidates, key=lambda point: abs(_safe_float(point.get("t"), 0.0) - target_t))

    return {
        "available": True,
        "wristPoints": len(wrist_track),
        "events": {
            "addressMs": round(address_t),
            "topMs": round(_safe_float(top.get("t"), 0.0)) if top else None,
            "impactMs": round(_safe_float(impact.get("t"), 0.0)) if impact else None,
            "finishMs": round(_safe_float(finish.get("t"), 0.0)) if finish else None,
        },
        "sources": _source_counts(wrist_track),
    }


def _phase_turnaround_events(wrist_track: list[Dict[str, Any]]) -> Dict[str, Any]:
    if len(wrist_track) < 5:
        return {"available": False, "wristPoints": len(wrist_track)}

    address = wrist_track[0]
    address_t = _safe_float(address.get("t"), 0.0)
    address_x = _safe_float(address.get("x"), 0.0)
    address_y = _safe_float(address.get("y"), 0.0)
    last_t = _safe_float(wrist_track[-1].get("t"), address_t)
    duration = max(1.0, last_t - address_t)
    scale = _track_scale(wrist_track)

    search_start = address_t + min(140.0, duration * 0.1)
    search_end = address_t + min(560.0, duration * 0.34)
    early = [
        point
        for point in wrist_track
        if search_start <= _safe_float(point.get("t"), 0.0) <= search_end
    ]
    if len(early) < 3:
        return {"available": False, "wristPoints": len(wrist_track)}

    extreme = max(early, key=lambda point: abs(_safe_float(point.get("x"), 0.0) - address_x))
    extreme_t = _safe_float(extreme.get("t"), 0.0)
    extreme_x = _safe_float(extreme.get("x"), 0.0)
    excursion = extreme_x - address_x
    if abs(excursion) < max(0.045, scale * 0.18):
        return {"available": False, "wristPoints": len(wrist_track)}

    direction = 1.0 if excursion > 0 else -1.0
    top = None
    return_candidates = [
        point
        for point in wrist_track
        if extreme_t + 60.0 <= _safe_float(point.get("t"), 0.0) <= extreme_t + 220.0
    ]
    for point in return_candidates:
        returned = direction * (extreme_x - _safe_float(point.get("x"), 0.0))
        height_gain = address_y - _safe_float(point.get("y"), 0.0)
        if returned >= abs(excursion) * 0.28 and height_gain >= max(0.06, scale * 0.22):
            top = point
            break
    if top is None and return_candidates:
        target_t = extreme_t + 120.0
        top = min(return_candidates, key=lambda point: abs(_safe_float(point.get("t"), 0.0) - target_t))
    if top is None:
        top = extreme

    top_t = _safe_float(top.get("t"), 0.0)
    impact = None
    impact_candidates = [
        point
        for point in wrist_track
        if top_t + 70.0 <= _safe_float(point.get("t"), 0.0) <= top_t + 320.0
    ]
    for point in impact_candidates:
        point_y = _safe_float(point.get("y"), 0.0)
        point_x = _safe_float(point.get("x"), 0.0)
        near_address_height = point_y >= address_y - max(0.045, scale * 0.22)
        crossed_back = direction * (point_x - extreme_x) >= abs(excursion) * 0.75
        if near_address_height and crossed_back:
            impact = point
            break
    if impact is None and impact_candidates:
        impact = max(
            impact_candidates,
            key=lambda point: (
                _safe_float(point.get("y"), 0.0)
                - abs(_safe_float(point.get("x"), 0.0) - address_x) * 0.35
            ),
        )

    finish = None
    if impact:
        impact_t = _safe_float(impact.get("t"), 0.0)
        finish_candidates = [
            point
            for point in wrist_track
            if impact_t + 220.0 <= _safe_float(point.get("t"), 0.0) <= impact_t + 460.0
        ]
        if finish_candidates:
            target_t = impact_t + 335.0
            finish = min(finish_candidates, key=lambda point: abs(_safe_float(point.get("t"), 0.0) - target_t))

    return {
        "available": True,
        "wristPoints": len(wrist_track),
        "events": {
            "addressMs": round(address_t),
            "topMs": round(_safe_float(top.get("t"), 0.0)) if top else None,
            "impactMs": round(_safe_float(impact.get("t"), 0.0)) if impact else None,
            "finishMs": round(_safe_float(finish.get("t"), 0.0)) if finish else None,
        },
        "sources": _source_counts(wrist_track),
        "debug": {
            "extremeMs": round(extreme_t),
            "extremeX": round(extreme_x, 4),
            "addressX": round(address_x, 4),
            "excursion": round(excursion, 4),
            "direction": "positive" if direction > 0 else "negative",
        },
    }


def _unwrap_angle(prev: Optional[float], angle: float) -> float:
    if prev is None:
        return angle
    while angle - prev > math.pi:
        angle -= math.tau
    while angle - prev < -math.pi:
        angle += math.tau
    return angle


def _arm_angle_track(body_payload: Optional[Dict[str, Any]], side: str) -> list[Dict[str, Any]]:
    frames = body_payload.get("frames") if isinstance(body_payload, dict) else None
    if not isinstance(frames, list):
        return []
    track: list[Dict[str, Any]] = []
    prev_angle: Optional[float] = None
    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        if side == "mid":
            left_wrist = _keypoint_xyc(frame, "left_wrist")
            right_wrist = _keypoint_xyc(frame, "right_wrist")
            left_shoulder = _keypoint_xyc(frame, "left_shoulder")
            right_shoulder = _keypoint_xyc(frame, "right_shoulder")
            if not (left_wrist and right_wrist and left_shoulder and right_shoulder):
                continue
            if min(left_wrist[2], right_wrist[2], left_shoulder[2], right_shoulder[2]) < 0.02:
                continue
            wrist = ((left_wrist[0] + right_wrist[0]) / 2.0, (left_wrist[1] + right_wrist[1]) / 2.0)
            shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2.0, (left_shoulder[1] + right_shoulder[1]) / 2.0)
            conf = min(left_wrist[2], right_wrist[2], left_shoulder[2], right_shoulder[2])
        else:
            wrist_point = _keypoint_xyc(frame, f"{side}_wrist")
            shoulder_point = _keypoint_xyc(frame, f"{side}_shoulder")
            if not wrist_point or not shoulder_point:
                continue
            if min(wrist_point[2], shoulder_point[2]) < 0.02:
                continue
            wrist = (wrist_point[0], wrist_point[1])
            shoulder = (shoulder_point[0], shoulder_point[1])
            conf = min(wrist_point[2], shoulder_point[2])
        dx = wrist[0] - shoulder[0]
        dy = wrist[1] - shoulder[1]
        angle = _unwrap_angle(prev_angle, math.atan2(dy, dx))
        prev_angle = angle
        track.append(
            {
                "t": _safe_float(frame.get("timeMs"), idx * 33.33),
                "frame": _safe_int(frame.get("frameIndex"), idx),
                "angle": angle,
                "radius": math.hypot(dx, dy),
                "conf": conf,
                "source": f"{side}_arm_angle",
            }
        )
    track.sort(key=lambda point: (_safe_int(point.get("frame"), 0), _safe_float(point.get("t"), 0.0)))
    return track


def _angle_phase_events(angle_track: list[Dict[str, Any]]) -> Dict[str, Any]:
    if len(angle_track) < 5:
        return {"available": False, "points": len(angle_track)}

    address = angle_track[0]
    address_t = _safe_float(address.get("t"), 0.0)
    address_angle = _safe_float(address.get("angle"), 0.0)
    last_t = _safe_float(angle_track[-1].get("t"), address_t)
    duration = max(1.0, last_t - address_t)

    search_start = address_t + min(80.0, duration * 0.06)
    search_end = address_t + min(760.0, duration * 0.38)
    candidates = [
        point
        for point in angle_track
        if search_start <= _safe_float(point.get("t"), 0.0) <= search_end
    ]
    if len(candidates) < 3:
        return {"available": False, "points": len(angle_track)}

    deltas = [abs(_safe_float(point.get("angle"), 0.0) - address_angle) for point in candidates]
    max_delta = max(deltas) if deltas else 0.0
    if max_delta < 0.08:
        return {"available": False, "points": len(angle_track)}
    top = max(candidates, key=lambda point: abs(_safe_float(point.get("angle"), 0.0) - address_angle))
    top_t = _safe_float(top.get("t"), 0.0)
    top_angle = _safe_float(top.get("angle"), 0.0)
    direction = 1.0 if top_angle - address_angle >= 0 else -1.0

    impact = None
    impact_candidates = [
        point
        for point in angle_track
        if top_t + 80.0 <= _safe_float(point.get("t"), 0.0) <= top_t + 330.0
    ]
    for point in impact_candidates:
        return_ratio = direction * (top_angle - _safe_float(point.get("angle"), 0.0)) / max(max_delta, 1e-6)
        if return_ratio >= 0.62:
            impact = point
            break
    if impact is None and impact_candidates:
        target_t = top_t + 160.0
        impact = min(impact_candidates, key=lambda point: abs(_safe_float(point.get("t"), 0.0) - target_t))

    finish = None
    if impact:
        impact_t = _safe_float(impact.get("t"), 0.0)
        finish_candidates = [
            point
            for point in angle_track
            if impact_t + 220.0 <= _safe_float(point.get("t"), 0.0) <= impact_t + 520.0
        ]
        if finish_candidates:
            target_t = impact_t + 330.0
            finish = min(finish_candidates, key=lambda point: abs(_safe_float(point.get("t"), 0.0) - target_t))

    return {
        "available": True,
        "points": len(angle_track),
        "events": {
            "addressMs": round(address_t),
            "topMs": round(_safe_float(top.get("t"), 0.0)) if top else None,
            "impactMs": round(_safe_float(impact.get("t"), 0.0)) if impact else None,
            "finishMs": round(_safe_float(finish.get("t"), 0.0)) if finish else None,
        },
        "debug": {
            "addressAngle": round(address_angle, 4),
            "topAngle": round(top_angle, 4),
            "maxDelta": round(max_delta, 4),
            "direction": "positive" if direction > 0 else "negative",
        },
    }


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(a[1] - b[1], a[0] - b[0])


def _midpoint(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, min(a[2], b[2]))


def _body_feature_tracks(body_payload: Optional[Dict[str, Any]]) -> Dict[str, list[Dict[str, Any]]]:
    frames = body_payload.get("frames") if isinstance(body_payload, dict) else None
    if not isinstance(frames, list):
        return {}
    tracks: Dict[str, list[Dict[str, Any]]] = {}

    def add(name: str, t: float, value: float, conf: float) -> None:
        if conf < 0.02 or not math.isfinite(value):
            return
        tracks.setdefault(name, []).append({"t": t, "value": value, "conf": conf})

    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        t = _safe_float(frame.get("timeMs"), idx * 33.33)
        points = {
            name: _keypoint_xyc(frame, name)
            for name in (
                "nose",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            )
        }
        for name, point in points.items():
            if not point:
                continue
            add(f"{name}_x", t, point[0], point[2])
            add(f"{name}_y", t, point[1], point[2])

        pairs = {
            "wrist_mid": ("left_wrist", "right_wrist"),
            "shoulder_mid": ("left_shoulder", "right_shoulder"),
            "hip_mid": ("left_hip", "right_hip"),
            "ankle_mid": ("left_ankle", "right_ankle"),
        }
        mids: Dict[str, tuple[float, float, float]] = {}
        for name, (left_name, right_name) in pairs.items():
            left = points.get(left_name)
            right = points.get(right_name)
            if not left or not right:
                continue
            mid = _midpoint(left, right)
            mids[name] = mid
            add(f"{name}_x", t, mid[0], mid[2])
            add(f"{name}_y", t, mid[1], mid[2])

        distance_pairs = {
            "wrist_gap": ("left_wrist", "right_wrist"),
            "shoulder_width": ("left_shoulder", "right_shoulder"),
            "hip_width": ("left_hip", "right_hip"),
            "left_arm_span": ("left_shoulder", "left_wrist"),
            "right_arm_span": ("right_shoulder", "right_wrist"),
            "left_elbow_span": ("left_elbow", "left_wrist"),
            "right_elbow_span": ("right_elbow", "right_wrist"),
        }
        for name, (a_name, b_name) in distance_pairs.items():
            a = points.get(a_name)
            b = points.get(b_name)
            if not a or not b:
                continue
            add(name, t, _distance((a[0], a[1]), (b[0], b[1])), min(a[2], b[2]))

        if "wrist_mid" in mids and "shoulder_mid" in mids:
            add(
                "wrist_to_shoulder_mid",
                t,
                _distance((mids["wrist_mid"][0], mids["wrist_mid"][1]), (mids["shoulder_mid"][0], mids["shoulder_mid"][1])),
                min(mids["wrist_mid"][2], mids["shoulder_mid"][2]),
            )
            add(
                "wrist_shoulder_angle",
                t,
                _angle_between((mids["wrist_mid"][0], mids["wrist_mid"][1]), (mids["shoulder_mid"][0], mids["shoulder_mid"][1])),
                min(mids["wrist_mid"][2], mids["shoulder_mid"][2]),
            )
        if "shoulder_mid" in mids and "hip_mid" in mids:
            add(
                "torso_center_x",
                t,
                (mids["shoulder_mid"][0] + mids["hip_mid"][0]) / 2.0,
                min(mids["shoulder_mid"][2], mids["hip_mid"][2]),
            )
            add(
                "torso_center_y",
                t,
                (mids["shoulder_mid"][1] + mids["hip_mid"][1]) / 2.0,
                min(mids["shoulder_mid"][2], mids["hip_mid"][2]),
            )
            add(
                "torso_axis_angle",
                t,
                _angle_between((mids["shoulder_mid"][0], mids["shoulder_mid"][1]), (mids["hip_mid"][0], mids["hip_mid"][1])),
                min(mids["shoulder_mid"][2], mids["hip_mid"][2]),
            )

        angle_pairs = {
            "shoulder_line_angle": ("left_shoulder", "right_shoulder"),
            "hip_line_angle": ("left_hip", "right_hip"),
            "left_upper_arm_angle": ("left_elbow", "left_shoulder"),
            "right_upper_arm_angle": ("right_elbow", "right_shoulder"),
            "left_forearm_angle": ("left_wrist", "left_elbow"),
            "right_forearm_angle": ("right_wrist", "right_elbow"),
        }
        for name, (a_name, b_name) in angle_pairs.items():
            a = points.get(a_name)
            b = points.get(b_name)
            if not a or not b:
                continue
            add(name, t, _angle_between((a[0], a[1]), (b[0], b[1])), min(a[2], b[2]))

    for track in tracks.values():
        track.sort(key=lambda point: _safe_float(point.get("t"), 0.0))
    return tracks


def _probe_candidate_times(track: list[Dict[str, Any]], kind: str) -> list[float]:
    if len(track) < 3:
        return []
    values = [_safe_float(point.get("value"), 0.0) for point in track]
    times = [_safe_float(point.get("t"), 0.0) for point in track]
    if kind == "global_min":
        return [times[min(range(len(values)), key=lambda idx: values[idx])]]
    if kind == "global_max":
        return [times[max(range(len(values)), key=lambda idx: values[idx])]]

    candidates: list[float] = []
    if kind in {"local_min", "local_max"}:
        for idx in range(1, len(values) - 1):
            if kind == "local_min" and values[idx] <= values[idx - 1] and values[idx] <= values[idx + 1]:
                candidates.append(times[idx])
            if kind == "local_max" and values[idx] >= values[idx - 1] and values[idx] >= values[idx + 1]:
                candidates.append(times[idx])
        return candidates

    velocities: list[tuple[float, float]] = []
    for idx in range(1, len(track)):
        dt = max(1e-6, times[idx] - times[idx - 1])
        velocities.append((times[idx], (values[idx] - values[idx - 1]) / dt))
    if len(velocities) < 3:
        return []
    velocity_values = [item[1] for item in velocities]
    velocity_times = [item[0] for item in velocities]
    if kind == "velocity_pos_peak":
        return [velocity_times[max(range(len(velocity_values)), key=lambda idx: velocity_values[idx])]]
    if kind == "velocity_neg_peak":
        return [velocity_times[min(range(len(velocity_values)), key=lambda idx: velocity_values[idx])]]
    if kind == "abs_velocity_peak":
        return [velocity_times[max(range(len(velocity_values)), key=lambda idx: abs(velocity_values[idx]))]]
    return []


def _nearest_candidate_time(candidates: list[float], target: Any) -> Optional[float]:
    if target is None or not candidates:
        return None
    target_t = _safe_float(target, 0.0)
    return min(candidates, key=lambda candidate: abs(candidate - target_t))


def _print_feature_probe_experiment(
    body_payload: Optional[Dict[str, Any]],
    labels: Dict[str, Any],
    tolerance_ms: float,
) -> None:
    print("\nexperiment body-feature-probe oracle")
    feature_tracks = _body_feature_tracks(body_payload)
    if not feature_tracks:
        print("  missing")
        return

    scored: list[tuple[float, str, Dict[str, Optional[int]]]] = []
    for feature_name, track in feature_tracks.items():
        if len(track) < 5:
            continue
        for kind in (
            "global_min",
            "global_max",
            "local_min",
            "local_max",
            "velocity_pos_peak",
            "velocity_neg_peak",
            "abs_velocity_peak",
        ):
            candidates = _probe_candidate_times(track, kind)
            if not candidates:
                continue
            events: Dict[str, Optional[int]] = {"addressMs": labels.get("addressMs")}
            for key in ("topMs", "impactMs", "finishMs"):
                nearest = _nearest_candidate_time(candidates, labels.get(key))
                events[key] = round(nearest) if nearest is not None else None
            errors = _event_errors(events, labels)
            comparable = [value for key, value in errors.items() if key != "addressMs" and value is not None]
            if len(comparable) < 2:
                continue
            total = sum(comparable)
            scored.append((total, f"{feature_name}/{kind}", events))

    if not scored:
        print("  missing")
        return
    for total, name, events in sorted(scored, key=lambda item: item[0])[:10]:
        errors = _event_errors(events, labels)
        status = _status(errors, tolerance_ms)
        compact = " ".join(f"{key}={events.get(key)}" for key in EVENT_KEYS)
        print(f"  {name}: {status} {compact} probeError={total:.0f}ms")


FEATURE_VOTE_RULES = (
    ("wrist_to_shoulder_mid", "local_min", 1.2),
    ("wrist_to_shoulder_mid", "local_max", 1.0),
    ("left_forearm_angle", "local_min", 1.1),
    ("left_forearm_angle", "local_max", 1.0),
    ("left_arm_span", "local_min", 1.0),
    ("left_elbow_y", "local_max", 1.3),
    ("left_upper_arm_angle", "local_max", 1.2),
    ("right_upper_arm_angle", "local_min", 1.0),
    ("shoulder_width", "local_max", 1.3),
    ("shoulder_line_angle", "local_max", 1.0),
    ("hip_line_angle", "local_max", 0.9),
    ("wrist_mid_x", "local_min", 0.9),
    ("wrist_mid_x", "local_max", 0.9),
    ("nose_y", "local_max", 0.8),
    ("ankle_mid_x", "local_max", 0.7),
)


def _cluster_candidate_times(candidates: list[tuple[float, float, str]], band_ms: float = 70.0) -> list[Dict[str, Any]]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: item[0])
    clusters: list[list[tuple[float, float, str]]] = []
    current: list[tuple[float, float, str]] = []
    for candidate in ordered:
        if not current:
            current = [candidate]
            continue
        center = sum(item[0] * item[1] for item in current) / max(1e-6, sum(item[1] for item in current))
        if abs(candidate[0] - center) <= band_ms:
            current.append(candidate)
        else:
            clusters.append(current)
            current = [candidate]
    if current:
        clusters.append(current)

    result: list[Dict[str, Any]] = []
    for cluster in clusters:
        weight = sum(item[1] for item in cluster)
        center = sum(item[0] * item[1] for item in cluster) / max(1e-6, weight)
        sources = sorted({item[2] for item in cluster})
        result.append({"t": center, "weight": weight, "count": len(cluster), "sources": sources})
    result.sort(key=lambda item: _safe_float(item.get("t"), 0.0))
    return result


def _feature_vote_clusters(body_payload: Optional[Dict[str, Any]], band_ms: float = 70.0) -> list[Dict[str, Any]]:
    feature_tracks = _body_feature_tracks(body_payload)
    if not feature_tracks:
        return []

    all_times = [
        _safe_float(point.get("t"), 0.0)
        for track in feature_tracks.values()
        for point in track
    ]
    if not all_times:
        return []
    address_t = min(all_times)
    last_t = max(all_times)
    duration = max(1.0, last_t - address_t)

    candidates: list[tuple[float, float, str]] = []
    for feature_name, kind, weight in FEATURE_VOTE_RULES:
        track = feature_tracks.get(feature_name)
        if not track:
            continue
        for candidate_t in _probe_candidate_times(track, kind):
            if candidate_t < address_t + 20.0 or candidate_t > address_t + min(2300.0, duration):
                continue
            candidates.append((candidate_t, weight, f"{feature_name}/{kind}"))
    return _cluster_candidate_times(candidates, band_ms=band_ms)


def _feature_vote_events(body_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    clusters = _feature_vote_clusters(body_payload)
    if len(clusters) < 2:
        return {"available": False, "clusters": clusters}

    address_t = min(_safe_float(cluster.get("t"), 0.0) for cluster in clusters)
    last_t = max(_safe_float(cluster.get("t"), 0.0) for cluster in clusters)
    duration = max(1.0, last_t - address_t)

    top_window_end = address_t + min(850.0, duration * 0.45)
    top_candidates = [
        cluster
        for cluster in clusters
        if address_t + 20.0 <= _safe_float(cluster.get("t"), 0.0) <= top_window_end
    ]
    if not top_candidates:
        return {"available": False, "clusters": clusters}
    # Prefer the first strong consensus cluster; late high-weight clusters often belong to follow-through.
    max_top_weight = max(_safe_float(cluster.get("weight"), 0.0) for cluster in top_candidates)
    strong_top = [
        cluster
        for cluster in top_candidates
        if _safe_float(cluster.get("weight"), 0.0) >= max_top_weight * 0.55
    ]
    top_cluster = min(strong_top, key=lambda cluster: _safe_float(cluster.get("t"), 0.0))
    top_t = _safe_float(top_cluster.get("t"), 0.0)

    impact_candidates = [
        cluster
        for cluster in clusters
        if top_t + 90.0 <= _safe_float(cluster.get("t"), 0.0) <= top_t + 650.0
    ]
    if impact_candidates:
        impact_cluster = max(
            impact_candidates,
            key=lambda cluster: (
                _safe_float(cluster.get("weight"), 0.0),
                -abs((_safe_float(cluster.get("t"), 0.0) - top_t) - 180.0),
            ),
        )
    else:
        impact_cluster = None
    impact_t = _safe_float(impact_cluster.get("t"), 0.0) if impact_cluster else None

    finish_cluster = None
    if impact_t is not None:
        finish_candidates = [
            cluster
            for cluster in clusters
            if impact_t + 150.0 <= _safe_float(cluster.get("t"), 0.0) <= impact_t + 780.0
        ]
        if finish_candidates:
            finish_cluster = max(
                finish_candidates,
                key=lambda cluster: (
                    _safe_float(cluster.get("weight"), 0.0),
                    -abs((_safe_float(cluster.get("t"), 0.0) - impact_t) - 330.0),
                ),
            )

    return {
        "available": True,
        "events": {
            "addressMs": round(address_t),
            "topMs": round(top_t),
            "impactMs": round(impact_t) if impact_t is not None else None,
            "finishMs": round(_safe_float(finish_cluster.get("t"), 0.0)) if finish_cluster else None,
        },
        "clusters": clusters[:12],
        "debug": {
            "topWeight": round(_safe_float(top_cluster.get("weight"), 0.0), 2),
            "impactWeight": round(_safe_float(impact_cluster.get("weight"), 0.0), 2) if impact_cluster else None,
            "finishWeight": round(_safe_float(finish_cluster.get("weight"), 0.0), 2) if finish_cluster else None,
        },
    }


def _feature_vote_events_from_clusters(
    clusters: list[Dict[str, Any]],
    start_t: float,
    *,
    use_offset_model: bool = False,
) -> Dict[str, Any]:
    if len(clusters) < 2:
        return {"available": False}

    top_candidates = [
        cluster
        for cluster in clusters
        if start_t + 20.0 <= _safe_float(cluster.get("t"), 0.0) <= start_t + 760.0
    ]
    if not top_candidates:
        return {"available": False}

    if use_offset_model:
        target_offset = 70.0 if start_t < 100.0 else 350.0
        top_cluster = min(
            top_candidates,
            key=lambda cluster: (
                abs((_safe_float(cluster.get("t"), 0.0) - start_t) - target_offset),
                -_safe_float(cluster.get("weight"), 0.0),
            ),
        )
    else:
        max_top_weight = max(_safe_float(cluster.get("weight"), 0.0) for cluster in top_candidates)
        strong_top = [
            cluster
            for cluster in top_candidates
            if _safe_float(cluster.get("weight"), 0.0) >= max_top_weight * 0.55
        ]
        top_cluster = min(strong_top, key=lambda cluster: _safe_float(cluster.get("t"), 0.0))
    top_t = _safe_float(top_cluster.get("t"), 0.0)

    impact_candidates = [
        cluster
        for cluster in clusters
        if top_t + 90.0 <= _safe_float(cluster.get("t"), 0.0) <= top_t + 650.0
    ]
    if impact_candidates:
        target_gap = 180.0 if use_offset_model else 180.0
        impact_cluster = min(
            impact_candidates,
            key=lambda cluster: (
                abs((_safe_float(cluster.get("t"), 0.0) - top_t) - target_gap),
                -_safe_float(cluster.get("weight"), 0.0),
            ),
        )
    else:
        impact_cluster = None
    impact_t = _safe_float(impact_cluster.get("t"), 0.0) if impact_cluster else None

    finish_cluster = None
    if impact_t is not None:
        finish_candidates = [
            cluster
            for cluster in clusters
            if impact_t + 150.0 <= _safe_float(cluster.get("t"), 0.0) <= impact_t + 780.0
        ]
        if finish_candidates:
            finish_cluster = min(
                finish_candidates,
                key=lambda cluster: (
                    abs((_safe_float(cluster.get("t"), 0.0) - impact_t) - 330.0),
                    -_safe_float(cluster.get("weight"), 0.0),
                ),
            )

    return {
        "available": True,
        "events": {
            "addressMs": round(start_t),
            "topMs": round(top_t),
            "impactMs": round(impact_t) if impact_t is not None else None,
            "finishMs": round(_safe_float(finish_cluster.get("t"), 0.0)) if finish_cluster else None,
        },
        "debug": {
            "startMs": round(start_t),
            "topWeight": round(_safe_float(top_cluster.get("weight"), 0.0), 2),
            "impactWeight": round(_safe_float(impact_cluster.get("weight"), 0.0), 2) if impact_cluster else None,
            "finishWeight": round(_safe_float(finish_cluster.get("weight"), 0.0), 2) if finish_cluster else None,
            "offsetModel": use_offset_model,
        },
    }


def _print_feature_vote_experiment(
    body_payload: Optional[Dict[str, Any]],
    labels: Dict[str, Any],
    tolerance_ms: float,
) -> None:
    print("\nexperiment body-feature-vote")
    candidate = _feature_vote_events(body_payload)
    if not candidate.get("available"):
        print("  missing")
        return
    events = candidate["events"]
    errors = _event_errors(events, labels)
    total = sum(value for value in errors.values() if value is not None)
    compact = " ".join(f"{key}={events.get(key)}" for key in EVENT_KEYS)
    print(f"  result: {_status(errors, tolerance_ms)} {compact} totalError={total:.0f}ms")
    print(f"  debug={json.dumps(candidate.get('debug', {}), ensure_ascii=False, sort_keys=True)}")
    clusters = candidate.get("clusters") if isinstance(candidate.get("clusters"), list) else []
    preview = [
        {
            "t": round(_safe_float(cluster.get("t"), 0.0)),
            "w": round(_safe_float(cluster.get("weight"), 0.0), 2),
            "n": _safe_int(cluster.get("count"), 0),
        }
        for cluster in clusters[:8]
    ]
    print(f"  clusters={json.dumps(preview, ensure_ascii=False)}")


def _print_feature_vote_gated_experiment(
    body_payload: Optional[Dict[str, Any]],
    labels: Dict[str, Any],
    tolerance_ms: float,
) -> None:
    print("\nexperiment body-feature-vote gated-candidates")
    base = _feature_vote_events(body_payload)
    clusters = base.get("clusters") if isinstance(base.get("clusters"), list) else []
    if not clusters:
        print("  missing")
        return

    scored: list[tuple[float, str, Dict[str, Any], Dict[str, Any]]] = []
    for cluster in clusters[:10]:
        start_t = _safe_float(cluster.get("t"), 0.0)
        for mode_name, use_offset_model in (("consensus", False), ("offset", True)):
            candidate = _feature_vote_events_from_clusters(clusters, start_t, use_offset_model=use_offset_model)
            if not candidate.get("available"):
                continue
            events = candidate["events"]
            errors = _event_errors(events, labels)
            total = sum(value for value in errors.values() if value is not None)
            scored.append((total, mode_name, events, candidate.get("debug", {})))

    if not scored:
        print("  missing")
        return
    for total, mode_name, events, debug in sorted(scored, key=lambda item: item[0])[:8]:
        errors = _event_errors(events, labels)
        compact = " ".join(f"{key}={events.get(key)}" for key in EVENT_KEYS)
        print(f"  {mode_name}: {_status(errors, tolerance_ms)} {compact} totalError={total:.0f}ms")
        print(f"    debug={json.dumps(debug, ensure_ascii=False, sort_keys=True)}")


def _print_feature_vote_band_sweep_experiment(
    body_payload: Optional[Dict[str, Any]],
    labels: Dict[str, Any],
    tolerance_ms: float,
) -> None:
    print("\nexperiment body-feature-vote band-sweep")
    scored: list[tuple[float, float, str, Dict[str, Any], Dict[str, Any]]] = []
    for band_ms in (45.0, 60.0, 75.0, 90.0, 110.0, 135.0):
        clusters = _feature_vote_clusters(body_payload, band_ms=band_ms)
        if not clusters:
            continue
        for cluster in clusters[:10]:
            start_t = _safe_float(cluster.get("t"), 0.0)
            for mode_name, use_offset_model in (("consensus", False), ("offset", True)):
                candidate = _feature_vote_events_from_clusters(clusters, start_t, use_offset_model=use_offset_model)
                if not candidate.get("available"):
                    continue
                events = candidate["events"]
                errors = _event_errors(events, labels)
                total = sum(value for value in errors.values() if value is not None)
                scored.append((total, band_ms, mode_name, events, candidate.get("debug", {})))

    if not scored:
        print("  missing")
        return
    for total, band_ms, mode_name, events, debug in sorted(scored, key=lambda item: item[0])[:10]:
        errors = _event_errors(events, labels)
        compact = " ".join(f"{key}={events.get(key)}" for key in EVENT_KEYS)
        print(
            f"  band={band_ms:.0f}/{mode_name}: {_status(errors, tolerance_ms)} "
            f"{compact} totalError={total:.0f}ms"
        )
        print(f"    debug={json.dumps(debug, ensure_ascii=False, sort_keys=True)}")


def _feature_sequence_events(
    feature_name: str,
    kind: str,
    feature_tracks: Dict[str, list[Dict[str, Any]]],
    start_t: float,
) -> Dict[str, Any]:
    track = feature_tracks.get(feature_name)
    if not track:
        return {"available": False}
    candidates = [
        candidate_t
        for candidate_t in _probe_candidate_times(track, kind)
        if candidate_t >= start_t + 20.0
    ]
    if len(candidates) < 3:
        return {"available": False}

    target_top_offset = 70.0 if start_t < 100.0 else 340.0
    top_pool = [candidate_t for candidate_t in candidates if start_t + 20.0 <= candidate_t <= start_t + 760.0]
    if not top_pool:
        return {"available": False}
    top_t = min(top_pool, key=lambda candidate_t: abs((candidate_t - start_t) - target_top_offset))

    impact_pool = [candidate_t for candidate_t in candidates if top_t + 70.0 <= candidate_t <= top_t + 420.0]
    if not impact_pool:
        return {"available": False}
    impact_t = min(impact_pool, key=lambda candidate_t: abs((candidate_t - top_t) - 170.0))

    finish_pool = [candidate_t for candidate_t in candidates if impact_t + 130.0 <= candidate_t <= impact_t + 620.0]
    if not finish_pool:
        return {"available": False}
    finish_t = min(finish_pool, key=lambda candidate_t: abs((candidate_t - impact_t) - 320.0))

    return {
        "available": True,
        "events": {
            "addressMs": round(start_t),
            "topMs": round(top_t),
            "impactMs": round(impact_t),
            "finishMs": round(finish_t),
        },
        "debug": {
            "feature": f"{feature_name}/{kind}",
            "candidateCount": len(candidates),
            "startMs": round(start_t),
        },
    }


SEQUENCE_FEATURE_PRIORITIES = {
    "left_elbow_y/local_max": 0.0,
    "shoulder_width/local_max": 0.2,
    "nose_y/local_max": 0.35,
    "ankle_mid_x/local_max": 0.45,
    "left_forearm_angle/local_max": 0.55,
    "left_upper_arm_angle/local_max": 0.65,
    "hip_line_angle/local_max": 0.8,
    "right_upper_arm_angle/local_min": 0.9,
}


def _sequence_internal_score(events: Dict[str, Any], debug: Dict[str, Any]) -> float:
    address = _safe_float(events.get("addressMs"), 0.0)
    top = _safe_float(events.get("topMs"), address)
    impact = _safe_float(events.get("impactMs"), top)
    finish = _safe_float(events.get("finishMs"), impact)
    top_gap = top - address
    down_gap = impact - top
    finish_gap = finish - impact
    if top_gap <= 0 or down_gap <= 0 or finish_gap <= 0:
        return float("inf")
    target_top_gap = 70.0 if address < 100.0 else 340.0
    feature_penalty = SEQUENCE_FEATURE_PRIORITIES.get(str(debug.get("feature")), 1.2)
    late_start_penalty = max(0.0, address - 120.0) / 520.0
    return (
        feature_penalty
        + late_start_penalty
        + abs(top_gap - target_top_gap) / 220.0
        + abs(down_gap - 150.0) / 180.0
        + abs(finish_gap - 310.0) / 260.0
        - min(0.4, _safe_float(debug.get("candidateCount"), 0.0) / 40.0)
    )


def _feature_sequence_ranked(body_payload: Optional[Dict[str, Any]]) -> list[tuple[float, Dict[str, Any], Dict[str, Any]]]:
    feature_tracks = _body_feature_tracks(body_payload)
    clusters = _feature_vote_clusters(body_payload, band_ms=70.0)
    if not feature_tracks or not clusters:
        return []

    ranked: list[tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for cluster in clusters[:10]:
        start_t = _safe_float(cluster.get("t"), 0.0)
        for feature_name, kind, _weight in FEATURE_VOTE_RULES:
            candidate = _feature_sequence_events(feature_name, kind, feature_tracks, start_t)
            if not candidate.get("available"):
                continue
            events = candidate["events"]
            debug = candidate.get("debug", {})
            ranked.append((_sequence_internal_score(events, debug), events, debug))
    ranked.sort(key=lambda item: item[0])
    return ranked


def _print_feature_sequence_experiment(
    body_payload: Optional[Dict[str, Any]],
    labels: Dict[str, Any],
    tolerance_ms: float,
) -> None:
    print("\nexperiment body-feature-sequence")
    ranked = _feature_sequence_ranked(body_payload)
    if not ranked:
        print("  missing")
        return
    scored = []
    for internal_score, events, debug in ranked:
        errors = _event_errors(events, labels)
        total = sum(value for value in errors.values() if value is not None)
        scored.append((total, internal_score, events, debug))
    for total, internal_score, events, debug in sorted(scored, key=lambda item: item[0])[:10]:
        errors = _event_errors(events, labels)
        compact = " ".join(f"{key}={events.get(key)}" for key in EVENT_KEYS)
        print(f"  {_status(errors, tolerance_ms)} {compact} totalError={total:.0f}ms")
        print(f"    score={internal_score:.2f} debug={json.dumps(debug, ensure_ascii=False, sort_keys=True)}")


def _body_event_selector_result(body_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    sequence_ranked = _feature_sequence_ranked(body_payload)
    vote = _feature_vote_events(body_payload)
    candidate_name = None
    candidate_events = None
    candidate_debug: Dict[str, Any] = {}

    if vote.get("available"):
        events = vote.get("events", {})
        clusters = vote.get("clusters") if isinstance(vote.get("clusters"), list) else []
        if clusters:
            first_cluster = clusters[0]
            first_t = _safe_float(first_cluster.get("t"), 9999.0)
            first_weight = _safe_float(first_cluster.get("weight"), 0.0)
            if first_t <= 120.0 and first_weight >= 9.0:
                impact_candidates = [
                    cluster
                    for cluster in clusters
                    if first_t + 300.0 <= _safe_float(cluster.get("t"), 0.0) <= first_t + 540.0
                ]
                impact_cluster = max(
                    impact_candidates,
                    key=lambda cluster: _safe_float(cluster.get("weight"), 0.0),
                    default=None,
                )
                impact_t = _safe_float(impact_cluster.get("t"), 0.0) if impact_cluster else None
                finish_cluster = None
                if impact_t is not None:
                    finish_candidates = [
                        cluster
                        for cluster in clusters
                        if impact_t + 160.0 <= _safe_float(cluster.get("t"), 0.0) <= impact_t + 360.0
                    ]
                    finish_cluster = max(
                        finish_candidates,
                        key=lambda cluster: _safe_float(cluster.get("weight"), 0.0),
                        default=None,
                    )
                if impact_cluster and finish_cluster:
                    candidate_name = "feature-vote-early-top-cluster"
                    candidate_events = {
                        "addressMs": 0,
                        "topMs": round(first_t),
                        "impactMs": round(impact_t),
                        "finishMs": round(_safe_float(finish_cluster.get("t"), 0.0)),
                    }
                    candidate_debug = {
                        "topWeight": round(first_weight, 2),
                        "impactWeight": round(_safe_float(impact_cluster.get("weight"), 0.0), 2),
                        "finishWeight": round(_safe_float(finish_cluster.get("weight"), 0.0), 2),
                    }
        address_t = _safe_float(events.get("addressMs"), 9999.0)
        top_t = _safe_float(events.get("topMs"), 9999.0)
        impact_t = _safe_float(events.get("impactMs"), 9999.0)
        finish_t = _safe_float(events.get("finishMs"), 9999.0)
        top_weight = _safe_float(vote.get("debug", {}).get("topWeight"), 0.0)
        # Short swings can have top almost immediately after address. Preserve that vote
        # before the generic down-the-line fallback drifts into follow-through clusters.
        if (
            candidate_events is None
            and address_t <= 120.0
            and top_t <= 260.0
            and impact_t <= 760.0
            and finish_t <= 1050.0
            and top_weight >= 9.0
        ):
            candidate_name = "feature-vote-early"
            candidate_events = events
            candidate_debug = vote.get("debug", {})

    if sequence_ranked:
        sequence_score, sequence_events, sequence_debug = sequence_ranked[0]
        if (
            str(sequence_debug.get("feature")) == "left_elbow_y/local_max"
            and _safe_float(sequence_debug.get("startMs"), 0.0) < 300.0
        ):
            shoulder_alternative = next(
                (
                    item
                    for item in sequence_ranked
                    if str(item[2].get("feature")) == "shoulder_width/local_max"
                    and item[0] <= sequence_score + 0.55
                ),
                None,
            )
            if shoulder_alternative:
                shoulder_score = shoulder_alternative[0]
                shoulder_alternatives = [
                    item
                    for item in sequence_ranked
                    if str(item[2].get("feature")) == "shoulder_width/local_max"
                    and item[0] <= shoulder_score + 0.2
                ]
                sequence_score, sequence_events, sequence_debug = min(
                    shoulder_alternatives,
                    key=lambda item: _safe_float(item[2].get("startMs"), 9999.0),
                )
        start_t = _safe_float(sequence_events.get("addressMs"), 0.0)
        if candidate_events is None and start_t >= 120.0 and sequence_score <= 2.6:
            candidate_name = "feature-sequence"
            candidate_events = sequence_events
            candidate_debug = {"score": round(sequence_score, 3), **sequence_debug}

    if candidate_events is None:
        clusters = vote.get("clusters") if isinstance(vote.get("clusters"), list) else []
        fallback: list[tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for cluster in clusters[:10]:
            start_t = _safe_float(cluster.get("t"), 0.0)
            candidate = _feature_vote_events_from_clusters(clusters, start_t, use_offset_model=True)
            if not candidate.get("available"):
                continue
            events = candidate["events"]
            top_gap = _safe_float(events.get("topMs"), 0.0) - _safe_float(events.get("addressMs"), 0.0)
            impact_gap = _safe_float(events.get("impactMs"), 0.0) - _safe_float(events.get("topMs"), 0.0)
            finish_gap = _safe_float(events.get("finishMs"), 0.0) - _safe_float(events.get("impactMs"), 0.0)
            score = abs(top_gap - 360.0) / 260.0 + abs(impact_gap - 170.0) / 220.0 + abs(finish_gap - 320.0) / 320.0
            fallback.append((score, events, candidate.get("debug", {})))
        if fallback:
            score, events, debug = sorted(fallback, key=lambda item: item[0])[0]
            candidate_name = "feature-vote-offset-fallback"
            candidate_events = events
            candidate_debug = {"score": round(score, 3), **debug}

    if candidate_events is None:
        return {
            "available": False,
            "sequenceRanked": sequence_ranked,
            "vote": vote,
        }
    return {
        "available": True,
        "method": candidate_name,
        "events": candidate_events,
        "debug": candidate_debug,
        "sequenceRanked": sequence_ranked,
        "vote": vote,
    }


def _print_body_event_selector_experiment(
    body_payload: Optional[Dict[str, Any]],
    labels: Dict[str, Any],
    tolerance_ms: float,
) -> None:
    print("\nexperiment body-event-selector")
    result = _body_event_selector_result(body_payload)
    if not result.get("available"):
        print("  missing")
        return
    candidate_name = result.get("method")
    candidate_events = result.get("events", {})
    candidate_debug = result.get("debug", {})
    vote = result.get("vote") if isinstance(result.get("vote"), dict) else {}
    sequence_ranked = result.get("sequenceRanked") if isinstance(result.get("sequenceRanked"), list) else []
    errors = _event_errors(candidate_events, labels)
    total = sum(value for value in errors.values() if value is not None)
    compact = " ".join(f"{key}={candidate_events.get(key)}" for key in EVENT_KEYS)
    print(f"  {candidate_name}: {_status(errors, tolerance_ms)} {compact} totalError={total:.0f}ms")
    print(f"    debug={json.dumps(candidate_debug, ensure_ascii=False, sort_keys=True)}")
    if vote.get("available"):
        print(
            "    voteEvents="
            f"{json.dumps(vote.get('events', {}), ensure_ascii=False, sort_keys=True)} "
            f"voteDebug={json.dumps(vote.get('debug', {}), ensure_ascii=False, sort_keys=True)}"
        )

    if sequence_ranked:
        debug_rows = []
        for internal_score, events, debug in sequence_ranked[:8]:
            candidate_errors = _event_errors(events, labels)
            candidate_total = sum(value for value in candidate_errors.values() if value is not None)
            debug_rows.append(
                {
                    "score": round(internal_score, 3),
                    "totalError": round(candidate_total),
                    "events": {key: events.get(key) for key in EVENT_KEYS},
                    "feature": debug.get("feature"),
                    "startMs": debug.get("startMs"),
                }
            )
        print(f"    sequenceRanked={json.dumps(debug_rows, ensure_ascii=False)}")


def _single_source_track(body_payload: Optional[Dict[str, Any]], source: str) -> list[Dict[str, Any]]:
    frames = body_payload.get("frames") if isinstance(body_payload, dict) else None
    if not isinstance(frames, list):
        return []
    track: list[Dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        point = None
        if source in {"left_wrist", "right_wrist"}:
            point = _keypoint_xyc(frame, source)
        elif source == "weighted_midpoint":
            left = _keypoint_xyc(frame, "left_wrist")
            right = _keypoint_xyc(frame, "right_wrist")
            if left and right and left[2] >= 0.02 and right[2] >= 0.02:
                total_conf = max(1e-6, left[2] + right[2])
                point = (
                    (left[0] * left[2] + right[0] * right[2]) / total_conf,
                    (left[1] * left[2] + right[1] * right[2]) / total_conf,
                    (left[2] + right[2]) / 2.0,
                )
        if point is None:
            continue
        x, y, conf = point
        if conf < 0.02:
            continue
        track.append(
            {
                "x": x,
                "y": y,
                "t": _safe_float(frame.get("timeMs"), idx * 33.33),
                "frame": _safe_int(frame.get("frameIndex"), idx),
                "conf": conf,
                "source": "pose_wrist",
                "wristSource": source,
            }
        )
    track.sort(key=lambda point: (_safe_int(point.get("frame"), 0), _safe_float(point.get("t"), 0.0)))
    return track


def _source_counts(track: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for point in track:
        source = str(point.get("wristSource") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _print_event_block(title: str, events: Dict[str, Any], labels: Dict[str, Any], tolerance_ms: float) -> None:
    errors = _event_errors(events, labels)
    print(f"\n{title}: {_status(errors, tolerance_ms)}")
    for key in EVENT_KEYS:
        predicted = events.get(key)
        gold = labels.get(key)
        error = errors.get(key)
        error_text = "-" if error is None else f"{error:.0f}ms"
        print(f"  {key}: predicted={predicted} gold={gold} error={error_text}")


def _print_delta_block(title: str, left: Dict[str, Any], right: Dict[str, Any]) -> None:
    print(f"\n{title}")
    for key in EVENT_KEYS:
        if left.get(key) is None or right.get(key) is None:
            print(f"  {key}: -")
            continue
        delta = _safe_float(left.get(key), 0.0) - _safe_float(right.get(key), 0.0)
        print(f"  {key}: {delta:+.0f}ms ({right.get(key)} -> {left.get(key)})")


def _motion_track_from_meta(meta_payload: Dict[str, Any]) -> tuple[list[Dict[str, Any]], str]:
    frames = meta_payload.get("frames", [])
    if not isinstance(frames, list):
        return [], "none"
    fps = max(1, _safe_int(meta_payload.get("fps"), 60))
    times_ms = _normalize_times(frames, fps)
    for idx, frame in enumerate(frames):
        if isinstance(frame, dict):
            frame["_t_ms"] = times_ms[idx] if times_ms[idx] is not None else idx * (1000.0 / fps)
    club_head_track = _select_best_track(frames, CLUBHEAD_LABELS)
    handle_track = _select_best_track(frames, HANDLE_LABELS)
    club_track = _select_best_track(frames, CLUB_LABELS)
    return _choose_motion_track(club_head_track, handle_track, club_track)


def _print_remap_diagnostics(meta_payload: Dict[str, Any], wrist_events: Dict[str, Any]) -> None:
    motion_track, motion_source = _motion_track_from_meta(meta_payload)
    print("\nmotion remap")
    print(f"  motionSource: {motion_source}")
    print(f"  motionFrames: {len(motion_track)}")
    if len(motion_track) >= 2:
        gaps = [
            _safe_float(motion_track[idx].get("t"), 0.0) - _safe_float(motion_track[idx - 1].get("t"), 0.0)
            for idx in range(1, len(motion_track))
        ]
        print(f"  maxGapMs: {max(gaps):.0f}")
    for key in ("topMs", "impactMs", "finishMs"):
        event_time = wrist_events.get(key)
        if event_time is None:
            print(f"  {key}: -")
            continue
        remap_idx = _nearest_track_index_by_time(motion_track, _safe_float(event_time, 0.0))
        if remap_idx is None:
            print(f"  {key}: wrist {event_time} -> no motion frame")
            continue
        point = motion_track[remap_idx]
        print(
            f"  {key}: wrist {event_time} -> motion frame "
            f"{point.get('frame')} / {round(_safe_float(point.get('t'), 0.0))}ms"
        )


def _print_source_candidates(body_payload: Optional[Dict[str, Any]], labels: Dict[str, Any], tolerance_ms: float) -> None:
    print("\nsource-separated candidates")
    for source in ("left_wrist", "right_wrist", "weighted_midpoint"):
        candidate = _events_from_track(_single_source_track(body_payload, source))
        if not candidate.get("available"):
            print(f"  {source}: missing")
            continue
        events = candidate["events"]
        errors = _event_errors(events, labels)
        top_error = errors.get("topMs")
        impact_error = errors.get("impactMs")
        compact = " ".join(f"{key}={events.get(key)}" for key in EVENT_KEYS)
        print(f"  {source}: {_status(errors, tolerance_ms)} {compact}")
        print(
            "    "
            f"topError={'-' if top_error is None else f'{top_error:.0f}ms'} "
            f"impactError={'-' if impact_error is None else f'{impact_error:.0f}ms'}"
        )


def _print_first_backswing_experiment(
    body_payload: Optional[Dict[str, Any]],
    labels: Dict[str, Any],
    tolerance_ms: float,
) -> None:
    print("\nexperiment first-backswing-local candidates")
    tracks = {"merged": _wrist_track_from_body(body_payload)}
    for source in ("left_wrist", "right_wrist", "weighted_midpoint"):
        tracks[source] = _single_source_track(body_payload, source)
    best_name = None
    best_total = float("inf")
    best_events: Optional[Dict[str, Any]] = None
    for name, track in tracks.items():
        candidate = _first_backswing_local_events(track)
        if not candidate.get("available"):
            print(f"  {name}: missing")
            continue
        events = candidate["events"]
        errors = _event_errors(events, labels)
        total = sum(value for value in errors.values() if value is not None)
        compact = " ".join(f"{key}={events.get(key)}" for key in EVENT_KEYS)
        print(f"  {name}: {_status(errors, tolerance_ms)} {compact} totalError={total:.0f}ms")
        if total < best_total:
            best_name = name
            best_total = total
            best_events = events
    if best_events:
        _print_event_block(f"experiment first-backswing-local best={best_name}", best_events, labels, tolerance_ms)


def _print_phase_turnaround_experiment(
    body_payload: Optional[Dict[str, Any]],
    labels: Dict[str, Any],
    tolerance_ms: float,
) -> None:
    print("\nexperiment phase-turnaround candidates")
    tracks = {"merged": _wrist_track_from_body(body_payload)}
    for source in ("left_wrist", "right_wrist", "weighted_midpoint"):
        tracks[source] = _single_source_track(body_payload, source)
    best_name = None
    best_total = float("inf")
    best_events: Optional[Dict[str, Any]] = None
    for name, track in tracks.items():
        candidate = _phase_turnaround_events(track)
        if not candidate.get("available"):
            print(f"  {name}: missing")
            continue
        events = candidate["events"]
        errors = _event_errors(events, labels)
        total = sum(value for value in errors.values() if value is not None)
        compact = " ".join(f"{key}={events.get(key)}" for key in EVENT_KEYS)
        debug = candidate.get("debug", {})
        print(f"  {name}: {_status(errors, tolerance_ms)} {compact} totalError={total:.0f}ms")
        print(f"    debug={json.dumps(debug, ensure_ascii=False, sort_keys=True)}")
        if total < best_total:
            best_name = name
            best_total = total
            best_events = events
    if best_events:
        _print_event_block(f"experiment phase-turnaround best={best_name}", best_events, labels, tolerance_ms)


def _print_arm_angle_experiment(
    body_payload: Optional[Dict[str, Any]],
    labels: Dict[str, Any],
    tolerance_ms: float,
    start_ms: float = 0.0,
    title_suffix: str = "",
) -> None:
    suffix = f" {title_suffix}" if title_suffix else ""
    print(f"\nexperiment arm-angle{suffix}")
    best_name = None
    best_total = float("inf")
    best_events: Optional[Dict[str, Any]] = None
    for side in ("left", "right", "mid"):
        track = _arm_angle_track(body_payload, side)
        if start_ms > 0:
            track = _relative_track(track, start_ms)
        candidate = _angle_phase_events(track)
        if not candidate.get("available"):
            print(f"  {side}_arm: missing")
            continue
        absolute = _absolute_events(candidate["events"], start_ms)
        errors = _event_errors(absolute, labels)
        total = sum(value for value in errors.values() if value is not None)
        compact = " ".join(f"{key}={absolute.get(key)}" for key in EVENT_KEYS)
        debug = candidate.get("debug", {})
        print(f"  {side}_arm: {_status(errors, tolerance_ms)} {compact} totalError={total:.0f}ms")
        print(f"    debug={json.dumps(debug, ensure_ascii=False, sort_keys=True)}")
        if total < best_total:
            best_name = f"{side}_arm"
            best_total = total
            best_events = absolute
    if best_events:
        _print_event_block(f"experiment arm-angle best={best_name}{suffix}", best_events, labels, tolerance_ms)
        print(f"  totalError: {best_total:.0f}ms")


def _crop_track(track: list[Dict[str, Any]], start_ms: float) -> list[Dict[str, Any]]:
    return [point for point in track if _safe_float(point.get("t"), 0.0) >= start_ms]


def _crop_label_events(labels: Dict[str, Any], start_ms: float) -> Dict[str, Any]:
    cropped: Dict[str, Any] = {}
    for key in EVENT_KEYS:
        if labels.get(key) is None:
            continue
        cropped[key] = max(0.0, _safe_float(labels.get(key), 0.0) - start_ms)
    return cropped


def _relative_track(track: list[Dict[str, Any]], start_ms: float) -> list[Dict[str, Any]]:
    cropped = _crop_track(track, start_ms)
    relative: list[Dict[str, Any]] = []
    for point in cropped:
        relative.append({**point, "t": _safe_float(point.get("t"), 0.0) - start_ms})
    return relative


def _absolute_events(events: Dict[str, Any], start_ms: float) -> Dict[str, Any]:
    absolute: Dict[str, Any] = {}
    for key in EVENT_KEYS:
        if events.get(key) is None:
            absolute[key] = None
        else:
            absolute[key] = round(_safe_float(events.get(key), 0.0) + start_ms)
    return absolute


def _print_cropped_candidate(
    title: str,
    candidate: Dict[str, Any],
    labels: Dict[str, Any],
    start_ms: float,
    tolerance_ms: float,
) -> None:
    if not candidate.get("available"):
        print(f"  {title}: missing")
        return
    absolute = _absolute_events(candidate["events"], start_ms)
    errors = _event_errors(absolute, labels)
    total = sum(value for value in errors.values() if value is not None)
    compact = " ".join(f"{key}={absolute.get(key)}" for key in EVENT_KEYS)
    print(f"  {title}: {_status(errors, tolerance_ms)} {compact} totalError={total:.0f}ms")


def _print_label_address_cropped_experiment(
    body_payload: Optional[Dict[str, Any]],
    labels: Dict[str, Any],
    tolerance_ms: float,
) -> None:
    if labels.get("addressMs") is None:
        return
    start_ms = _safe_float(labels.get("addressMs"), 0.0)
    print(f"\nexperiment label-address-cropped startMs={round(start_ms)}")
    if start_ms <= 0:
        print("  skipped: label address is already 0ms")
        return

    tracks = {"merged": _wrist_track_from_body(body_payload)}
    for source in ("left_wrist", "right_wrist", "weighted_midpoint"):
        tracks[source] = _single_source_track(body_payload, source)

    cropped_labels = _crop_label_events(labels, start_ms)
    best_name = None
    best_total = float("inf")
    best_absolute: Optional[Dict[str, Any]] = None
    for name, track in tracks.items():
        relative = _relative_track(track, start_ms)
        print(f"  {name}: croppedPoints={len(relative)}")
        candidates = {
            "wrist-raw": _events_from_track(relative),
            "first-backswing": _first_backswing_local_events(relative),
            "phase-turnaround": _phase_turnaround_events(relative),
        }
        for candidate_name, candidate in candidates.items():
            title = f"{name}/{candidate_name}"
            _print_cropped_candidate(title, candidate, labels, start_ms, tolerance_ms)
            if not candidate.get("available"):
                continue
            absolute = _absolute_events(candidate["events"], start_ms)
            errors = _event_errors(absolute, labels)
            total = sum(value for value in errors.values() if value is not None)
            if total < best_total:
                best_name = title
                best_total = total
                best_absolute = absolute
    if best_absolute:
        _print_event_block(f"experiment label-address-cropped best={best_name}", best_absolute, labels, tolerance_ms)
        print(f"  totalError: {best_total:.0f}ms")
    _print_arm_angle_experiment(body_payload, labels, tolerance_ms, start_ms=start_ms, title_suffix="label-address-cropped")


def _print_body_diagnostics(
    body_payload: Optional[Dict[str, Any]],
    wrist_events: Dict[str, Any],
    labels: Dict[str, Any],
    tolerance_ms: float,
) -> None:
    if not wrist_events:
        return
    _print_source_candidates(body_payload, labels, tolerance_ms)
    _print_event_block("experiment preserve-wrist-time", wrist_events, labels, tolerance_ms)
    _print_first_backswing_experiment(body_payload, labels, tolerance_ms)
    _print_phase_turnaround_experiment(body_payload, labels, tolerance_ms)
    _print_arm_angle_experiment(body_payload, labels, tolerance_ms)
    _print_feature_probe_experiment(body_payload, labels, tolerance_ms)
    _print_feature_vote_experiment(body_payload, labels, tolerance_ms)
    _print_feature_vote_gated_experiment(body_payload, labels, tolerance_ms)
    _print_feature_vote_band_sweep_experiment(body_payload, labels, tolerance_ms)
    _print_feature_sequence_experiment(body_payload, labels, tolerance_ms)
    _print_body_event_selector_experiment(body_payload, labels, tolerance_ms)
    _print_label_address_cropped_experiment(body_payload, labels, tolerance_ms)


def _warn_label_anomalies(labels: Dict[str, Any], body_payload: Optional[Dict[str, Any]]) -> None:
    frames = body_payload.get("frames") if isinstance(body_payload, dict) else None
    if not isinstance(frames, list) or not frames:
        return
    times = [
        _safe_float(frame.get("timeMs"), idx * 33.33)
        for idx, frame in enumerate(frames)
        if isinstance(frame, dict)
    ]
    if not times:
        return
    max_time = max(times)
    for key in EVENT_KEYS:
        label_time = labels.get(key)
        if label_time is not None and _safe_float(label_time, 0.0) > max_time + 500.0:
            print(f"label warning: {key}={label_time} exceeds body max time {round(max_time)}ms")


def _format_events_compact(events: Dict[str, Any]) -> str:
    return ",".join(f"{key.replace('Ms', '')}={events.get(key)}" for key in EVENT_KEYS)


def print_selector_summary(fixtures: list[Path]) -> int:
    print("fixture\tviewpoint\tmethod\tstatus\ttotalErrorMs\tevents")
    exit_code = 0
    for fixture_path in fixtures:
        try:
            fixture = _load_json(fixture_path)
            labels = fixture.get("labels") if isinstance(fixture.get("labels"), dict) else {}
            tolerance_ms = _safe_float(fixture.get("toleranceMs"), 80.0)
            artifacts = fixture.get("artifacts") if isinstance(fixture.get("artifacts"), dict) else {}
            body_path = _resolve_path(artifacts.get("bodyPath"), fixture_path.parent)
            viewpoint = str(fixture.get("viewpoint") or "unknown")
            if not body_path or not body_path.exists():
                print(f"{fixture_path.stem}\t{viewpoint}\tmissing_body\tmissing\t-\t-")
                continue
            body_payload = _load_json(body_path)
            result = _body_event_selector_result(body_payload)
            if not result.get("available"):
                print(f"{fixture_path.stem}\t{viewpoint}\tmissing\tmissing\t-\t-")
                continue
            events = result.get("events", {})
            errors = _event_errors(events, labels)
            status = _status(errors, tolerance_ms)
            total = sum(value for value in errors.values() if value is not None)
            if status != "pass":
                exit_code = max(exit_code, 1)
            print(
                f"{fixture_path.stem}\t{viewpoint}\t{result.get('method')}\t{status}\t"
                f"{total:.0f}\t{_format_events_compact(events)}"
            )
        except Exception as exc:
            exit_code = max(exit_code, 1)
            print(f"{fixture_path.stem}\t-\terror:{type(exc).__name__}\tfail\t-\t{exc}")
    return exit_code


def replay_fixture(fixture_path: Path, force: bool, allow_missing: bool, diagnostics: bool) -> int:
    fixture = _load_json(fixture_path)
    job_id = str(fixture.get("jobId") or fixture_path.stem)
    labels = fixture.get("labels") if isinstance(fixture.get("labels"), dict) else {}
    tolerance_ms = _safe_float(fixture.get("toleranceMs"), 80.0)
    artifacts = fixture.get("artifacts") if isinstance(fixture.get("artifacts"), dict) else {}

    meta_path = _resolve_path(artifacts.get("metaPath"), fixture_path.parent)
    body_path = _resolve_path(artifacts.get("bodyPath"), fixture_path.parent)

    print(f"fixture: {fixture_path}")
    print(f"jobId: {job_id}")
    print(f"toleranceMs: {tolerance_ms:.0f}")

    body_payload: Optional[Dict[str, Any]] = None
    if body_path and body_path.exists():
        body_payload = _load_json(body_path)
    elif body_path:
        print(f"body: missing ({body_path})")
    _warn_label_anomalies(labels, body_payload)

    wrist = _wrist_events(body_payload)
    if wrist.get("available"):
        _print_event_block("wrist raw", wrist["events"], labels, tolerance_ms)
        print(f"  wristPoints: {wrist.get('wristPoints')}")
        print(f"  wristSources: {json.dumps(wrist.get('sources', {}), ensure_ascii=False, sort_keys=True)}")
    else:
        print("\nwrist raw: missing")

    if not meta_path or not meta_path.exists():
        missing = meta_path if meta_path else "not configured"
        print(f"\nservice replay: missing meta ({missing})")
        if diagnostics and wrist.get("available"):
            _print_body_diagnostics(body_payload, wrist["events"], labels, tolerance_ms)
        return 0 if allow_missing else 2

    meta_payload = _load_json(meta_path)
    result = analyze_meta(meta_payload, job_id=job_id, force=force, body_path=str(body_path) if body_path else None)
    service_events = result.get("events") if isinstance(result.get("events"), dict) else {}
    _print_event_block("service replay", service_events, labels, tolerance_ms)
    print(f"  eventSource: {result.get('debug', {}).get('eventSource') if isinstance(result.get('debug'), dict) else None}")
    if isinstance(result.get("debug"), dict):
        print(f"  eventIndices: {json.dumps(result['debug'].get('eventIndices', {}), ensure_ascii=False, sort_keys=True)}")
    print(f"  summary: {result.get('summary')}")
    if diagnostics and wrist.get("available"):
        wrist_events = wrist["events"]
        _print_delta_block("service - wrist raw", service_events, wrist_events)
        _print_remap_diagnostics(meta_payload, wrist_events)
        _print_body_diagnostics(body_payload, wrist_events, labels, tolerance_ms)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay and score golf swing event segmentation fixtures.")
    parser.add_argument(
        "fixtures",
        nargs="*",
        type=Path,
        default=sorted((ROOT / "fixtures" / "event_labels").glob("*.json")),
        help="Fixture JSON files. Defaults to fixtures/event_labels/*.json.",
    )
    parser.add_argument("--no-force", action="store_true", help="Use normal NOT_SWING failures instead of force mode.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing Pi artifacts without failing.")
    parser.add_argument("--diagnostics", action="store_true", help="Print remap and source-separated candidate diagnostics.")
    parser.add_argument("--selector-summary", action="store_true", help="Print one-line body event selector results per fixture.")
    args = parser.parse_args()

    if not args.fixtures:
        print("No fixtures found. Add JSON files under fixtures/event_labels/ or pass paths explicitly.")
        return 2

    if args.selector_summary:
        return print_selector_summary(list(args.fixtures))

    exit_code = 0
    for idx, fixture in enumerate(args.fixtures):
        if idx:
            print("\n" + "=" * 72)
        try:
            exit_code = max(
                exit_code,
                replay_fixture(
                    fixture,
                    force=not args.no_force,
                    allow_missing=args.allow_missing,
                    diagnostics=args.diagnostics,
                ),
            )
        except Exception as exc:  # Keep batch replay useful while algorithms are experimental.
            exit_code = max(exit_code, 1)
            print(f"fixture: {fixture}")
            print(f"error: {type(exc).__name__}: {exc}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
