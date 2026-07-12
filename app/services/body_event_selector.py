"""Pose-led swing phase selection with club and ROI-motion evidence.

Pose timing remains the primary coordinate system.  Club tracks and dynamic
ROI flow are optional evidence sources, so sparse club detections do not force
event times to be remapped away from the body sequence.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from app.services.swing_phase_decoder import decode_swing_phases

EVENT_KEYS = ("addressMs", "topMs", "impactMs", "finishMs")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        num = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return num if math.isfinite(num) else default


def _keypoint_xyc(frame: dict, name: str) -> Optional[tuple[float, float, float]]:
    keypoints = frame.get("keypoints")
    if not isinstance(keypoints, dict):
        return None
    raw = keypoints.get(name)
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    conf = _safe_float(raw[2], 0.0) if len(raw) >= 3 else 1.0
    return _safe_float(raw[0], 0.0), _safe_float(raw[1], 0.0), conf


def _midpoint(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, min(a[2], b[2]))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(a[1] - b[1], a[0] - b[0])


def body_feature_tracks(body_payload: Optional[Dict[str, Any]]) -> Dict[str, list[Dict[str, Any]]]:
    frames = body_payload.get("frames") if isinstance(body_payload, dict) else None
    if not isinstance(frames, list):
        return {}

    tracks: Dict[str, list[Dict[str, Any]]] = {}

    def add(name: str, t: float, value: float, conf: float) -> None:
        if conf < 0.02:
            return
        tracks.setdefault(name, []).append({"t": t, "value": value, "conf": conf})

    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        t = _safe_float(frame.get("timeMs"), idx * 33.33)
        points: Dict[str, tuple[float, float, float]] = {}
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
        ):
            point = _keypoint_xyc(frame, name)
            if point:
                points[name] = point
                add(f"{name}_x", t, point[0], point[2])
                add(f"{name}_y", t, point[1], point[2])

        mids: Dict[str, tuple[float, float, float]] = {}
        pairs = {
            "shoulder_mid": ("left_shoulder", "right_shoulder"),
            "hip_mid": ("left_hip", "right_hip"),
            "wrist_mid": ("left_wrist", "right_wrist"),
            "ankle_mid": ("left_ankle", "right_ankle"),
        }
        for name, (left_name, right_name) in pairs.items():
            left = points.get(left_name)
            right = points.get(right_name)
            if not left or not right:
                continue
            mid = _midpoint(left, right)
            mids[name] = mid
            add(f"{name}_x", t, mid[0], mid[2])
            add(f"{name}_y", t, mid[1], mid[2])

        if "left_shoulder" in points and "right_shoulder" in points:
            add(
                "shoulder_width",
                t,
                _distance(
                    (points["left_shoulder"][0], points["left_shoulder"][1]),
                    (points["right_shoulder"][0], points["right_shoulder"][1]),
                ),
                min(points["left_shoulder"][2], points["right_shoulder"][2]),
            )
        if "left_hip" in points and "right_hip" in points:
            add(
                "hip_width",
                t,
                _distance(
                    (points["left_hip"][0], points["left_hip"][1]),
                    (points["right_hip"][0], points["right_hip"][1]),
                ),
                min(points["left_hip"][2], points["right_hip"][2]),
            )
        if "left_wrist" in points and "right_wrist" in points:
            add(
                "wrist_gap",
                t,
                _distance(
                    (points["left_wrist"][0], points["left_wrist"][1]),
                    (points["right_wrist"][0], points["right_wrist"][1]),
                ),
                min(points["left_wrist"][2], points["right_wrist"][2]),
            )

        span_pairs = {
            "left_arm_span": ("left_wrist", "left_shoulder"),
            "right_arm_span": ("right_wrist", "right_shoulder"),
            "left_elbow_span": ("left_elbow", "left_shoulder"),
            "right_elbow_span": ("right_elbow", "right_shoulder"),
        }
        for name, (a_name, b_name) in span_pairs.items():
            a = points.get(a_name)
            b = points.get(b_name)
            if a and b:
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
            if a and b:
                add(name, t, _angle_between((a[0], a[1]), (b[0], b[1])), min(a[2], b[2]))

    for track in tracks.values():
        track.sort(key=lambda point: _safe_float(point.get("t"), 0.0))
    return tracks


def probe_candidate_times(track: list[Dict[str, Any]], kind: str) -> list[float]:
    if len(track) < 3:
        return []
    values = [_safe_float(point.get("value"), 0.0) for point in track]
    times = [_safe_float(point.get("t"), 0.0) for point in track]
    if kind == "global_min":
        return [times[min(range(len(values)), key=lambda idx: values[idx])]]
    if kind == "global_max":
        return [times[max(range(len(values)), key=lambda idx: values[idx])]]
    if kind in {"local_min", "local_max"}:
        candidates: list[float] = []
        for idx in range(1, len(values) - 1):
            if kind == "local_min" and values[idx] <= values[idx - 1] and values[idx] <= values[idx + 1]:
                candidates.append(times[idx])
            if kind == "local_max" and values[idx] >= values[idx - 1] and values[idx] >= values[idx + 1]:
                candidates.append(times[idx])
        return candidates
    return []


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


STATE_MACHINE_METHOD_BIAS = {
    "feature-vote-early-top-cluster": 0.25,
    "feature-vote-early": 0.35,
    "feature-sequence": 0.0,
    "feature-vote-offset-fallback": 0.15,
}


def state_machine_score(events: Dict[str, Any], debug: Optional[Dict[str, Any]] = None, method: str = "") -> tuple[float, list[str]]:
    """Score whether event times obey a plausible swing phase sequence.

    This is not an optical-flow state machine yet. It is the first guard layer:
    candidate event sequences still come from pose feature votes, but they must
    pass Ready -> Backswing -> Top -> Downswing -> Impact Candidate -> Finish
    timing constraints before the selector trusts them.
    """

    debug = debug or {}
    address = _safe_float(events.get("addressMs"), 0.0)
    top = _safe_float(events.get("topMs"), address)
    impact = _safe_float(events.get("impactMs"), top)
    finish = _safe_float(events.get("finishMs"), impact)
    top_gap = top - address
    down_gap = impact - top
    finish_gap = finish - impact
    reasons: list[str] = []

    if top <= address:
        reasons.append("top_not_after_address")
    if impact <= top:
        reasons.append("impact_not_after_top")
    if finish <= impact:
        reasons.append("finish_not_after_impact")
    if reasons:
        return float("inf"), reasons

    score = STATE_MACHINE_METHOD_BIAS.get(method, 0.2)

    # Very early tops are usually the first noisy cluster, not a real top. If a
    # clip genuinely starts near top, we still require refinement evidence.
    top_refined = bool(debug.get("topRefined"))
    if top_gap < 120.0 and not top_refined:
        reasons.append("top_gap_too_short_without_refinement")
        score += 6.0
    elif top_gap < 160.0:
        reasons.append("top_gap_short")
        score += 1.2
    elif top_gap > 950.0:
        reasons.append("top_gap_too_late")
        score += 2.0

    if down_gap < 80.0:
        reasons.append("downswing_gap_too_short")
        score += 5.0
    elif down_gap > 620.0:
        reasons.append("downswing_gap_too_long")
        score += 2.0

    if finish_gap < 120.0:
        reasons.append("finish_gap_too_short")
        score += 4.0
    elif finish_gap > 900.0:
        reasons.append("finish_gap_too_long")
        score += 1.5

    # Target windows are intentionally broad. The purpose is to rank plausible
    # sequences, not force a single tempo model across all players.
    target_top_gap = 340.0
    target_down_gap = 190.0
    target_finish_gap = 330.0
    score += abs(top_gap - target_top_gap) / 360.0
    score += abs(down_gap - target_down_gap) / 320.0
    score += abs(finish_gap - target_finish_gap) / 420.0

    return score, reasons


def cluster_candidate_times(candidates: list[tuple[float, float, str]], band_ms: float = 70.0) -> list[Dict[str, Any]]:
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
        result.append({"t": center, "weight": weight, "count": len(cluster), "sources": sorted({item[2] for item in cluster})})
    result.sort(key=lambda item: _safe_float(item.get("t"), 0.0))
    return result


def feature_vote_clusters(body_payload: Optional[Dict[str, Any]], band_ms: float = 70.0) -> list[Dict[str, Any]]:
    feature_tracks = body_feature_tracks(body_payload)
    if not feature_tracks:
        return []
    all_times = [_safe_float(point.get("t"), 0.0) for track in feature_tracks.values() for point in track]
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
        for candidate_t in probe_candidate_times(track, kind):
            if candidate_t < address_t + 20.0 or candidate_t > address_t + min(2300.0, duration):
                continue
            candidates.append((candidate_t, weight, f"{feature_name}/{kind}"))
    return cluster_candidate_times(candidates, band_ms=band_ms)


def feature_vote_events(body_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    clusters = feature_vote_clusters(body_payload)
    if len(clusters) < 2:
        return {"available": False, "clusters": clusters}
    address_t = min(_safe_float(cluster.get("t"), 0.0) for cluster in clusters)
    last_t = max(_safe_float(cluster.get("t"), 0.0) for cluster in clusters)
    duration = max(1.0, last_t - address_t)
    top_window_end = address_t + min(850.0, duration * 0.45)
    top_candidates = [cluster for cluster in clusters if address_t + 20.0 <= _safe_float(cluster.get("t"), 0.0) <= top_window_end]
    if not top_candidates:
        return {"available": False, "clusters": clusters}
    max_top_weight = max(_safe_float(cluster.get("weight"), 0.0) for cluster in top_candidates)
    strong_top = [cluster for cluster in top_candidates if _safe_float(cluster.get("weight"), 0.0) >= max_top_weight * 0.55]
    top_cluster = min(strong_top, key=lambda cluster: _safe_float(cluster.get("t"), 0.0))
    top_t = _safe_float(top_cluster.get("t"), 0.0)
    impact_candidates = [cluster for cluster in clusters if top_t + 90.0 <= _safe_float(cluster.get("t"), 0.0) <= top_t + 650.0]
    impact_cluster = max(
        impact_candidates,
        key=lambda cluster: (_safe_float(cluster.get("weight"), 0.0), -abs((_safe_float(cluster.get("t"), 0.0) - top_t) - 180.0)),
        default=None,
    )
    impact_t = _safe_float(impact_cluster.get("t"), 0.0) if impact_cluster else None
    finish_cluster = None
    if impact_t is not None:
        finish_candidates = [cluster for cluster in clusters if impact_t + 150.0 <= _safe_float(cluster.get("t"), 0.0) <= impact_t + 780.0]
        finish_cluster = max(
            finish_candidates,
            key=lambda cluster: (_safe_float(cluster.get("weight"), 0.0), -abs((_safe_float(cluster.get("t"), 0.0) - impact_t) - 330.0)),
            default=None,
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


def feature_vote_events_from_clusters(clusters: list[Dict[str, Any]], start_t: float, *, use_offset_model: bool = False) -> Dict[str, Any]:
    if len(clusters) < 2:
        return {"available": False}
    top_candidates = [cluster for cluster in clusters if start_t + 20.0 <= _safe_float(cluster.get("t"), 0.0) <= start_t + 760.0]
    if not top_candidates:
        return {"available": False}
    if use_offset_model:
        target_offset = 70.0 if start_t < 100.0 else 350.0
        top_cluster = min(top_candidates, key=lambda cluster: (abs((_safe_float(cluster.get("t"), 0.0) - start_t) - target_offset), -_safe_float(cluster.get("weight"), 0.0)))
    else:
        max_top_weight = max(_safe_float(cluster.get("weight"), 0.0) for cluster in top_candidates)
        strong_top = [cluster for cluster in top_candidates if _safe_float(cluster.get("weight"), 0.0) >= max_top_weight * 0.55]
        top_cluster = min(strong_top, key=lambda cluster: _safe_float(cluster.get("t"), 0.0))
    top_t = _safe_float(top_cluster.get("t"), 0.0)
    impact_candidates = [cluster for cluster in clusters if top_t + 90.0 <= _safe_float(cluster.get("t"), 0.0) <= top_t + 650.0]
    impact_cluster = min(
        impact_candidates,
        key=lambda cluster: (abs((_safe_float(cluster.get("t"), 0.0) - top_t) - 180.0), -_safe_float(cluster.get("weight"), 0.0)),
        default=None,
    )
    impact_t = _safe_float(impact_cluster.get("t"), 0.0) if impact_cluster else None
    finish_cluster = None
    if impact_t is not None:
        finish_candidates = [cluster for cluster in clusters if impact_t + 150.0 <= _safe_float(cluster.get("t"), 0.0) <= impact_t + 780.0]
        finish_cluster = min(
            finish_candidates,
            key=lambda cluster: (abs((_safe_float(cluster.get("t"), 0.0) - impact_t) - 330.0), -_safe_float(cluster.get("weight"), 0.0)),
            default=None,
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


def feature_sequence_events(feature_name: str, kind: str, feature_tracks: Dict[str, list[Dict[str, Any]]], start_t: float) -> Dict[str, Any]:
    track = feature_tracks.get(feature_name)
    if not track:
        return {"available": False}
    candidates = [candidate_t for candidate_t in probe_candidate_times(track, kind) if candidate_t >= start_t + 20.0]
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
        "events": {"addressMs": round(start_t), "topMs": round(top_t), "impactMs": round(impact_t), "finishMs": round(finish_t)},
        "debug": {"feature": f"{feature_name}/{kind}", "candidateCount": len(candidates), "startMs": round(start_t)},
    }


def sequence_internal_score(events: Dict[str, Any], debug: Dict[str, Any]) -> float:
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


def feature_sequence_ranked(body_payload: Optional[Dict[str, Any]]) -> list[tuple[float, Dict[str, Any], Dict[str, Any]]]:
    feature_tracks = body_feature_tracks(body_payload)
    clusters = feature_vote_clusters(body_payload, band_ms=70.0)
    if not feature_tracks or not clusters:
        return []
    ranked: list[tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for cluster in clusters[:10]:
        start_t = _safe_float(cluster.get("t"), 0.0)
        for feature_name, kind, _weight in FEATURE_VOTE_RULES:
            candidate = feature_sequence_events(feature_name, kind, feature_tracks, start_t)
            if not candidate.get("available"):
                continue
            events = candidate["events"]
            debug = candidate.get("debug", {})
            ranked.append((sequence_internal_score(events, debug), events, debug))
    ranked.sort(key=lambda item: item[0])
    return ranked


def early_top_cluster_candidate(clusters: list[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if not clusters:
        return None, {}
    first_cluster = clusters[0]
    first_t = _safe_float(first_cluster.get("t"), 9999.0)
    first_weight = _safe_float(first_cluster.get("weight"), 0.0)
    if first_t > 120.0 or first_weight < 9.0:
        return None, {}

    impact_candidates = [cluster for cluster in clusters if first_t + 300.0 <= _safe_float(cluster.get("t"), 0.0) <= first_t + 540.0]
    impact_cluster = max(impact_candidates, key=lambda cluster: _safe_float(cluster.get("weight"), 0.0), default=None)
    impact_t = _safe_float(impact_cluster.get("t"), 0.0) if impact_cluster else None
    if impact_t is None:
        return None, {}

    top_cluster = first_cluster
    refined_top = False
    refined_top_candidates = [
        cluster
        for cluster in clusters
        if first_t + 120.0 <= _safe_float(cluster.get("t"), 0.0) <= impact_t - 90.0
        and _safe_float(cluster.get("weight"), 0.0) >= first_weight * 1.1
    ]
    if refined_top_candidates:
        top_cluster = max(refined_top_candidates, key=lambda cluster: _safe_float(cluster.get("weight"), 0.0))
        refined_top = True

    finish_candidates = [cluster for cluster in clusters if impact_t + 160.0 <= _safe_float(cluster.get("t"), 0.0) <= impact_t + 360.0]
    finish_cluster = max(finish_candidates, key=lambda cluster: _safe_float(cluster.get("weight"), 0.0), default=None)
    if not finish_cluster:
        return None, {}
    finish_t = _safe_float(finish_cluster.get("t"), 0.0)
    top_t = _safe_float(top_cluster.get("t"), first_t)

    original_impact_t = impact_t
    impact_refined = False
    if refined_top:
        ratio_impact_t = top_t + (finish_t - top_t) * 0.48
        latest_safe_impact_t = finish_t - 120.0
        if impact_t < ratio_impact_t <= latest_safe_impact_t:
            impact_t = ratio_impact_t
            impact_refined = True

    return {
        "addressMs": 0,
        "topMs": round(top_t),
        "impactMs": round(impact_t),
        "finishMs": round(finish_t),
    }, {
        "topWeight": round(_safe_float(top_cluster.get("weight"), first_weight), 2),
        "impactWeight": round(_safe_float(impact_cluster.get("weight"), 0.0), 2),
        "finishWeight": round(_safe_float(finish_cluster.get("weight"), 0.0), 2),
        "originalTopMs": round(first_t),
        "originalTopWeight": round(first_weight, 2),
        "originalImpactMs": round(original_impact_t),
        "impactRefined": impact_refined,
        "topRefined": refined_top,
    }


def _motion_value(frame: Dict[str, Any]) -> float:
    """Read dynamic ROI motion from a body artifact without requiring it yet."""
    raw = frame.get("roiMotion") or frame.get("roi_motion")
    if not isinstance(raw, dict):
        return 0.0
    values: list[float] = []
    for name in ("upper", "torso", "lower"):
        value = raw.get(name)
        if isinstance(value, dict):
            values.append(_safe_float(value.get("magnitude"), 0.0))
        else:
            values.append(_safe_float(value, 0.0))
    return sum(values) / len(values) if values else 0.0


def _club_speed_at(times: list[float], club_track: Optional[list[Dict[str, Any]]]) -> list[float]:
    if not club_track:
        return [0.0 for _ in times]
    ordered = sorted((point for point in club_track if isinstance(point, dict)), key=lambda point: _safe_float(point.get("t"), 0.0))
    if len(ordered) < 2:
        return [0.0 for _ in times]
    club_speeds: list[tuple[float, float]] = []
    for idx in range(1, len(ordered)):
        before, after = ordered[idx - 1], ordered[idx]
        dt = max(1.0, _safe_float(after.get("t"), 0.0) - _safe_float(before.get("t"), 0.0))
        distance = math.hypot(
            _safe_float(after.get("x"), 0.0) - _safe_float(before.get("x"), 0.0),
            _safe_float(after.get("y"), 0.0) - _safe_float(before.get("y"), 0.0),
        )
        club_speeds.append((_safe_float(after.get("t"), 0.0), distance / dt))
    return [
        min(club_speeds, key=lambda item: abs(item[0] - time_ms))[1]
        for time_ms in times
    ]


def _relative_activity(values: list[float]) -> list[float]:
    """Normalise motion above its own quiet baseline, not above zero."""
    if not values:
        return []
    ordered = sorted(values)
    low = ordered[int((len(ordered) - 1) * 0.2)]
    high = ordered[int((len(ordered) - 1) * 0.9)]
    if high - low < 1e-7:
        return [0.0 for _ in values]
    return [max(0.0, min(1.0, (value - low) / (high - low))) for value in values]


def phase_evidence_from_body(
    body_payload: Optional[Dict[str, Any]], club_track: Optional[list[Dict[str, Any]]] = None
) -> list[Dict[str, Any]]:
    """Build phase evidence from pose-relative velocity, club, and ROI motion.

    Missing sources simply contribute zero.  This keeps the decoder useful for
    pose-only artifacts today while making ``roiMotion`` and club input first
    class evidence as soon as their producers are available.
    """
    frames = body_payload.get("frames") if isinstance(body_payload, dict) else None
    if not isinstance(frames, list):
        return []
    samples: list[Dict[str, Any]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        left = _keypoint_xyc(frame, "left_wrist")
        right = _keypoint_xyc(frame, "right_wrist")
        if not left and not right:
            continue
        # A partially occluded wrist must not erase a confident opposite hand.
        # Using the midpoint confidence (the lower of the two confidences)
        # previously dropped the entire address/early-backswing window when one
        # wrist was hidden, causing the phase decoder to begin mid-swing.
        if left and right and left[2] >= 0.15 and right[2] >= 0.15:
            wrist = _midpoint(left, right)
        else:
            candidates = [point for point in (left, right) if point and point[2] >= 0.15]
            wrist = max(candidates, key=lambda point: point[2]) if candidates else None
        if not wrist or wrist[2] < 0.15:
            continue
        samples.append(
            {
                "timeMs": _safe_float(frame.get("timeMs"), index * 33.333),
                "x": wrist[0],
                "y": wrist[1],
                "roiMotion": _motion_value(frame),
            }
        )
    if len(samples) < 7:
        return []

    pose_speeds = [0.0]
    directions: list[tuple[float, float]] = [(0.0, 0.0)]
    for index in range(1, len(samples)):
        before, after = samples[index - 1], samples[index]
        dt = max(1.0, _safe_float(after["timeMs"]) - _safe_float(before["timeMs"]))
        dx = (_safe_float(after["x"]) - _safe_float(before["x"])) / dt
        dy = (_safe_float(after["y"]) - _safe_float(before["y"])) / dt
        pose_speeds.append(math.hypot(dx, dy))
        directions.append((dx, dy))
    club_speeds = _club_speed_at([_safe_float(sample["timeMs"]) for sample in samples], club_track)
    pose_activity_values = _relative_activity(pose_speeds)
    club_activity_values = _relative_activity(club_speeds)
    roi_activity_values = _relative_activity([_safe_float(sample["roiMotion"]) for sample in samples])

    evidence: list[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        pose_activity = pose_activity_values[index]
        club_activity = club_activity_values[index]
        roi_activity = roi_activity_values[index]
        activity = max(pose_activity, club_activity * 0.9, roi_activity * 0.7)
        reversal = 0.0
        if 0 < index < len(directions) - 1:
            previous = next(
                (direction for direction in reversed(directions[:index]) if math.hypot(*direction) > 1e-7),
                (0.0, 0.0),
            )
            following = next(
                (direction for direction in directions[index + 1 :] if math.hypot(*direction) > 1e-7),
                (0.0, 0.0),
            )
            previous_length = math.hypot(*previous)
            following_length = math.hypot(*following)
            if previous_length > 1e-7 and following_length > 1e-7:
                reversal = max(0.0, -((previous[0] * following[0] + previous[1] * following[1]) / (previous_length * following_length)))
        still = 1.0 - activity
        evidence.append(
            {
                "timeMs": round(_safe_float(sample["timeMs"])),
                "scores": {
                    "ready": still,
                    "backswing": activity * (1.0 - reversal * 0.35),
                    "top": min(1.0, reversal * 0.85 + still * 0.25),
                    "downswing": max(pose_activity, club_activity, roi_activity * 0.65),
                    "impact_candidate": max(club_activity, pose_activity * 0.65),
                    "follow_through": activity * (1.0 - reversal * 0.2),
                    "finish": still,
                },
                "sources": {
                    "pose": round(pose_activity, 3),
                    "club": round(club_activity, 3),
                    "roiMotion": round(roi_activity, 3),
                    "directionReversal": round(reversal, 3),
                },
            }
        )
    return evidence


def select_body_events(
    body_payload: Optional[Dict[str, Any]], club_track: Optional[list[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    phase_evidence = phase_evidence_from_body(body_payload, club_track)
    decoded = decode_swing_phases(phase_evidence)
    if decoded.get("available"):
        return {
            "available": True,
            "method": "forward-phase-decoder",
            "events": decoded["events"],
            "confidence": decoded.get("confidence", 0.0),
            "debug": {
                **(decoded.get("debug") if isinstance(decoded.get("debug"), dict) else {}),
                "phasePath": decoded.get("phasePath", []),
                "evidenceSources": ["pose", "club", "roiMotion"],
            },
        }

    sequence_ranked = feature_sequence_ranked(body_payload)
    vote = feature_vote_events(body_payload)
    candidates: list[tuple[float, str, Dict[str, Any], Dict[str, Any]]] = []

    def add_candidate(method: str, events: Dict[str, Any], debug: Optional[Dict[str, Any]] = None) -> None:
        if not events:
            return
        score, reasons = state_machine_score(events, debug, method)
        next_debug = {**(debug or {})}
        next_debug["stateMachineScore"] = round(score, 3) if math.isfinite(score) else None
        next_debug["stateMachineReasons"] = reasons
        candidates.append((score, method, events, next_debug))

    if vote.get("available"):
        events = vote.get("events", {})
        clusters = vote.get("clusters") if isinstance(vote.get("clusters"), list) else []
        if clusters:
            early_candidate_events, early_candidate_debug = early_top_cluster_candidate(clusters)
            if early_candidate_events:
                add_candidate("feature-vote-early-top-cluster", early_candidate_events, early_candidate_debug)
        address_t = _safe_float(events.get("addressMs"), 9999.0)
        top_t = _safe_float(events.get("topMs"), 9999.0)
        impact_t = _safe_float(events.get("impactMs"), 9999.0)
        finish_t = _safe_float(events.get("finishMs"), 9999.0)
        top_weight = _safe_float(vote.get("debug", {}).get("topWeight"), 0.0)
        if address_t <= 120.0 and top_t <= 260.0 and impact_t <= 760.0 and finish_t <= 1050.0 and top_weight >= 9.0:
            add_candidate("feature-vote-early", events, vote.get("debug", {}))

    if sequence_ranked:
        sequence_score, sequence_events, sequence_debug = sequence_ranked[0]
        if str(sequence_debug.get("feature")) == "left_elbow_y/local_max" and _safe_float(sequence_debug.get("startMs"), 0.0) < 300.0:
            shoulder_alternative = next(
                (item for item in sequence_ranked if str(item[2].get("feature")) == "shoulder_width/local_max" and item[0] <= sequence_score + 0.55),
                None,
            )
            if shoulder_alternative:
                shoulder_score = shoulder_alternative[0]
                shoulder_alternatives = [
                    item
                    for item in sequence_ranked
                    if str(item[2].get("feature")) == "shoulder_width/local_max" and item[0] <= shoulder_score + 0.2
                ]
                sequence_score, sequence_events, sequence_debug = min(
                    shoulder_alternatives,
                    key=lambda item: _safe_float(item[2].get("startMs"), 9999.0),
                )
        start_t = _safe_float(sequence_events.get("addressMs"), 0.0)
        if start_t >= 120.0 and sequence_score <= 2.6:
            add_candidate("feature-sequence", sequence_events, {"score": round(sequence_score, 3), **sequence_debug})

    clusters = vote.get("clusters") if isinstance(vote.get("clusters"), list) else []
    for cluster in clusters[:10]:
        start_t = _safe_float(cluster.get("t"), 0.0)
        candidate = feature_vote_events_from_clusters(clusters, start_t, use_offset_model=True)
        if not candidate.get("available"):
            continue
        events = candidate["events"]
        top_gap = _safe_float(events.get("topMs"), 0.0) - _safe_float(events.get("addressMs"), 0.0)
        impact_gap = _safe_float(events.get("impactMs"), 0.0) - _safe_float(events.get("topMs"), 0.0)
        finish_gap = _safe_float(events.get("finishMs"), 0.0) - _safe_float(events.get("impactMs"), 0.0)
        legacy_score = abs(top_gap - 360.0) / 260.0 + abs(impact_gap - 170.0) / 220.0 + abs(finish_gap - 320.0) / 320.0
        add_candidate(
            "feature-vote-offset-fallback",
            events,
            {"score": round(legacy_score, 3), **candidate.get("debug", {})},
        )

    finite_candidates = [candidate for candidate in candidates if math.isfinite(candidate[0])]
    if not finite_candidates:
        return {"available": False, "sequenceRanked": sequence_ranked, "vote": vote}
    state_ranked = sorted(finite_candidates, key=lambda item: item[0])
    _score, candidate_name, candidate_events, candidate_debug = state_ranked[0]
    return {
        "available": True,
        "method": candidate_name,
        "events": candidate_events,
        "debug": candidate_debug,
        "stateMachineRanked": [
            {
                "score": round(score, 3),
                "method": method,
                "events": events,
                "debug": debug,
            }
            for score, method, events, debug in state_ranked[:8]
        ],
        "sequenceRanked": sequence_ranked,
        "vote": vote,
    }


def recommendation_for_selector(viewpoint: str, selector_result: Dict[str, Any], status: Optional[str] = None) -> str:
    if viewpoint == "face_on":
        return "use_face_on_phase_turnaround"
    if not selector_result.get("available"):
        return "selector_missing"
    if status == "pass":
        return "use_body_selector"
    return "low_confidence_body_selector"
