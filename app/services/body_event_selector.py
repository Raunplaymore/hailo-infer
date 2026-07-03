"""Pose-based body event selector for swing phase timing.

This module is intentionally independent from club motion tracks. It returns
event times from body pose features so callers can preserve pose timing instead
of remapping to sparse club detections.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

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


def select_body_events(body_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    sequence_ranked = feature_sequence_ranked(body_payload)
    vote = feature_vote_events(body_payload)
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
                impact_candidates = [cluster for cluster in clusters if first_t + 300.0 <= _safe_float(cluster.get("t"), 0.0) <= first_t + 540.0]
                impact_cluster = max(impact_candidates, key=lambda cluster: _safe_float(cluster.get("weight"), 0.0), default=None)
                impact_t = _safe_float(impact_cluster.get("t"), 0.0) if impact_cluster else None
                finish_cluster = None
                if impact_t is not None:
                    finish_candidates = [cluster for cluster in clusters if impact_t + 160.0 <= _safe_float(cluster.get("t"), 0.0) <= impact_t + 360.0]
                    finish_cluster = max(finish_candidates, key=lambda cluster: _safe_float(cluster.get("weight"), 0.0), default=None)
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
        if candidate_events is None and address_t <= 120.0 and top_t <= 260.0 and impact_t <= 760.0 and finish_t <= 1050.0 and top_weight >= 9.0:
            candidate_name = "feature-vote-early"
            candidate_events = events
            candidate_debug = vote.get("debug", {})

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
        if candidate_events is None and start_t >= 120.0 and sequence_score <= 2.6:
            candidate_name = "feature-sequence"
            candidate_events = sequence_events
            candidate_debug = {"score": round(sequence_score, 3), **sequence_debug}

    if candidate_events is None:
        clusters = vote.get("clusters") if isinstance(vote.get("clusters"), list) else []
        fallback: list[tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for cluster in clusters[:10]:
            start_t = _safe_float(cluster.get("t"), 0.0)
            candidate = feature_vote_events_from_clusters(clusters, start_t, use_offset_model=True)
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
        return {"available": False, "sequenceRanked": sequence_ranked, "vote": vote}
    return {
        "available": True,
        "method": candidate_name,
        "events": candidate_events,
        "debug": candidate_debug,
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
