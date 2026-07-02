#!/usr/bin/env python3
"""Replay swing event segmentation against hand-labeled fixtures.

This is intentionally a local experiment harness. It does not change service
output; it lets us score event logic against saved body/meta artifacts before
deploying another heuristic.
"""

from __future__ import annotations

import argparse
import json
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
        _print_source_candidates(body_payload, labels, tolerance_ms)
        _print_event_block("experiment preserve-wrist-time", wrist_events, labels, tolerance_ms)
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
    args = parser.parse_args()

    if not args.fixtures:
        print("No fixtures found. Add JSON files under fixtures/event_labels/ or pass paths explicitly.")
        return 2

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
