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
    _find_finish_from_wrist_track,
    _find_impact_from_wrist_track,
    _find_top_from_wrist_track,
    _safe_float,
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


def replay_fixture(fixture_path: Path, force: bool, allow_missing: bool) -> int:
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
    print(f"  summary: {result.get('summary')}")
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
    args = parser.parse_args()

    if not args.fixtures:
        print("No fixtures found. Add JSON files under fixtures/event_labels/ or pass paths explicitly.")
        return 2

    exit_code = 0
    for idx, fixture in enumerate(args.fixtures):
        if idx:
            print("\n" + "=" * 72)
        try:
            exit_code = max(exit_code, replay_fixture(fixture, force=not args.no_force, allow_missing=args.allow_missing))
        except Exception as exc:  # Keep batch replay useful while algorithms are experimental.
            exit_code = max(exit_code, 1)
            print(f"fixture: {fixture}")
            print(f"error: {type(exc).__name__}: {exc}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
