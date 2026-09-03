#!/usr/bin/env python3
"""Compare repeated analysis JSON payloads for deterministic pose-derived output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional


EVENT_KEYS = ("address", "takeaway", "top", "impact", "finish")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    analysis = value.get("analysis")
    return analysis if isinstance(analysis, dict) else value


def number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def event_ms(payload: dict[str, Any], key: str) -> Optional[float]:
    events = payload.get("events") if isinstance(payload.get("events"), dict) else {}
    return number(events.get(f"{key}Ms", events.get(key)))


def finding_keys(payload: dict[str, Any]) -> list[str]:
    findings = payload.get("coachFindings")
    if not isinstance(findings, list):
        return []
    return [str(item.get("key")) for item in findings if isinstance(item, dict) and item.get("key")]


def metric_number(payload: dict[str, Any], metric: str, field: str) -> Optional[float]:
    return number(nested(payload, "metrics", metric, *field.split(".")))


def absolute_check(name: str, baseline: Optional[float], candidate: Optional[float], tolerance: float) -> dict[str, Any]:
    if baseline is None and candidate is None:
        return {"name": name, "passed": True, "baseline": None, "candidate": None, "difference": None, "tolerance": tolerance}
    if baseline is None or candidate is None:
        return {"name": name, "passed": False, "baseline": baseline, "candidate": candidate, "difference": None, "tolerance": tolerance, "reason": "availability_mismatch"}
    difference = abs(candidate - baseline)
    return {"name": name, "passed": difference <= tolerance, "baseline": baseline, "candidate": candidate, "difference": round(difference, 6), "tolerance": tolerance}


def relative_check(name: str, baseline: Optional[float], candidate: Optional[float], tolerance: float) -> dict[str, Any]:
    if baseline is None and candidate is None:
        return {"name": name, "passed": True, "baseline": None, "candidate": None, "relativeDifference": None, "tolerance": tolerance}
    if baseline is None or candidate is None:
        return {"name": name, "passed": False, "baseline": baseline, "candidate": candidate, "relativeDifference": None, "tolerance": tolerance, "reason": "availability_mismatch"}
    denominator = max(abs(baseline), 1e-9)
    difference = abs(candidate - baseline) / denominator
    return {"name": name, "passed": difference <= tolerance, "baseline": baseline, "candidate": candidate, "relativeDifference": round(difference, 6), "tolerance": tolerance}


def compare(baseline: dict[str, Any], candidate: dict[str, Any], fps: float) -> dict[str, Any]:
    event_tolerance_ms = (2.0 * 1000.0) / max(fps, 1.0)
    checks: list[dict[str, Any]] = []
    for key in EVENT_KEYS:
        checks.append(absolute_check(f"event.{key}Ms", event_ms(baseline, key), event_ms(candidate, key), event_tolerance_ms))

    baseline_findings = finding_keys(baseline)
    candidate_findings = finding_keys(candidate)
    checks.append({
        "name": "coachFindingKeys",
        "passed": baseline_findings == candidate_findings,
        "baseline": baseline_findings,
        "candidate": candidate_findings,
    })
    checks.extend([
        absolute_check("body.shoulderTurnProxy.deltaDeg", metric_number(baseline, "body", "shoulderTurnProxy.deltaDeg"), metric_number(candidate, "body", "shoulderTurnProxy.deltaDeg"), 3.0),
        absolute_check("body.hipTurnProxy.deltaDeg", metric_number(baseline, "body", "hipTurnProxy.deltaDeg"), metric_number(candidate, "body", "hipTurnProxy.deltaDeg"), 3.0),
        relative_check("body.headStability.movementRatio", metric_number(baseline, "body", "headStability.movementRatio"), metric_number(candidate, "body", "headStability.movementRatio"), 0.05),
        absolute_check("tempo.ratio", metric_number(baseline, "tempo", "ratio"), metric_number(candidate, "tempo", "ratio"), 0.05),
        absolute_check("shaftPlane.angleDeg", metric_number(baseline, "shaftPlane", "angleDeg"), metric_number(candidate, "shaftPlane", "angleDeg"), 3.0),
    ])
    failed = [check["name"] for check in checks if not check["passed"]]
    return {"passed": not failed, "failedChecks": failed, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-fail", action="store_true", help="write a baseline report without returning exit 1")
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("at least two analysis JSON files are required")

    payloads = [read_json(path) for path in args.inputs]
    baseline = payloads[0]
    comparisons = []
    for path, candidate in zip(args.inputs[1:], payloads[1:]):
        comparisons.append({"candidate": str(path), **compare(baseline, candidate, args.fps)})
    result = {
        "schemaVersion": "pose-determinism-report-v1",
        "baseline": str(args.inputs[0]),
        "fps": args.fps,
        "passed": all(item["passed"] for item in comparisons),
        "comparisons": comparisons,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    print(rendered)
    if not result["passed"] and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
