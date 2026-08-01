#!/usr/bin/env python3
"""Safely migrate legacy four-event swing labels to Address/Takeaway labels.

Legacy datasets used the visual start of the club motion as ``address``.  This
script preserves that human label as ``takeaway`` and seeds a new Address from
the stable pose observation immediately before it.  Seeded Address labels are
explicitly marked as analysis-originated and every migrated annotation is put
back into draft review.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def wrist_point(frame: dict[str, Any]) -> tuple[float, float] | None:
    keypoints = frame.get("keypoints")
    if not isinstance(keypoints, dict):
        return None
    points: list[tuple[float, float]] = []
    for key in ("left_wrist", "right_wrist"):
        point = keypoints.get(key)
        if not isinstance(point, list) or len(point) < 2:
            continue
        confidence = number(point[2]) if len(point) > 2 else 1.0
        x, y = number(point[0]), number(point[1])
        if confidence is not None and confidence >= 0.25 and x is not None and y is not None:
            points.append((x, y))
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def address_seed(body: dict[str, Any] | None, takeaway: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    takeaway_ms = number(takeaway.get("timeMs"))
    if takeaway_ms is None:
        return None, "missing_takeaway_time"
    frames = body.get("frames") if isinstance(body, dict) else None
    if not isinstance(frames, list):
        return None, "missing_body"

    # A 0.4 s lead provides a distinct setup frame while remaining close enough
    # to the human-labelled first club movement.  We accept 0.2–0.65 s before
    # Takeaway to accommodate the sparse (about 6 fps) pose artifact.
    target_ms = max(0.0, takeaway_ms - 400.0)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for frame in frames:
        if not isinstance(frame, dict) or wrist_point(frame) is None:
            continue
        time_ms = number(frame.get("timeMs"))
        if time_ms is None or time_ms > takeaway_ms - 200.0 or time_ms < takeaway_ms - 650.0:
            continue
        candidates.append((time_ms, frame))
    if not candidates:
        return None, "no_stable_pose_before_takeaway"
    time_ms, frame = min(candidates, key=lambda item: abs(item[0] - target_ms))
    frame_index = number(frame.get("frameIndex"))
    if frame_index is None:
        return None, "missing_pose_frame_index"
    return {
        "frame": round(frame_index),
        "timeMs": round(time_ms),
        "source": "analysis",
    }, "pose_pre_takeaway_400ms"


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/ray/data")
    parser.add_argument("--apply", action="store_true", help="write changes after creating a backup")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    annotation_dir = data_dir / "annotations" / "swing-tracking"
    body_dir = data_dir / "body"
    files = sorted(annotation_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no annotations found: {annotation_dir}")

    audit: list[dict[str, Any]] = []
    changes: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        annotation = load_json(path)
        if annotation is None:
            audit.append({"file": path.name, "status": "skipped", "reason": "invalid_json"})
            continue
        events = annotation.get("events") if isinstance(annotation.get("events"), dict) else {}
        legacy_address = events.get("address") if isinstance(events.get("address"), dict) else None
        if not legacy_address:
            audit.append({"jobId": annotation.get("jobId"), "status": "skipped", "reason": "missing_legacy_address"})
            continue
        if events.get("takeaway"):
            audit.append({"jobId": annotation.get("jobId"), "status": "skipped", "reason": "takeaway_already_present"})
            continue
        job_id = str(annotation.get("jobId") or path.stem)
        body = load_json(body_dir / f"{job_id}.json")
        new_address, method = address_seed(body, legacy_address)
        if new_address is None:
            audit.append({
                "jobId": job_id,
                "status": "needs_manual_address",
                "legacyAddress": legacy_address,
                "reason": method,
            })
            continue
        annotation["schemaVersion"] = "swing-tracking-label-v2"
        annotation["events"] = {
            "address": new_address,
            "takeaway": legacy_address,
            "top": events.get("top"),
            "impact": events.get("impact"),
            "finish": events.get("finish"),
        }
        annotation["status"] = "draft"
        changes.append((path, annotation))
        audit.append({
            "jobId": job_id,
            "status": "migrated",
            "method": method,
            "address": new_address,
            "takeaway": legacy_address,
        })

    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "total": len(files),
        "migrated": len(changes),
        "needsManualAddress": sum(item.get("status") == "needs_manual_address" for item in audit),
        "skipped": sum(item.get("status") == "skipped" for item in audit),
        "items": audit,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.apply:
        return 0

    archive_dir = data_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = archive_dir / f"swing-tracking-before-takeaway-migration-{stamp}.tar.gz"
    with tarfile.open(backup_path, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.name)
    for path, annotation in changes:
        temporary = path.with_suffix(".json.migrating")
        temporary.write_text(f"{json.dumps(annotation, ensure_ascii=False, indent=2)}\n")
        temporary.replace(path)
    audit_path = archive_dir / f"swing-tracking-takeaway-migration-{stamp}.json"
    audit_path.write_text(f"{json.dumps(report, ensure_ascii=False, indent=2)}\n")
    print(json.dumps({"backup": str(backup_path), "audit": str(audit_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
