#!/usr/bin/env python3
"""Tune the DTL V2 Takeaway detector without touching production settings.

The evaluator reads the fixed video-disjoint split and manually labelled
Takeaway times. It searches only train samples, selects among the best train
candidates on validation, and prints the untouched test result once.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from app.services import coach_pipeline
from app.services.meta_loader import load_meta


DEFAULT_MANIFEST = "/home/ray/data/datasets/dtl-club-points-v2-73-r1/event_training_manifest.json"
DEFAULT_CACHE_DIR = "/home/ray/data/service-analysis-cache"
SPEED_MULTIPLIERS = (1.5, 2.0, 2.5, 3.0, 3.5)
DISPLACEMENT_RATIOS = (0.015, 0.025, 0.04, 0.055, 0.07)


def candidate_takeaway_index(
    track: List[dict],
    speeds: List[float],
    address_idx: int,
    top_idx: int,
    speed_multiplier: float,
    displacement_ratio: float,
) -> int:
    """Same sustained-motion detector as production, with two tunable gates."""
    if len(track) < 3 or not speeds:
        return address_idx
    start = max(0, min(address_idx, len(track) - 1))
    end = max(start, min(top_idx, len(track) - 1))
    if end - start < 2:
        return start

    address_time = coach_pipeline._safe_float(track[start].get("t"), 0.0)
    baseline_end = min(end, start + 3)
    for idx in range(start + 1, end + 1):
        if coach_pipeline._safe_float(track[idx].get("t"), address_time) - address_time > 120.0:
            baseline_end = max(start + 1, idx - 1)
            break
    baseline = track[start : baseline_end + 1]
    baseline_x = coach_pipeline._median([coach_pipeline._safe_float(point.get("x"), 0.0) for point in baseline])
    baseline_y = coach_pipeline._median([coach_pipeline._safe_float(point.get("y"), 0.0) for point in baseline])
    scale = coach_pipeline._coord_scale(track[start : end + 1])
    baseline_speeds = speeds[start : baseline_end + 1]
    speed_threshold = max(coach_pipeline._median(baseline_speeds) * speed_multiplier, scale * 0.006, 1e-6)
    displacement_threshold = max(scale * displacement_ratio, coach_pipeline._median(baseline_speeds) * 3.0, 1e-6)

    for idx in range(baseline_end + 1, end):
        point = track[idx]
        next_point = track[idx + 1]
        displacement = math.hypot(
            coach_pipeline._safe_float(point.get("x"), 0.0) - baseline_x,
            coach_pipeline._safe_float(point.get("y"), 0.0) - baseline_y,
        )
        next_displacement = math.hypot(
            coach_pipeline._safe_float(next_point.get("x"), 0.0) - baseline_x,
            coach_pipeline._safe_float(next_point.get("y"), 0.0) - baseline_y,
        )
        if max(speeds[idx], speeds[idx + 1]) >= speed_threshold and displacement >= displacement_threshold and next_displacement >= displacement_threshold:
            return idx
    return start


@contextmanager
def patched_takeaway_detector(speed_multiplier: float, displacement_ratio: float) -> Iterator[None]:
    original = coach_pipeline._find_takeaway_index

    def patched(track: List[dict], speeds: List[float], address_idx: int, top_idx: int) -> int:
        return candidate_takeaway_index(track, speeds, address_idx, top_idx, speed_multiplier, displacement_ratio)

    coach_pipeline._find_takeaway_index = patched
    try:
        yield
    finally:
        coach_pipeline._find_takeaway_index = original


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def load_records(manifest_path: Path, cache_dir: Path) -> List[Dict[str, Any]]:
    manifest = read_json(manifest_path)
    records = []
    for sample in manifest.get("samples", []):
        job_id = sample.get("jobId")
        truth = sample.get("events", {}).get("takeaway")
        if not isinstance(job_id, str) or not isinstance(truth, (int, float)):
            continue
        cache_path = cache_dir / f"{job_id}.json"
        if not cache_path.exists():
            continue
        cache = read_json(cache_path)
        sidecar = cache.get("dtlClubPointsV2")
        if not isinstance(sidecar, dict) or sidecar.get("status") != "succeeded" or not isinstance(sidecar.get("metaPath"), str):
            continue
        body_path = (cache.get("progress") or {}).get("bodyPath")
        records.append({
            "jobId": job_id,
            "split": sample.get("split"),
            "truthMs": int(round(truth)),
            "metaPath": sidecar["metaPath"],
            "bodyPath": body_path if isinstance(body_path, str) else None,
        })
    return records


def score(errors: Iterable[float]) -> Dict[str, Any]:
    values = list(errors)
    return {
        "samples": len(values),
        "maeMs": round(sum(values) / len(values)) if values else None,
        "within100MsRate": round(sum(value <= 100 for value in values) / len(values), 3) if values else None,
    }


def evaluate(
    records: List[Dict[str, Any]],
    speed_multiplier: float,
    displacement_ratio: float,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, Dict[str, Any]]:
    errors: Dict[str, List[float]] = {split: [] for split in splits}
    with patched_takeaway_detector(speed_multiplier, displacement_ratio):
        for record in records:
            try:
                result = coach_pipeline.analyze_meta(
                    load_meta(record["metaPath"]),
                    job_id=record["jobId"],
                    force=True,
                    body_path=record["bodyPath"],
                )
                predicted = result.get("events", {}).get("takeawayMs")
                if isinstance(predicted, (int, float)) and record["split"] in errors:
                    errors[record["split"]].append(abs(predicted - record["truthMs"]))
            except Exception as exc:  # A bad sidecar must not silently become a good score.
                print(f"skip {record['jobId']}: {exc}")
    return {split: score(values) for split, values in errors.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    parser.add_argument("--cache-dir", type=Path, default=Path(DEFAULT_CACHE_DIR))
    args = parser.parse_args()
    records = load_records(args.manifest, args.cache_dir)
    if not records:
        raise SystemExit("no completed V2 sidecars with labelled Takeaway events")

    tuning_records = [record for record in records if record["split"] in {"train", "val"}]
    test_records = [record for record in records if record["split"] == "test"]
    candidates = []
    for speed_multiplier in SPEED_MULTIPLIERS:
        for displacement_ratio in DISPLACEMENT_RATIOS:
            result = evaluate(tuning_records, speed_multiplier, displacement_ratio, ("train", "val"))
            candidates.append({
                "speedMultiplier": speed_multiplier,
                "displacementRatio": displacement_ratio,
                "scores": result,
            })

    # Train ranks candidates; validation chooses among the top three so the
    # fixed test split remains untouched until this single report.
    ranked = sorted(candidates, key=lambda item: (
        item["scores"]["train"]["maeMs"] if item["scores"]["train"]["maeMs"] is not None else float("inf"),
        -(item["scores"]["train"]["within100MsRate"] or 0),
    ))
    finalists = ranked[:3]
    selected = min(finalists, key=lambda item: (
        item["scores"]["val"]["maeMs"] if item["scores"]["val"]["maeMs"] is not None else float("inf"),
        -(item["scores"]["val"]["within100MsRate"] or 0),
    ))
    # Test is deliberately evaluated only once, after all tuning choices are
    # frozen from train/validation results.
    selected_test = evaluate(
        test_records,
        selected["speedMultiplier"],
        selected["displacementRatio"],
        ("test",),
    )["test"]
    print(json.dumps({
        "records": {split: sum(record["split"] == split for record in records) for split in ("train", "val", "test")},
        "productionBaseline": {"speedMultiplier": 2.5, "displacementRatio": 0.04},
        "finalists": finalists,
        "selected": {**selected, "test": selected_test},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
