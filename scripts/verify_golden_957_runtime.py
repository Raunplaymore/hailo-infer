#!/usr/bin/env python3
"""Verify the real Pi artifacts used by golden swing job 957e5457.

The artifacts remain device-side because they contain pose data.  This script
lets a Pi or a secure local checkout prove that it is replaying the exact
golden input before checking the shared video clock and the club-quality gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.coach_pipeline import _normalize_times, analyze_meta  # noqa: E402


JOB_ID = "957e5457-4d13-46bf-88c6-65c467af8487"
FIXTURE_PATH = ROOT / "fixtures" / "event_labels" / f"{JOB_ID}.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    fixture = load_json(FIXTURE_PATH)
    artifacts = fixture["artifacts"]
    parser = argparse.ArgumentParser(description="Verify golden 957 runtime artifacts.")
    parser.add_argument("--meta", type=Path, default=Path(artifacts["metaPath"]))
    parser.add_argument("--body", type=Path, default=Path(artifacts["bodyPath"]))
    args = parser.parse_args()

    expected_hashes = artifacts["sha256"]
    for key, path in (("meta", args.meta), ("body", args.body)):
        require(path.is_file(), f"{key} artifact missing: {path}")
        require(sha256(path) == expected_hashes[key], f"{key} artifact does not match golden checksum: {path}")

    meta = load_json(args.meta)
    body = load_json(args.body)
    timeline = fixture["timeline"]
    labels = fixture["labels"]
    tolerance_ms = float(fixture["toleranceMs"])

    frames = meta["frames"]
    require(len(frames) == int(timeline["lastFrame"]) + 1, "golden meta frame count changed")
    require(all(frame.get("frame") is None for frame in frames), "golden meta must reproduce missing frame fields")
    normalized = _normalize_times(frames, float(timeline["fps"]), float(timeline["durationMs"]))
    require(abs(float(normalized[-1]) - float(timeline["expectedVideoLastTimeMs"])) <= 2.0, "video frame clock was not restored")
    for event_key, frame_index in timeline["poseEventFrames"].items():
        require(
            abs(float(normalized[int(frame_index)]) - float(labels[event_key])) <= tolerance_ms,
            f"{event_key} frame no longer aligns to the video clock",
        )

    result = analyze_meta(copy.deepcopy(meta), JOB_ID, True, str(args.body))
    tracking = result["metrics"]["trackingQuality"]
    require(tracking["clubHeadRawFrames"] > tracking["clubHeadFrames"], "false club-head candidates were not filtered")
    require(tracking["clubHandleRawFrames"] > tracking["clubHandleFrames"], "shaft-sized handles were not filtered")
    require(tracking["ballRawFrames"] > tracking["ballFrames"], "person-sized ball candidates were not filtered")
    validation = result["eventValidation"]
    require(validation["status"] == "withheld", "sparse post-impact club evidence must withhold event coaching")
    require("CLUB_TRACK_INSUFFICIENT" in validation["codes"], "missing head/handle tracking was not reported")

    print(f"golden runtime verification passed ({JOB_ID})")


if __name__ == "__main__":
    main()
