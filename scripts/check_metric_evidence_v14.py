#!/usr/bin/env python3
"""Regression checks for v14 metric-level evidence gates."""

import sys
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.coach_pipeline import _finalize_metric_evidence, _metric_quality_item  # noqa: E402
from app.services import coach_pipeline  # noqa: E402


def usable_validation() -> dict:
    return {
        "status": "usable",
        "metricAvailability": {
            "tempo": "confirmed",
            "impact": "confirmed",
            "path": "confirmed",
            "shaft": "confirmed",
        },
    }


def check_no_unvalidated_shaft_or_backswing_coaching() -> None:
    frames, wrists = [], []
    for index in range(12):
        x, y = .4 + index * .01, .5 - index * .005
        frames.append({"frame": index, "sourceTimeMs": index * 100, "detections": [
            {"label": "club_head", "conf": .9, "bbox": {"x": x + .04, "y": y - .04, "w": .02, "h": .02}},
            {"label": "club_handle", "conf": .9, "bbox": {"x": x, "y": y, "w": .02, "h": .02}},
        ]})
        wrists.append({"frame": index, "t": index * 100, "x": x + .01, "y": y + .01, "conf": .9})
    meta = {"fps": 10, "width": 500, "height": 500, "durationMs": 1200,
            "viewpoint": "down_the_line", "handedness": "right", "frames": frames}
    validation = {"status": "usable", "message": "fixture", "warnings": [], "codes": [], "metricAvailability": {}}
    shaft = {"label": "neutral", "confidence": .9, "sampleCount": 12, "source": "head_handle", "angleDeg": 49}
    backswing = {"label": "adequate", "score": .8, "source": "pose_wrist"}
    with patch.dict(os.environ, {"REFERENCE_COACHING_ENABLED": "0", "VISIBLE_GRIP_TRACK_ENABLED": "0"}), \
         patch.object(coach_pipeline, "_wrist_track_from_body", return_value=wrists), \
         patch.object(coach_pipeline, "_segment_events", return_value=(0, 4, 8, 11, "fixture", None)), \
         patch.object(coach_pipeline, "_validate_event_evidence", return_value=validation), \
         patch.object(coach_pipeline, "_shaft_plane", return_value=shaft), \
         patch.object(coach_pipeline, "_backswing_metric", return_value=backswing):
        result = coach_pipeline.analyze_meta(meta, "fixture", True)
    assert result["metrics"]["shaftPlane"]["angleDeg"] == 49
    assert result["eventValidation"]["metricEvidence"]["shaft"]["status"] == "reference"
    assert not any(item.get("category") in {"shaft_plane", "backswing"} for item in result["coachFindings"])


def main() -> None:
    validation = usable_validation()
    evidence = _finalize_metric_evidence(
        validation,
        viewpoint="unknown",
        shaft={
            "label": "neutral",
            "confidence": 0.11,
            "sampleCount": 21,
            "source": "head_handle",
            "angleDeg": 51.6,
        },
        backswing={"label": "adequate", "source": "pose_wrist"},
        body_metrics={"poseCoverage": {"score": 1.0}},
    )

    assert evidence["tempo"]["status"] == "reference"
    assert evidence["impactStability"]["status"] == "withheld"
    assert evidence["path"]["status"] == "withheld"
    assert "VIEWPOINT_UNKNOWN" in evidence["path"]["reasons"]
    assert evidence["shaft"]["status"] == "reference"
    assert "SHAFT_CONFIDENCE_LOW" in evidence["shaft"]["reasons"]
    assert evidence["ball"]["status"] == "withheld"
    assert validation["metricAvailability"]["impactStability"] == "withheld"

    hidden = _metric_quality_item(
        {"status": "withheld", "source": "test", "reasons": ["UNSAFE"]},
        123.4,
        0.99,
    )
    assert hidden["value"] is None
    assert hidden["status"] == "withheld"
    reference = _metric_quality_item(
        {"status": "reference", "source": "test", "reasons": ["REFERENCE_ONLY"]},
        2.8,
        0.72,
    )
    assert reference["value"] == 2.8
    assert reference["confidence"] == 0.72

    strong_shaft_validation = usable_validation()
    strong_shaft = _finalize_metric_evidence(
        strong_shaft_validation,
        viewpoint="down_the_line",
        shaft={
            "label": "neutral",
            "confidence": 0.52,
            "sampleCount": 10,
            "source": "head_handle",
            "angleDeg": 49.0,
        },
        backswing={"label": "adequate", "source": "pose_wrist"},
        body_metrics={"poseCoverage": {"score": 0.9}},
    )
    assert strong_shaft["shaft"]["status"] == "reference"
    assert "SHAFT_2D_UNVALIDATED" in strong_shaft["shaft"]["reasons"]
    assert strong_shaft["backswing"]["status"] == "reference"
    assert strong_shaft["path"]["status"] == "withheld"
    assert "TARGET_LINE_NOT_CALIBRATED" in strong_shaft["path"]["reasons"]
    check_no_unvalidated_shaft_or_backswing_coaching()

    print("v14 metric evidence checks passed")


if __name__ == "__main__":
    main()
