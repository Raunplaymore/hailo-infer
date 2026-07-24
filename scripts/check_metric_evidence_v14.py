#!/usr/bin/env python3
"""Regression checks for v14 metric-level evidence gates."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.coach_pipeline import _finalize_metric_evidence  # noqa: E402


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
    assert strong_shaft["shaft"]["status"] == "confirmed"
    assert strong_shaft["path"]["status"] == "withheld"
    assert "TARGET_LINE_NOT_CALIBRATED" in strong_shaft["path"]["reasons"]

    print("v14 metric evidence checks passed")


if __name__ == "__main__":
    main()
