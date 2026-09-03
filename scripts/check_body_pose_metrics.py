#!/usr/bin/env python3
"""Regression checks for 180-degree-periodic body pose geometry."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.coach_pipeline import (
    _axial_angle_delta,
    _body_pose_metrics,
    _torso_normalized_head_point,
    _unwrap_axial_samples,
)


def close(actual: float, expected: float, tolerance: float = 1e-6) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"expected {expected}, got {actual}")


def main() -> None:
    # Reversing the endpoints of the same shoulder/hip line changes the
    # directed angle by 180 degrees but must not change the axial result.
    close(_axial_angle_delta(10.0, 190.0), 0.0)
    close(_axial_angle_delta(0.0, 173.6), 6.4)

    # Crossing the atan2 boundary is a two-degree movement, not 358 degrees.
    close(_axial_angle_delta(179.0, -179.0), 2.0)

    unwrapped = _unwrap_axial_samples([(0.0, 179.0), (16.0, -179.0), (32.0, 2.0)])
    values = [value for _, value in unwrapped]
    if values != [179.0, 181.0, 182.0]:
        raise AssertionError(f"unexpected axial unwrap: {values}")

    base = {
        "nose": [0.50, 0.20, 0.9],
        "left_shoulder": [0.40, 0.40, 0.9],
        "right_shoulder": [0.60, 0.40, 0.9],
        "left_hip": [0.43, 0.65, 0.9],
        "right_hip": [0.57, 0.65, 0.9],
    }

    def transformed(scale: float, angle_deg: float, tx: float, ty: float) -> dict:
        import math

        angle = math.radians(angle_deg)
        cosine, sine = math.cos(angle), math.sin(angle)
        keypoints = {}
        for name, (x, y, confidence) in base.items():
            keypoints[name] = [
                scale * (x * cosine - y * sine) + tx,
                scale * (x * sine + y * cosine) + ty,
                confidence,
            ]
        return {"keypoints": keypoints}

    normalized = _torso_normalized_head_point({"keypoints": base})
    moved = _torso_normalized_head_point(transformed(1.7, 31.0, 0.2, -0.1))
    if normalized is None or moved is None:
        raise AssertionError("synthetic torso-normalized head points must be available")
    close(normalized[0], moved[0])
    close(normalized[1], moved[1])

    body = {
        "frames": [
            {"timeMs": index * 33, "keypoints": base}
            for index in range(8)
        ]
    }
    head = _body_pose_metrics(body, 0, 132, 231)["headStability"]
    if head["status"] != "withheld" or head["label"] != "unknown":
        raise AssertionError(f"uncalibrated head metric must remain withheld: {head}")
    if "HEAD_THRESHOLDS_NOT_CALIBRATED" not in head["reasons"]:
        raise AssertionError(f"head threshold reason must remain traceable: {head}")

    print("body pose metric checks passed")


if __name__ == "__main__":
    main()
