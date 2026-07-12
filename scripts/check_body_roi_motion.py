#!/usr/bin/env python3
"""Regression checks for pose-anchored ROI optical-flow summaries."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.body_pipeline import _roi_flow_motion  # noqa: E402


def test_pose_anchored_roi_flow() -> None:
    previous = np.zeros((160, 120), dtype=np.uint8)
    current = np.zeros((160, 120), dtype=np.uint8)
    cv2.rectangle(previous, (38, 28), (82, 138), 220, -1)
    cv2.rectangle(current, (44, 28), (88, 138), 220, -1)
    keypoints = {
        "nose": [0.52, 0.18, 0.9],
        "left_shoulder": [0.42, 0.30, 0.9],
        "right_shoulder": [0.62, 0.30, 0.9],
        "left_elbow": [0.38, 0.46, 0.9],
        "right_elbow": [0.66, 0.46, 0.9],
        "left_wrist": [0.35, 0.60, 0.9],
        "right_wrist": [0.69, 0.60, 0.9],
        "left_hip": [0.44, 0.62, 0.9],
        "right_hip": [0.60, 0.62, 0.9],
        "left_knee": [0.44, 0.78, 0.9],
        "right_knee": [0.60, 0.78, 0.9],
        "left_ankle": [0.44, 0.92, 0.9],
        "right_ankle": [0.60, 0.92, 0.9],
    }
    motion = _roi_flow_motion(previous, current, keypoints)
    if set(motion) != {"upper", "torso", "lower"}:
        raise AssertionError(f"expected all dynamic ROIs, got {motion}")
    if motion["upper"]["magnitude"] <= 0 or motion["upper"]["dx"] <= 0:
        raise AssertionError(f"rightward motion must be visible in upper ROI, got {motion['upper']}")


if __name__ == "__main__":
    test_pose_anchored_roi_flow()
    print("body ROI motion checks passed")
