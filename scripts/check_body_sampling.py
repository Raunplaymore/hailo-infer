#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.body_pipeline import _effective_sample_fps, _pose_track_quality, _sample_stride


def main() -> None:
    cases = [
        ((197, 60.0, 0.0), 8, "legacy long video"),
        ((100, 60.0, 0.0), 1, "legacy short video"),
        ((197, 60.0, 30.0), 2, "60fps to 30fps"),
        ((197, 60.0, 60.0), 1, "native 60fps"),
        ((197, 30.0, 60.0), 1, "target capped at source fps"),
        ((0, 0.0, 30.0), 1, "unknown source metadata"),
    ]
    for args, expected, label in cases:
        actual = _sample_stride(*args)
        assert actual == expected, f"{label}: expected {expected}, got {actual}"

    assert _effective_sample_fps(60.0, 2) == 30.0
    assert _effective_sample_fps(0.0, 2) == 0.0

    frames = [
        {"keypoints": {"left_wrist": [0.1, 0.2, 0.9], "right_wrist": [0.2, 0.2, 0.9]}},
        {"keypoints": {"left_wrist": [0.1, 0.2, 0.1], "right_wrist": [0.2, 0.2, 0.9]}},
        {"keypoints": {"left_wrist": [0.1, 0.2, 0.1], "right_wrist": [0.2, 0.2, 0.9]}},
        {"keypoints": {"left_wrist": [0.1, 0.2, 0.9], "right_wrist": [0.2, 0.2, 0.9]}},
    ]
    quality = _pose_track_quality(frames, 20.0)
    assert quality["joints"]["left_wrist"]["usableFrames"] == 2
    assert quality["joints"]["left_wrist"]["maxGapFrames"] == 2
    assert quality["joints"]["left_wrist"]["maxGapMs"] == 100
    print("PASS: body sampling policy")


if __name__ == "__main__":
    main()
