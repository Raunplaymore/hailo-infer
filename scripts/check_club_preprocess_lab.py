#!/usr/bin/env python3
"""Small regression check for the offline club preprocessing lab."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.club_preprocess_lab import cv2, main, score_meta  # noqa: E402


def write_video(path: Path) -> None:
    assert cv2 is not None, "OpenCV is unavailable"
    import numpy as np
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (96, 64))
    assert writer.isOpened(), "test video writer unavailable"
    for index in range(6):
        frame = np.full((64, 96, 3), 25 + index * 15, dtype=np.uint8)
        cv2.line(frame, (10, 50), (80, 8), (170, 170, 170), 1)
        writer.write(frame)
    writer.release()


def test_prepare_and_score() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.mp4"
        body = root / "body.json"
        body.write_text(json.dumps({"frames": [{"keypoints": {"left_wrist": [0.35, 0.5, 0.9], "right_wrist": [0.5, 0.48, 0.9], "left_elbow": [0.3, 0.55, 0.9], "right_elbow": [0.55, 0.54, 0.9]}}]}))
        if cv2 is not None:
            write_video(source)
            workspace = root / "lab"
            original_argv = sys.argv
            try:
                sys.argv = ["club_preprocess_lab.py", "prepare", "--source", str(source), "--body", str(body), "--workspace", str(workspace)]
                assert main() == 0
            finally:
                sys.argv = original_argv
            manifest = json.loads((workspace / "variants.json").read_text())
            assert Path(manifest["variants"]["contrast"]["path"]).exists()
            assert Path(manifest["variants"]["wrist-roi"]["path"]).exists()
        meta = root / "source.meta.json"
        meta.write_text(json.dumps({"frames": [
            {"detections": [{"label": "club_head", "confidence": 0.8}, {"label": "club_handle", "confidence": 0.7}]},
            {"detections": [{"classId": 5, "confidence": 0.6}]},
        ]}))
        result = score_meta(meta)
        assert result["detectedFrames"]["club_head"] == 1
        assert result["pairedHeadHandleFrames"] == 1
        assert result["shaftEvidenceScore"] > 0


if __name__ == "__main__":
    test_prepare_and_score()
    print("club preprocess lab checks passed")
