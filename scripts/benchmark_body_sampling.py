#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.services.body_pipeline import analyze_body_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark body pose sampling for one video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--target-fps", type=float, default=0.0)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    os.environ["BODY_POSE_TARGET_FPS"] = str(args.target_fps)
    os.environ["DATA_DIR"] = str(args.data_dir)
    settings = Settings()

    started = time.perf_counter()
    response = analyze_body_video(
        settings=settings,
        job_id=args.job_id,
        filename=args.video.name,
        input_path=str(args.video),
        force=True,
    )
    elapsed = time.perf_counter() - started
    artifact = json.loads(Path(response["bodyPath"]).read_text())
    meta = artifact["meta"]
    metrics = artifact["metrics"]
    report = {
        "jobId": args.job_id,
        "targetFps": args.target_fps,
        "elapsedSeconds": round(elapsed, 3),
        "sampleMode": meta["sampleMode"],
        "sampleStride": meta["sampleStride"],
        "effectiveSampleFps": meta["effectiveSampleFps"],
        "sampledFrames": meta["sampledFrames"],
        "poseAvailable": meta["poseAvailable"],
        "bodyCoverage": metrics["bodyCoverage"]["score"],
        "poseCoverage": metrics["poseCoverage"]["score"],
        "wristCoverage": metrics["wristCoverage"]["score"],
        "poseTrackQuality": {
            "label": metrics["poseTrackQuality"]["label"],
            "coreCoverage": metrics["poseTrackQuality"]["score"],
            "joints": metrics["poseTrackQuality"]["joints"],
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
