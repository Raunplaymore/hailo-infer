#!/usr/bin/env python3
"""Offline experiment lab for improving club detection inputs.

This tool never changes the production inference path. It creates derived
videos only when invoked explicitly, then scores separately generated Hailo
meta files. The source video remains untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import cv2
except ModuleNotFoundError:  # Allows score-only review environments without OpenCV.
    cv2 = None  # type: ignore[assignment]


LABELS = {4: "club_head", 5: "club", 6: "club_handle"}
CLUB_LABELS = set(LABELS.values())


@dataclass
class VideoProfile:
    width: int
    height: int
    fps: float
    frames: int


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "-" for char in value).strip("-.") or "lab"


def require_cv2() -> Any:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video preparation; install requirements.txt first")
    return cv2


def video_profile(source: Path) -> VideoProfile:
    opencv = require_cv2()
    if not source.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source}")
    capture = opencv.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {source}")
    profile = VideoProfile(
        width=int(capture.get(opencv.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(capture.get(opencv.CAP_PROP_FRAME_HEIGHT) or 0),
        fps=float(capture.get(opencv.CAP_PROP_FPS) or 30.0),
        frames=int(capture.get(opencv.CAP_PROP_FRAME_COUNT) or 0),
    )
    capture.release()
    if profile.width <= 0 or profile.height <= 0:
        raise RuntimeError(f"Invalid source dimensions: {source}")
    return profile


def read_body_roi(body_path: Path, width: int, height: int) -> tuple[int, int, int, int] | None:
    payload = json.loads(body_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", []) if isinstance(payload, dict) else []
    points: list[tuple[float, float]] = []
    for frame in frames:
        keypoints = frame.get("keypoints", {}) if isinstance(frame, dict) else {}
        for name in ("left_wrist", "right_wrist", "left_elbow", "right_elbow"):
            point = keypoints.get(name) if isinstance(keypoints, dict) else None
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            confidence = float(point[2]) if len(point) > 2 and point[2] is not None else 1.0
            if confidence < 0.15:
                continue
            x, y = float(point[0]), float(point[1])
            if -0.1 <= x <= 1.1 and -0.1 <= y <= 1.1:
                points.append((x * width, y * height))
            else:
                points.append((x, y))
    if len(points) < 4:
        return None
    xs, ys = zip(*points)
    # A generous static arm envelope avoids a jittering crop and leaves room
    # for the shaft between handle and head.
    span = max(max(xs) - min(xs), max(ys) - min(ys), min(width, height) * 0.28)
    padding = span * 0.7
    left = max(0, int(math.floor(min(xs) - padding)))
    top = max(0, int(math.floor(min(ys) - padding)))
    right = min(width, int(math.ceil(max(xs) + padding)))
    bottom = min(height, int(math.ceil(max(ys) + padding)))
    if right - left < 32 or bottom - top < 32:
        return None
    return left, top, right, bottom


def contrast_frame(frame: Any) -> Any:
    opencv = require_cv2()
    lab = opencv.cvtColor(frame, opencv.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = opencv.split(lab)
    clahe = opencv.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    return opencv.cvtColor(opencv.merge((clahe.apply(lightness), a_channel, b_channel)), opencv.COLOR_LAB2BGR)


def write_variant(source: Path, destination: Path, profile: VideoProfile, transform) -> None:
    opencv = require_cv2()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.mp4")
    capture = opencv.VideoCapture(str(source))
    writer = opencv.VideoWriter(
        str(temporary),
        opencv.VideoWriter_fourcc(*"mp4v"),
        profile.fps,
        (profile.width, profile.height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Unable to create mp4v experiment video")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(transform(frame))
    finally:
        capture.release()
        writer.release()
    temporary.replace(destination)


def prepare(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    profile = video_profile(source)
    workspace.mkdir(parents=True, exist_ok=True)
    variants: dict[str, dict[str, Any]] = {
        "source": {"path": str(source), "kind": "unaltered_source"},
    }

    contrast_path = workspace / "contrast.mp4"
    write_variant(source, contrast_path, profile, contrast_frame)
    variants["contrast"] = {"path": str(contrast_path), "kind": "clahe_lightness", "clipLimit": 1.8}

    if args.body:
        roi = read_body_roi(Path(args.body).expanduser().resolve(), profile.width, profile.height)
        if roi:
            left, top, right, bottom = roi

            def wrist_roi(frame: Any) -> Any:
                crop = frame[top:bottom, left:right]
                return require_cv2().resize(crop, (profile.width, profile.height), interpolation=require_cv2().INTER_CUBIC)

            roi_path = workspace / "wrist-roi.mp4"
            write_variant(source, roi_path, profile, wrist_roi)
            variants["wrist-roi"] = {
                "path": str(roi_path),
                "kind": "static_wrist_arm_roi",
                "sourceCrop": {"left": left, "top": top, "right": right, "bottom": bottom},
                "note": "ROI coordinates must be projected back before production fusion; this experiment scores detection availability only.",
            }
        else:
            variants["wrist-roi"] = {"skipped": True, "reason": "insufficient_pose_wrist_elbow_points"}

    manifest = {
        "lab": "club-preprocess-v1",
        "source": str(source),
        "profile": asdict(profile),
        "variants": variants,
        "productionImpact": "none",
    }
    output = workspace / "variants.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def detection_label(detection: dict[str, Any]) -> str:
    raw = detection.get("label") or detection.get("name") or detection.get("className")
    if raw:
        return str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    for key in ("classId", "class_id", "id"):
        if key in detection:
            try:
                return LABELS.get(int(detection[key]), "")
            except (TypeError, ValueError):
                return ""
    return ""


def detection_confidence(detection: dict[str, Any]) -> float:
    for key in ("confidence", "conf", "score", "prob"):
        try:
            return max(0.0, min(1.0, float(detection.get(key))))
        except (TypeError, ValueError):
            continue
    return 0.0


def score_meta(meta_path: Path) -> dict[str, Any]:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", []) if isinstance(payload, dict) else []
    seen: dict[str, int] = {label: 0 for label in CLUB_LABELS}
    confidences: dict[str, list[float]] = {label: [] for label in CLUB_LABELS}
    paired = 0
    for frame in frames:
        detections = frame.get("detections", []) if isinstance(frame, dict) else []
        labels = set()
        for detection in detections if isinstance(detections, list) else []:
            if not isinstance(detection, dict):
                continue
            label = detection_label(detection)
            if label in CLUB_LABELS:
                labels.add(label)
                confidences[label].append(detection_confidence(detection))
        for label in labels:
            seen[label] += 1
        if "club_head" in labels and "club_handle" in labels:
            paired += 1
    total = max(1, len(frames))
    coverage = {label: round(seen[label] / total, 4) for label in sorted(CLUB_LABELS)}
    average_confidence = {
        label: round(sum(confidences[label]) / len(confidences[label]), 4) if confidences[label] else 0.0
        for label in sorted(CLUB_LABELS)
    }
    # A candidate is useful only if it materially increases the two points that
    # define shaft direction. club bbox is deliberately a small supporting term.
    quality = (
        coverage["club_head"] * 0.45
        + coverage["club_handle"] * 0.25
        + (paired / total) * 0.25
        + coverage["club"] * 0.05
    )
    return {
        "frames": len(frames),
        "detectedFrames": seen,
        "pairedHeadHandleFrames": paired,
        "coverage": coverage,
        "averageConfidence": average_confidence,
        "shaftEvidenceScore": round(quality, 4),
    }


def parse_named_paths(values: Iterable[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value}")
        name, raw_path = value.split("=", 1)
        parsed[safe_name(name)] = Path(raw_path).expanduser().resolve()
    return parsed


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def camera_meta(camera_url: str, job_id: str, input_path: Path, profile: VideoProfile, timeout: int, model: str | None) -> Path:
    duration_sec = profile.frames / profile.fps if profile.fps > 0 else None
    payload = {
        "jobId": job_id,
        "inputPath": str(input_path),
        "filename": input_path.name,
        "force": True,
        "durationSec": duration_sec,
        "width": profile.width,
        "height": profile.height,
        "fps": profile.fps,
        "durationMs": round(duration_sec * 1000) if duration_sec else None,
        "debugDetections": False,
    }
    # Omit the field for the lab's default mode: the camera selects its active
    # production detector itself. This avoids coupling an experiment to a
    # deployment-specific public alias.
    if model:
        payload["model"] = model
    request = urllib.request.Request(
        f"{camera_url.rstrip('/')}/api/meta/from-file",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Camera meta request failed ({error.code}): {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Camera meta request failed: {error.reason}") from error
    meta_path = result.get("metaPath") if isinstance(result, dict) else None
    if not result.get("ok") or not isinstance(meta_path, str):
        raise RuntimeError(f"Camera did not return a meta path: {result}")
    return Path(meta_path)


def run(args: argparse.Namespace) -> int:
    """Prepare lab variants, ask only pi_camera for metadata, then score them.

    The analysis API is intentionally never called, so no job, coach output, or
    NAS archive is created by this command.
    """
    prepare(args)
    source = Path(args.source).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    upload_dir = Path(args.upload_dir).expanduser().resolve()
    if not is_within(source, upload_dir):
        raise ValueError(f"Source must be inside upload directory: {upload_dir}")
    manifest = json.loads((workspace / "variants.json").read_text(encoding="utf-8"))
    profile = VideoProfile(**manifest["profile"])
    copied_dir = upload_dir / "lab" / safe_name(args.job_prefix)
    copied_dir.mkdir(parents=True, exist_ok=True)
    meta_paths: dict[str, Path] = {}
    for name, variant in manifest["variants"].items():
        if variant.get("skipped"):
            continue
        input_path = Path(variant["path"])
        if name != "source":
            destination = copied_dir / f"{safe_name(name)}.mp4"
            shutil.copy2(input_path, destination)
            input_path = destination
        meta_paths[name] = camera_meta(
            args.camera_url,
            f"{safe_name(args.job_prefix)}-{safe_name(name)}",
            input_path,
            profile,
            args.timeout,
            args.model,
        )

    score_args = argparse.Namespace(meta=[f"{name}={path}" for name, path in meta_paths.items()], output=str(workspace / "score.json"))
    score(score_args)
    run_report = {
        "lab": "club-preprocess-v1",
        "productionImpact": "none",
        "cameraUrl": args.camera_url,
        "metaPaths": {name: str(path) for name, path in meta_paths.items()},
        "derivedUploadDir": str(copied_dir),
        "scorePath": str(workspace / "score.json"),
    }
    (workspace / "run.json").write_text(json.dumps(run_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def score(args: argparse.Namespace) -> int:
    metas = parse_named_paths(args.meta)
    if "source" not in metas:
        raise ValueError("A source=PATH meta file is required as the baseline")
    results = {name: score_meta(path) for name, path in metas.items()}
    baseline = results["source"]
    candidates = []
    for name, result in results.items():
        if name == "source":
            continue
        gain = result["shaftEvidenceScore"] - baseline["shaftEvidenceScore"]
        head_gain = result["detectedFrames"]["club_head"] - baseline["detectedFrames"]["club_head"]
        paired_gain = result["pairedHeadHandleFrames"] - baseline["pairedHeadHandleFrames"]
        qualifies = gain >= 0.08 and head_gain >= 2 and paired_gain >= 0
        candidates.append({"variant": name, "scoreGain": round(gain, 4), "headFrameGain": head_gain, "pairedFrameGain": paired_gain, "qualifies": qualifies})
    report = {
        "lab": "club-preprocess-v1",
        "results": results,
        "candidates": candidates,
        "decision": "candidate_for_visual_review" if any(row["qualifies"] for row in candidates) else "no_candidate",
        "guardrail": "A numeric candidate is not production-ready until false positives and coordinate projection are visually reviewed.",
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).expanduser().resolve().write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline club preprocessing experiment lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="create derived videos without running inference")
    prepare_parser.add_argument("--source", required=True, help="source video path; never modified")
    prepare_parser.add_argument("--workspace", required=True, help="experiment output directory")
    prepare_parser.add_argument("--body", help="optional pose body JSON for static wrist/arm ROI")
    prepare_parser.set_defaults(handler=prepare)
    score_parser = subparsers.add_parser("score", help="compare separately generated Hailo meta files")
    score_parser.add_argument("--meta", required=True, action="append", help="NAME=meta.json; source=PATH is required")
    score_parser.add_argument("--output", help="optional report JSON path")
    score_parser.set_defaults(handler=score)
    run_parser = subparsers.add_parser("run", help="prepare, generate camera-only meta, and score a complete lab run")
    run_parser.add_argument("--source", required=True, help="source video below --upload-dir; never modified")
    run_parser.add_argument("--workspace", required=True, help="experiment output directory")
    run_parser.add_argument("--body", help="optional pose body JSON for static wrist/arm ROI")
    run_parser.add_argument("--job-prefix", required=True, help="unique lab-only ID prefix")
    run_parser.add_argument("--camera-url", default="http://127.0.0.1:3001", help="local pi_camera URL")
    run_parser.add_argument("--model", help="optional pi_camera model override")
    run_parser.add_argument("--upload-dir", default="/home/ray/uploads", help="pi_camera upload directory")
    run_parser.add_argument("--timeout", type=int, default=180, help="per-variant camera request timeout in seconds")
    run_parser.set_defaults(handler=run)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
