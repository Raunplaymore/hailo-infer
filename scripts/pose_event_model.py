"""Small offline pose-event candidate. NumPy inference needs no training runtime.

Inputs contain observations and visibility masks only; no event labels or club
predictions. Outputs are research candidates, not calibrated coaching claims.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

JOINTS = ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip",
          "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_knee", "right_knee", "left_ankle", "right_ankle")
EVENTS = ("takeaway", "top", "impact", "finish")
DILATIONS = (1, 2, 4, 8)
GRID_MS = 1000 / 30
CHANNELS = len(JOINTS) * 6


def pose_features(body):
    frames = sorted(body.get("frames", []), key=lambda f: f.get("timeMs", 0))
    if len(frames) < 4:
        raise ValueError("Insufficient observed pose frames")
    observed_times = np.array([float(f["timeMs"]) for f in frames])
    if not np.all(np.isfinite(observed_times)) or np.any(np.diff(observed_times) <= 0):
        raise ValueError("Pose timestamps must be finite and strictly increasing")
    times = np.arange(0, observed_times[-1] + .01, GRID_MS)
    points = np.zeros((len(JOINTS), 4, len(times)), dtype=np.float32)
    width, height = float(body["meta"]["width"]), float(body["meta"]["height"])
    if min(width, height) <= 0:
        raise ValueError("Source image dimensions required")
    axes = np.array([width, height]) / max(width, height)
    roots, scales = [], []
    for frame in frames:
        kp = frame.get("keypoints") or {}
        names = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
        if all(isinstance(kp.get(k), list) and len(kp[k]) >= 3 and kp[k][2] >= .25 for k in names):
            hip = (np.array(kp[names[0]][:2]) + kp[names[1]][:2]) * axes / 2
            shoulder = (np.array(kp[names[2]][:2]) + kp[names[3]][:2]) * axes / 2
            roots.append(hip); scales.append(np.linalg.norm(shoulder - hip))
    if not roots or np.median(scales) < .02:
        raise ValueError("No usable observed torso for coordinate normalization")
    root, scale = np.median(roots, axis=0), float(np.median(scales))
    for i, now in enumerate(times):
        source = int(np.argmin(abs(observed_times - now)))
        if abs(observed_times[source] - now) > GRID_MS / 2 + 1:
            continue  # Keep gaps as missing; do not reconstruct hidden joints.
        kp = frames[source].get("keypoints") or {}
        for j, name in enumerate(JOINTS):
            point = kp.get(name)
            if not isinstance(point, list) or len(point) < 3 or not np.all(np.isfinite(point[:3])):
                continue
            confidence = np.clip(point[2], 0, 1)
            points[j, 2, i] = confidence
            if confidence >= .25:
                points[j, :2, i] = np.clip((np.array(point[:2]) * axes - root) / scale, -4, 4)
                points[j, 3, i] = 1
    velocity = np.zeros((len(JOINTS), 2, len(times)), dtype=np.float32)
    valid_pairs = points[:, 3, 1:] * points[:, 3, :-1]
    velocity[:, :, 1:] = np.diff(points[:, :2, :], axis=2) * valid_pairs[:, None, :]
    features = np.concatenate([points.reshape(-1, len(times)), velocity.reshape(-1, len(times))])
    return features.astype(np.float32), times


def ordered_decode(logits):
    if logits.ndim != 2 or logits.shape[0] != len(EVENTS) or logits.shape[1] < len(EVENTS):
        raise ValueError("Expected four event score sequences")
    if not np.all(np.isfinite(logits)):
        raise ValueError("Non-finite event scores")
    length = logits.shape[1]
    score = logits[0].copy()
    pointers = []
    for event in range(1, len(EVENTS)):
        next_score = np.full(length, -np.inf)
        back = np.full(length, -1, dtype=int)
        best = -1
        for t in range(1, length):
            if best < 0 or score[t - 1] > score[best]:
                best = t - 1
            if np.isfinite(score[best]):
                next_score[t] = score[best] + logits[event, t]
                back[t] = best
        pointers.append(back)
        score = next_score
    current = int(np.argmax(score))
    if not np.isfinite(score[current]):
        raise ValueError("No ordered event sequence")
    indices = [current]
    for back in reversed(pointers):
        current = int(back[current]); indices.append(current)
    return list(reversed(indices))


def conv1d(values, weight, bias, dilation=1):
    kernel = weight.shape[2]
    padding = dilation * (kernel // 2)
    padded = np.pad(values, ((0, 0), (padding, padding)))
    windows = np.stack([padded[:, k * dilation:k * dilation + values.shape[1]] for k in range(kernel)])
    return np.einsum("ock,kct->ot", weight, windows, optimize=True) + bias[:, None]


def numpy_logits(features, weights):
    hidden = np.maximum(0, conv1d(features, weights["input.weight"], weights["input.bias"]))
    for i, dilation in enumerate(DILATIONS):
        hidden = np.maximum(0, hidden + conv1d(hidden, weights[f"blocks.{i}.weight"], weights[f"blocks.{i}.bias"], dilation))
    return conv1d(hidden, weights["output.weight"], weights["output.bias"])


def predict(body, model_path):
    features, times = pose_features(body)
    with np.load(model_path, allow_pickle=False) as weights:
        logits = numpy_logits(features, weights)
    indices = ordered_decode(logits)
    return {"status": "research_candidate", "source": "pose-tcn-v1", "gridMs": GRID_MS,
            "events": {key + "Ms": round(float(times[index])) for key, index in zip(EVENTS, indices)},
            "publicQuality": "withheld_until_validation",
            "observedJointGridCoverage": float(features[:len(JOINTS) * 4].reshape(len(JOINTS), 4, -1)[:, 3].mean())}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(predict(json.loads(args.body.read_text()), args.model)))
