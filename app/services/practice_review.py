"""A user-selected 2D posture review, independent of inferred swing events."""
import json
import math
from pathlib import Path
from statistics import median
from app.services.movement_guide import build_movement_guide

VERSION = "dtl-posture-review-v1"
# Review sensitivity, NOT a validated golf-fault boundary. Calibrate against
# independently labelled DTL clips before enabling automatic diagnosis.
REVIEW_CHANGE_DEG = 12.0
WINDOW_MS = 65.0


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _angle(frame, side, width, height):
    if frame.get("keypointSource") not in (None, "observed"):
        return None
    points = frame.get("keypoints") or {}
    if not isinstance(points, dict):
        return None
    shoulder, hip = points.get(f"{side}_shoulder"), points.get(f"{side}_hip")
    for point in (shoulder, hip):
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            return None
        if not all(number(v) for v in point[:3]) or not (0.02 < point[0] < .98 and .02 < point[1] < .98 and .65 <= point[2] <= 1):
            return None
    dx, dy = (shoulder[0] - hip[0]) * width, (hip[1] - shoulder[1]) * height
    length = math.hypot(dx, dy)
    if dy <= 0 or length < height * .08:
        return None
    return math.degrees(math.atan2(abs(dx), dy)), length, min(shoulder[2], hip[2])


def _window(frames, time_ms, side, width, height):
    samples = []
    for frame in frames:
        if not isinstance(frame, dict) or not number(frame.get("timeMs")):
            continue
        if abs(frame["timeMs"] - time_ms) <= WINDOW_MS:
            value = _angle(frame, side, width, height)
            if value is not None:
                samples.append((frame["timeMs"], *value))
    samples.sort()
    if len(samples) < 3 or len({s[0] for s in samples}) != len(samples):
        return None
    nearest = min(samples, key=lambda s: abs(s[0] - time_ms))
    if abs(nearest[0] - time_ms) > 40 or max(b[0] - a[0] for a, b in zip(samples, samples[1:])) > 70:
        return None
    angles = [s[1] for s in samples]
    spread = max(angles) - min(angles)
    if spread > 8:
        return None
    return {
        "timeMs": nearest[0], "selectedTimeMs": time_ms,
        "angleDeg": round(median(angles), 1), "sampleCount": len(samples),
        "windowSpreadDeg": round(spread, 1),
        "length": median(s[2] for s in samples),
        "quality": min(s[3] for s in samples),
    }


def measure_posture(body, job_id, address_ms, comparison_ms):
    result = {"ok": True, "jobId": job_id, "version": VERSION,
              "status": "withheld", "reasons": [],
              "message": "선택한 구간의 어깨와 골반을 안정적으로 추적하지 못했습니다. 자세 선이 잘 보이는 인접 장면을 선택해 주세요."}
    meta = body.get("meta") or {}
    if not isinstance(meta, dict):
        return {**result, "reasons": ["VIDEO_GEOMETRY_MISSING"]}
    width, height, duration = (meta.get(k) for k in ("width", "height", "durationMs"))
    if not all(number(v) and v > 0 for v in (width, height, duration)):
        return {**result, "reasons": ["VIDEO_GEOMETRY_MISSING"]}
    if not all(number(v) for v in (address_ms, comparison_ms)) or not 0 <= address_ms < comparison_ms <= duration or comparison_ms - address_ms < 150:
        raise ValueError("어드레스 다음의 임팩트 직전 장면을 0.15초 이상 간격으로 선택해 주세요.")
    frames = body.get("frames")
    if not isinstance(frames, list):
        return {**result, "reasons": ["POSE_MISSING"]}
    choices = []
    for side in ("left", "right"):
        a = _window(frames, address_ms, side, width, height)
        b = _window(frames, comparison_ms, side, width, height)
        if a and b and 10 <= a["angleDeg"] <= 65 and .8 <= b["length"] / a["length"] <= 1.2:
            choices.append((side, a, b))
    if not choices:
        return {**result, "reasons": ["POSE_WINDOW_UNRELIABLE"]}
    # Never switch the tracked side between the two selected positions.
    if len(choices) == 2 and abs((choices[0][1]["angleDeg"] - choices[0][2]["angleDeg"]) - (choices[1][1]["angleDeg"] - choices[1][2]["angleDeg"])) > 8:
        return {**result, "reasons": ["TORSO_SIDES_DISAGREE"]}
    side, address, comparison = max(choices, key=lambda c: min(c[1]["quality"], c[2]["quality"]))
    change = round(address["angleDeg"] - comparison["angleDeg"], 1)
    evidence = {"side": side, "changeDeg": change}
    for name, window in (("address", address), ("comparison", comparison)):
        evidence[name] = {k: v for k, v in window.items() if k not in {"length", "quality"}}
    result.update(evidence=evidence, reasons=["SINGLE_CAMERA_2D_OBSERVATION", "USER_SELECTED_PHASES", "REVIEW_THRESHOLD_NOT_VALIDATED"])
    if change < REVIEW_CHANGE_DEG:
        return {**result, "status": "no_clear_change", "message": "선택한 두 구간에서 상체가 크게 세워지는 변화는 잡히지 않았습니다. 스윙 전체가 좋다는 판정은 아닙니다."}
    return {**result, "status": "review_candidate",
            "message": "선택한 임팩트 직전 구간에서 상체 선이 어드레스보다 세워지는 변화가 관측됐습니다. 두 장면과 자세 선이 실제 움직임에 맞는지 먼저 확인하세요.",
            "practice": {
                "title": "작은 스윙으로 자세 변화 확인하기",
                "steps": ["공 없이 편안한 어드레스를 만든 뒤 허리 높이의 작은 백스윙을 합니다.",
                          "절반 속도로 공이 놓일 위치를 지나가 보세요. 몸을 억지로 고정하지 말고 자연스럽게 피니시합니다.",
                          "작은 스윙을 3회 연습한 뒤, 같은 클럽으로 처음과 같은 완전한 스윙을 같은 카메라 위치에서 다시 촬영하세요."],
                "checkpoint": "다음 영상의 같은 두 구간을 선택해 변화를 비교하세요. 타점이나 편안함이 나빠지면 이 과제를 중단하세요. 수치 감소만으로 교정 성공을 판단하지 않습니다.",
            }}


def load_practice_body(settings, payload):
    source = Path(payload.inputPath).resolve(strict=True)
    body_path = Path(payload.bodyPath).resolve(strict=True)
    if not any(source.is_relative_to(root.resolve()) for root in (settings.upload_dir, settings.data_dir)) or not body_path.is_relative_to(settings.data_dir.resolve()):
        raise ValueError("저장된 영상과 자세 입력만 사용할 수 있습니다.")
    if body_path.stat().st_size > 30_000_000:
        raise ValueError("자세 입력이 너무 큽니다.")
    body = json.loads(body_path.read_text())
    if (not isinstance(body, dict) or body.get("jobId") != payload.jobId
            or not isinstance(body.get("source"), dict)
            or Path((body.get("source") or {}).get("inputPath", "")).resolve() != source
            or body_path.stat().st_mtime_ns < source.stat().st_mtime_ns):
        raise ValueError("영상과 자세 분석이 일치하지 않습니다. 이 영상을 다시 분석해 주세요.")
    return body


def review_saved_body(settings, payload):
    return measure_posture(load_practice_body(settings, payload), payload.jobId, payload.addressMs, payload.comparisonMs)


def preview_saved_body(settings, payload):
    body = load_practice_body(settings, payload)
    guide = build_movement_guide(body, payload.addressMs, getattr(payload, "impactMs", None))
    # Only search near the proposed phases, by distance, never by desired angle.
    pairs = [(max(100, payload.addressMs) + a, payload.comparisonMs + b)
             for a in (0, -50, 50, -100, 100, 150) for b in (0, -33, 33, -66, -100)]
    pairs.sort(key=lambda p: abs(p[0] - max(100, payload.addressMs)) + abs(p[1] - payload.comparisonMs))
    for address, comparison in pairs:
        if address < 0 or comparison - address < 150 or comparison > min(30000, body.get("meta", {}).get("durationMs", 0)):
            continue
        result = measure_posture(body, payload.jobId, address, comparison)
        if result["status"] != "withheld":
            return {"ok": True, "jobId": payload.jobId, "status": "ready", "addressMs": address, "comparisonMs": comparison, "movementGuide": guide}
    return {"ok": True, "jobId": payload.jobId, "status": "unavailable", "movementGuide": guide}
