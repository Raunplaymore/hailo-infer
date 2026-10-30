"""Bounded 2D silhouette observations; these are not validated swing diagnoses."""
import json
import math
import re
import threading
from pathlib import Path
from statistics import median
from types import SimpleNamespace

from app.services.movement_guide import build_movement_guide
from app.services.impact_review import stat_identity
from app.services.practice_review import load_practice_body, number

VERSION = "dtl-coaching-review-v1"
CACHE_REVISION = 10
REVIEW_LOCK = threading.Lock()


class CoachingBusy(Exception):
    pass


def unavailable(job_id, reason):
    return {"ok": True, "jobId": job_id, "version": VERSION, "status": "unavailable",
            "diagnosisStatus": "unvalidated", "reason": reason, "evidence": None,
            "summary": "이 영상에서는 골반 뒤쪽 윤곽을 안정적으로 확인하지 못했습니다."}


def _exclusion_reason(result):
    excluded = result.get("excludedFrames")
    if (result.get("status") != "unavailable" or result.get("reason") != "SILHOUETTE_UNRELIABLE"
            or not isinstance(excluded, list) or not 7 <= len(excluded) <= 14
            or not all(isinstance(item, dict) and isinstance(item.get("reason"), str) for item in excluded)):
        return result
    for reason in ("CAMERA_MOTION_OR_BACKGROUND_UNCLEAR", "VIDEO_TIMING_UNAVAILABLE", "SILHOUETTE_UNRELIABLE"):
        if sum(item["reason"] == reason for item in excluded) > len(excluded) / 2:
            return {**result, "reason": reason}
    return {**result, "reason": "PHASE_COVERAGE_INSUFFICIENT"}


def summarize_observation(samples, job_id, impact_ms, width, height, direction):
    result = unavailable(job_id, "SILHOUETTE_UNRELIABLE")
    if direction not in (-1, 1) or not all(number(v) and v > 0 for v in (impact_ms, width, height)):
        return result
    if not 8 <= len(samples) <= 18 or any(not isinstance(s, dict) for s in samples):
        return result
    for s in samples:
        if not all(number(s.get(k)) for k in ("timeMs", "edgePx", "y", "uncertaintyPx")):
            return result
        if not (0 <= s["timeMs"] <= 30000 and .02 * width < s["edgePx"] < .98 * width
                and .02 < s["y"] < .98 and 0 <= s["uncertaintyPx"] <= 5):
            return result
        if not s.get("cameraStable"):
            return unavailable(job_id, "CAMERA_MOTION_OR_BACKGROUND_UNCLEAR")
        matrix = s.get("transform", [[1, 0, 0], [0, 1, 0]])
        if (not isinstance(matrix, list) or len(matrix) != 2
                or any(not isinstance(row, list) or len(row) != 3 or not all(number(v) for v in row) for row in matrix)):
            return unavailable(job_id, "CAMERA_MOTION_OR_BACKGROUND_UNCLEAR")
    samples = sorted(samples, key=lambda s: s["timeMs"])
    if len({s["timeMs"] for s in samples}) != len(samples):
        return result
    base = [s for s in samples if s.get("baseline")]
    before = [s for s in samples if not s.get("baseline") and impact_ms - 500 <= s["timeMs"] <= impact_ms - 40]
    if len(base) < 3 or len(before) < 6 or base[-1]["timeMs"] >= before[0]["timeMs"]:
        return unavailable(job_id, "PHASE_COVERAGE_INSUFFICIENT")
    if (before[0]["timeMs"] > impact_ms - 420 or before[-1]["timeMs"] < impact_ms - 110
            or any(b["timeMs"] - a["timeMs"] > 140 for a, b in zip(before, before[1:]))):
        return unavailable(job_id, "PHASE_COVERAGE_INSUFFICIENT")
    baseline = median(s["edgePx"] for s in base)
    baseline_y = median(s["y"] for s in base)
    spread = max(s["edgePx"] for s in base) - min(s["edgePx"] for s in base)
    def project(s, x, y):
        m = s.get("transform", [[1, 0, 0], [0, 1, 0]])
        return [(m[0][0]*x + m[0][1]*y + m[0][2])/width,
                (m[1][0]*x + m[1][1]*y + m[1][2])/height]
    points = []
    for sample in samples:
        x, y = project(sample, sample["edgePx"], sample["y"]*height)
        rx, ry = project(sample, baseline, baseline_y*height)
        top = project(sample, baseline, max(0, baseline_y-.16)*height)
        bottom = project(sample, baseline, min(1, baseline_y+.16)*height)
        if not all(.0 < v < 1 for v in (x, y, rx, ry, *top, *bottom)):
            return unavailable(job_id, "SILHOUETTE_UNRELIABLE")
        points.append({"timeMs": sample["timeMs"], "x": round(x, 6), "y": round(y, 6),
                       "referenceX": round(rx, 6), "referenceY": round(ry, 6),
                       "referenceTop": [round(v, 6) for v in top], "referenceBottom": [round(v, 6) for v in bottom]})
    if spread > 3:
        return unavailable(job_id, "ADDRESS_UNSTABLE")
    # shortcut: overlapping sampled bands do not prove anatomical correspondence; validate against independent labels.
    radius = max(3, round(height * .0125))
    if any(abs(s["y"] - baseline_y) * height > radius for s in base + before):
        return unavailable(job_id, "PELVIS_HEIGHT_UNCERTAIN")
    # Pixel uncertainty only; this is not an anatomical or golf-fault threshold.
    noise = max(2.0, spread * 2, max(s["uncertaintyPx"] + s.get("registrationErrorPx", 0) for s in samples) * 2)
    def image_x(sample):
        m = sample.get("transform", [[1, 0, 0], [0, 1, 0]])
        return m[0][0] * sample["edgePx"] + m[0][1] * sample["y"] * height + m[0][2]
    # A moving background can mimic camera motion, so keep only correction-insensitive observations.
    image_baseline = median(image_x(s) for s in base)
    if any(abs((image_x(s) - image_baseline) - (s["edgePx"] - baseline)) > noise for s in before):
        return unavailable(job_id, "CAMERA_BACKGROUND_AMBIGUOUS")
    shifts = [(s["edgePx"] - baseline) * direction for s in before]
    image_shifts = [(image_x(s) - image_baseline) * direction for s in before]
    windows = [i for i in range(2, len(before))
               if all(before[j]["timeMs"]-before[j-1]["timeMs"] <= 85 for j in (i-1, i))]
    if not windows:
        return unavailable(job_id, "PHASE_COVERAGE_INSUFFICIENT")
    sustained = [i for i in windows if min(shifts[i-2:i+1]) > noise]
    # A replay window must remain sustained under both background assumptions.
    agreed = [i for i in sustained if min(image_shifts[i-2:i+1]) > noise]
    if not agreed and (sustained or any(min(image_shifts[i-2:i+1]) > noise for i in windows)):
        return unavailable(job_id, "CAMERA_BACKGROUND_AMBIGUOUS")
    # A possible movement next to a gap cannot establish or rule out a sustained shift.
    if not agreed and any((shifts[i] > noise or image_shifts[i] > noise) and
                          ((i > 0 and before[i]["timeMs"]-before[i-1]["timeMs"] > 85) or
                           (i+1 < len(before) and before[i+1]["timeMs"]-before[i]["timeMs"] > 85))
                          for i in range(len(before))):
        return unavailable(job_id, "PHASE_COVERAGE_INSUFFICIENT")
    focus = before[max(agreed, key=lambda i: shifts[i])] if agreed else before[max(range(len(before)), key=lambda i: shifts[i])]
    observation = "toward_ball" if agreed else "no_clear_shift"
    return {**result, "status": "ready", "reason": None,
            "summary": "검사한 구간에서 골반 뒤쪽 윤곽이 시작 기준선보다 추정한 공 방향에 보입니다." if agreed else
                       "검사한 구간에서 골반 뒤쪽 윤곽의 뚜렷한 추정 공 방향 이동은 찾지 못했습니다.",
            "evidence": {"addressMs": min(base, key=lambda s: abs(s["edgePx"] - baseline))["timeMs"],
                         "startMs": before[0]["timeMs"], "endMs": before[-1]["timeMs"],
                         "focusMs": focus["timeMs"], "impactMs": impact_ms,
                         "width": width, "height": height, "ballDirection": "right" if direction == 1 else "left",
                         "baselineX": round(baseline / width, 6), "baselineY": round(baseline_y, 6),
                         "noisePx": round(noise, 2), "shiftPx": round((focus["edgePx"] - baseline) * direction, 2),
                         "cameraMotion": "similarity_corrected",
                         "observation": observation, "points": points}}



def _direction(body, address_ms):
    directions = []
    frames = body.get("frames")
    if not isinstance(frames, list) or len(frames) > 1200:
        return None
    def visible(p):
        return (isinstance(p, list) and len(p) >= 3 and all(number(v) for v in p[:3])
                and .02 < p[0] < .98 and .02 < p[1] < .98 and .75 <= p[2] <= 1)
    for frame in frames:
        if not isinstance(frame, dict) or frame.get("keypointSource") != "observed" or not number(frame.get("timeMs")) or abs(frame["timeMs"] - address_ms) > 100:
            continue
        points = frame.get("keypoints") or {}
        if not isinstance(points, dict):
            continue
        names = [f"{side}_{joint}" for joint in ("hip", "shoulder") for side in ("left", "right")]
        values = [points.get(k) for k in names]
        if not all(visible(p) for p in values):
            continue
        hx, sx = [sum(values[i+j][0] for j in (0, 1)) / 2 for i in (0, 2)]
        sy = (values[2][1] + values[3][1]) / 2
        # The far wrist is routinely occluded in DTL; use only visible wrists.
        wrists = [points.get(f"{side}_wrist") for side in ("left", "right") if visible(points.get(f"{side}_wrist"))]
        if (abs(sx-hx) < .025 or not wrists
                or any(abs(w[0]-hx) < .04 or (sx-hx)*(w[0]-hx) <= 0 or w[1] < sy+.04 for w in wrists)):
            continue
        directions.append(1 if sx > hx else -1)
    return directions[0] if len(directions) >= 3 and len(set(directions)) == 1 else None


def _silhouette(result, direction, width, height):
    import cv2
    import numpy as np
    if result.pose_landmarks is None or result.segmentation_mask is None:
        return None
    landmarks = result.pose_landmarks.landmark
    if any(p.visibility < .75 or not (.02 < p.x < .98 and .02 < p.y < .98) for p in (landmarks[23], landmarks[24])):
        return None
    hx, hy = [(getattr(landmarks[23], axis) + getattr(landmarks[24], axis)) / 2 for axis in ("x", "y")]
    mask = result.segmentation_mask
    _, labels, stats, _ = cv2.connectedComponentsWithStats((mask > .7).astype("uint8"))
    label = labels[int(hy * height), int(hx * width)]
    if label == 0 or stats[label, cv2.CC_STAT_AREA] < width * height * .015:
        return None
    if any(stats[i, cv2.CC_STAT_AREA] > stats[label, cv2.CC_STAT_AREA] * .3 for i in range(1, len(stats)) if i != label):
        return None
    radius = max(3, round(height * .0125))
    edges, ambiguous = [], []
    for y in range(max(0, int(hy*height)-radius), min(height, int(hy*height)+radius+1)):
        xs = np.flatnonzero(labels[y] == label)
        if not len(xs):
            continue
        edge = int(xs[0] if direction == 1 else xs[-1])
        if abs(edge-hx*width) > width*.18:
            return None
        edges.append(edge)
        for x in range(max(0, edge-5), min(width, edge+6)):
            if .3 < mask[y, x] < .7:
                ambiguous.append(x-edge)
    if len(edges) < radius * 1.5 or max(edges)-min(edges) > 8:
        return None
    uncertainty = max(1.0, (max(edges)-min(edges))/2, len(ambiguous)/len(edges))
    if uncertainty > 5:
        return None
    return {"edgePx": float(np.median(edges)), "y": hy, "uncertaintyPx": uncertainty}


def _background_motion(first_gray, gray, first_mask, mask):
    import cv2
    import numpy as np
    allowed = cv2.erode((first_mask < .1).astype("uint8") * 255, np.ones((17, 17), "uint8"))
    points = cv2.goodFeaturesToTrack(first_gray, 100, .02, 12, mask=allowed)
    if points is None or len(points) < 16:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(first_gray, gray, points, None)
    if moved is None:
        return None
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, first_gray, moved, None)
    if back is None:
        return None
    a, b = points[:, 0], moved[:, 0]
    keep = (status[:, 0] == 1) & (back_status[:, 0] == 1) & (np.linalg.norm(back[:, 0]-a, axis=1) < 1)
    height, width = gray.shape
    safe = cv2.erode((mask < .1).astype("uint8"), np.ones((17, 17), "uint8"))
    for i, (x, y) in enumerate(b):
        if not (0 <= x < width and 0 <= y < height) or not safe[min(height-1,max(0,int(y))), min(width-1,max(0,int(x)))]:
            keep[i] = False
    a, b = a[keep], b[keep]
    if len(a) < 16 or np.ptp(a[:, 0]) < width*.25 or np.ptp(a[:, 1]) < height*.15:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if matrix is None or inliers is None or inliers.mean() < .85:
        return None
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    rotation = abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    displacement = corners @ matrix[:, :2].T + matrix[:, 2] - corners
    # Bounded similarity only: perspective and independently moving backgrounds stay unsupported.
    if rotation > 1 or abs(scale-1) > .02 or np.any(np.abs(displacement) > [width*.05, height*.05]):
        return None
    # Check points not used to fit the validation transform, not just its RANSAC inliers.
    train = np.arange(len(a)) % 2 == 0
    for subset in (a[train], a[~train]):
        if np.ptp(subset[:, 0]) < width*.25 or np.ptp(subset[:, 1]) < height*.15:
            return None
    validation, _ = cv2.estimateAffinePartial2D(a[train], b[train], method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if validation is None:
        return None
    residual = np.linalg.norm(a[~train] @ validation[:, :2].T + validation[:, 2] - b[~train], axis=1)
    error = float(np.percentile(residual, 90))
    if np.median(residual) > .5 or error > 1:
        return None
    return matrix, error


def _stabilize_pose(result, matrix, width, height):
    import cv2
    import numpy as np
    inverse = cv2.invertAffineTransform(matrix)
    landmarks = []
    for point in result.pose_landmarks.landmark:
        x, y = inverse @ np.array([point.x*width, point.y*height, 1])
        landmarks.append(SimpleNamespace(x=x/width, y=y/height, visibility=point.visibility))
    mask = cv2.warpAffine(result.segmentation_mask, inverse, (width, height))
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks), segmentation_mask=mask)


def _observe(body, payload):
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except ImportError:
        return unavailable(payload.jobId, "ANALYSIS_RUNTIME_UNAVAILABLE")
    direction = _direction(body, payload.addressMs)
    if direction is None:
        return unavailable(payload.jobId, "BALL_DIRECTION_UNCLEAR")
    meta = body.get("meta")
    duration = meta.get("durationMs") if isinstance(meta, dict) else None
    if not number(duration) or not 0 < duration <= 30000 or payload.impactMs > duration:
        return unavailable(payload.jobId, "VIDEO_GEOMETRY_MISSING")
    start = payload.impactMs - 500
    times = [payload.addressMs + delta for delta in (0, 34, 68)]
    if times[-1] >= start or times[0] < 0:
        return unavailable(payload.jobId, "PHASE_COVERAGE_INSUFFICIENT")
    times += np.linspace(start, min(duration-1, payload.impactMs+100), 11).tolist()
    cap = cv2.VideoCapture(str(Path(payload.inputPath).resolve()))
    samples, excluded, first_gray, first_mask = [], [], None, None
    try:
        with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=True,
                                    min_detection_confidence=.5) as pose:
            # Keep the selected baseline samples; use a nearby frame only for registration.
            anchor_requested = payload.addressMs + 100
            if anchor_requested < start:
                try:
                    cap.set(cv2.CAP_PROP_POS_MSEC, anchor_requested)
                    anchor_ok, anchor_frame = cap.read()
                    anchor_actual = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if (anchor_ok and anchor_frame is not None and number(anchor_actual)
                            and abs(anchor_actual-anchor_requested) <= 50 and anchor_actual < start):
                        anchor_scale = min(1, 480/max(anchor_frame.shape[:2]))
                        anchor_frame = cv2.resize(anchor_frame, (round(anchor_frame.shape[1]*anchor_scale), round(anchor_frame.shape[0]*anchor_scale)))
                        anchor_pose = pose.process(cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2RGB))
                        if anchor_pose.pose_landmarks is not None and anchor_pose.segmentation_mask is not None:
                            first_gray, first_mask = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY), anchor_pose.segmentation_mask.copy()
                except (cv2.error, RuntimeError, OSError):
                    pass  # Optional reference failure retains the original registration path.
            for index, requested in enumerate(times):
                cap.set(cv2.CAP_PROP_POS_MSEC, requested)
                ok, frame = cap.read()
                actual = cap.get(cv2.CAP_PROP_POS_MSEC)
                if not ok or not number(actual) or abs(actual-requested) > 50:
                    excluded.append({"timeMs": round(requested, 3), "reason": "VIDEO_TIMING_UNAVAILABLE"})
                    continue
                scale = min(1, 480/max(frame.shape[:2]))
                frame = cv2.resize(frame, (round(frame.shape[1]*scale), round(frame.shape[0]*scale)))
                height, width = frame.shape[:2]
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.pose_landmarks is None or result.segmentation_mask is None:
                    excluded.append({"timeMs": round(actual, 3), "reason": "SILHOUETTE_UNRELIABLE"})
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion = (np.array([[1., 0, 0], [0, 1., 0]]), 0.) if first_gray is None else _background_motion(first_gray, gray, first_mask, result.segmentation_mask)
                if motion is None:
                    excluded.append({"timeMs": round(actual, 3), "reason": "CAMERA_MOTION_OR_BACKGROUND_UNCLEAR"})
                    continue
                matrix, error = motion
                stable = _stabilize_pose(result, matrix, width, height)
                sample = _silhouette(stable, direction, width, height)
                if sample is None:
                    excluded.append({"timeMs": round(actual, 3), "reason": "SILHOUETTE_UNRELIABLE"})
                    continue
                if first_gray is None:
                    first_gray, first_mask = gray, result.segmentation_mask.copy()
                samples.append({**sample, "timeMs": round(actual, 3), "baseline": index < 3, "cameraStable": True,
                                "transform": matrix.tolist(), "registrationErrorPx": error})
    except (cv2.error, RuntimeError, OSError):
        return unavailable(payload.jobId, "ANALYSIS_RUNTIME_UNAVAILABLE")
    finally:
        cap.release()
    if not samples:
        reason = max((item["reason"] for item in excluded), key=lambda reason: sum(item["reason"] == reason for item in excluded))
        result = unavailable(payload.jobId, reason)
    else:
        result = summarize_observation(samples, payload.jobId, payload.impactMs, width, height, direction)
    result["excludedFrames"] = excluded
    return _exclusion_reason(result)


def _identity(payload):
    files = []
    for value in (payload.inputPath, payload.bodyPath):
        path = Path(value).resolve(strict=True)
        stat = path.stat()
        if not path.is_file():
            raise ValueError("Expected saved files")
        files.append([str(path), *stat_identity(stat)])
    return {"version": VERSION, "algorithmRevision": CACHE_REVISION, "files": files, "addressMs": payload.addressMs, "impactMs": payload.impactMs,
            "viewpoint": payload.viewpoint}


def _read_cache(path, identity, job_id):
    try:
        value = json.loads(path.read_text())
        result = value["result"]
        if (value["identity"] == identity and result.get("version") == VERSION
                and result.get("jobId") == job_id and result.get("diagnosisStatus") == "unvalidated"
                and result.get("status") in ("ready", "unavailable")):
            return _exclusion_reason(result)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return None


def review_coaching(settings, payload):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", payload.jobId):
        raise ValueError("Invalid job id")
    impact = getattr(payload, "impactMs", None)
    if (payload.viewpoint != "down_the_line" or not number(impact) or not number(payload.addressMs)
            or not 0 <= payload.addressMs < impact <= 30000):
        return unavailable(payload.jobId, "PHASE_OR_VIEWPOINT_UNAVAILABLE")
    identity = _identity(payload)
    body = load_practice_body(settings, payload)
    guide = build_movement_guide(body, payload.addressMs, impact)
    if _identity(payload) != identity:
        raise ValueError("Input changed while reading pose")
    cache = settings.data_dir / "coaching-review" / f"{payload.jobId}.json"
    cached = _read_cache(cache, identity, payload.jobId)
    if cached is not None:
        return {**cached, "movementGuide": guide}
    if not REVIEW_LOCK.acquire(blocking=False):
        raise CoachingBusy("A silhouette observation is running")
    try:
        cached = _read_cache(cache, identity, payload.jobId)
        if cached is not None:
            return {**cached, "movementGuide": guide}
        result = _observe(body, payload)
        result["movementGuide"] = guide
        if _identity(payload) != identity:
            raise ValueError("Input changed during observation")
        if result.get("reason") == "ANALYSIS_RUNTIME_UNAVAILABLE":
            return result
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_suffix(".tmp")
        try:
            temp.write_text(json.dumps({"identity": identity, "result": result}, allow_nan=False))
            temp.replace(cache)
        finally:
            temp.unlink(missing_ok=True)
        return result
    finally:
        REVIEW_LOCK.release()
