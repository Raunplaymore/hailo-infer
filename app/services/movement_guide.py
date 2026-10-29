"""Observed 2D landmarks for visual review, never a swing-fault diagnosis."""
import math
from bisect import bisect_left, bisect_right
from statistics import median


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def build_movement_guide(body, address_ms, impact_ms):
    empty = {"version": "movement-guide-v1", "status": "unavailable"}
    meta = body.get("meta") or {}
    if not isinstance(meta, dict):
        return empty
    width, height, duration = (meta.get(k) for k in ("width", "height", "durationMs"))
    if not all(finite(v) and v > 0 for v in (width, height, duration)) or not finite(impact_ms) or not finite(address_ms) or not 0 <= address_ms < impact_ms <= min(duration, 30000):
        return empty
    frames = body.get("frames")
    if not isinstance(frames, list) or len(frames) > 1200:
        return empty
    frames = [f for f in frames if isinstance(f, dict) and finite(f.get("timeMs")) and 0 <= f["timeMs"] <= duration]
    frames.sort(key=lambda f: f["timeMs"])
    if len({f["timeMs"] for f in frames}) != len(frames):
        return empty

    def point(frame, kind):
        if frame.get("keypointSource") != "observed":
            return None
        points = frame.get("keypoints")
        if not isinstance(points, dict):
            return None
        keys = ("nose",) if kind == "head" else ("left_hip", "right_hip")
        values = [points.get(k) for k in keys]
        for p in values:
            if not isinstance(p, (list, tuple)) or len(p) < 3 or not all(finite(v) for v in p[:3]) or not (.02 < p[0] < .98 and .02 < p[1] < .98 and .75 <= p[2] <= 1):
                return None
        return [sum(p[i] for p in values) / len(values) for i in (0, 1)]

    times = [f["timeMs"] for f in frames]
    observed = {kind: [point(f, kind) for f in frames] for kind in ("head", "pelvis")}

    def window(time_ms, kind):
        lo, hi = bisect_left(times, time_ms - 65), bisect_right(times, time_ms + 65)
        samples = [(times[i], observed[kind][i]) for i in range(lo, hi)]
        samples = [(t, p) for t, p in samples if p is not None]
        if len(samples) < 3 or min(abs(t - time_ms) for t, _ in samples) > 40 or any(b[0] - a[0] > 70 for a, b in zip(samples, samples[1:])):
            return None
        x, y = (median(p[i] for _, p in samples) for i in (0, 1))
        if any(math.hypot((p[0]-x)*width/height, p[1]-y) > .02 for _, p in samples):
            return None
        return {"timeMs": min(samples, key=lambda s: abs(s[0]-time_ms))[0], "x": round(x, 5), "y": round(y, 5)}

    tracks = {}
    for kind in ("head", "pelvis"):
        baseline = None
        for offset in (0, -33, 33, -66, 66, 100, 150):
            t = max(100, address_ms) + offset
            if 0 <= t < impact_ms - 150:
                baseline = window(t, kind)
                if baseline:
                    break
        if baseline is None:
            continue
        points = []
        for f in frames:
            if baseline["timeMs"] <= f["timeMs"] <= min(duration, impact_ms + 250):
                value = window(f["timeMs"], kind)
                if value is not None:
                    points.append(value)
        points = list({p["timeMs"]: p for p in points}.values())
        near = [p for p in points if max(baseline["timeMs"] + 150, impact_ms - 500) <= p["timeMs"] <= impact_ms + 200]
        # Do not call a sparse surviving sample the largest movement of a phase.
        if len(near) < 6 or near[0]["timeMs"] > impact_ms - 350 or near[-1]["timeMs"] < impact_ms + 80 or any(b["timeMs"] - a["timeMs"] > 100 for a, b in zip(near, near[1:])):
            continue
        def distance(p):
            dx = (p["x"] - baseline["x"]) * width / height
            return math.hypot(dx, p["y"] - baseline["y"]) if kind == "head" else abs(dx)
        # shortcut: an impact candidate can be wrong; validate events before claiming phase accuracy.
        peak = max((p for p in near if kind != "head" or p["timeMs"] <= impact_ms), key=distance)
        tracks[kind] = {"baseline": baseline, "points": points, "focusTimeMs": peak["timeMs"],
                        "phase": "before" if peak["timeMs"] < impact_ms - 40 else "after" if peak["timeMs"] > impact_ms + 40 else "near"}
    if not tracks:
        return empty
    return {"version": "movement-guide-v1", "status": "ready", "width": width, "height": height,
            "impactMs": impact_ms, "tracks": tracks}
