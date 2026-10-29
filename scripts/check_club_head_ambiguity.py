import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.coach_pipeline import _ambiguous_club_head_frames, _build_overlay_payload, _select_best_track


def detection(label, x, confidence):
    return {"label": label, "conf": confidence, "bbox": [x, .3, .02, .02]}


frames = []
for i in range(500):
    detections = [detection("club_head", .2, .8), detection("club_handle", .4, .9)]
    if i == 101:
        detections.append(detection("club_head", .7, .7))
    if i == 102:
        detections.append(detection("club_head", .7, .29))
    if i == 103:
        detections.append({"label": "club_head", "conf": .9, "bbox": [2, 2, .02, .02]})
    frames.append({"frame": i, "sourceTimeMs": i * 17, "_t_ms": i * 17, "detections": detections})

original = copy.deepcopy(frames)
ambiguous = _ambiguous_club_head_frames(frames)
assert ambiguous == {101}
assert frames == original
heads = _select_best_track(frames, ["club_head"])
handles = _select_best_track(frames, ["club_handle"])
legacy = _build_overlay_payload(None, heads, handles, frames)
guarded = _build_overlay_payload(None, [p for p in heads if p["frame"] not in ambiguous], handles, frames)
assert len(heads) == 500 and legacy["clubFrames"][50].get("headValidUntilMs") is None
before = next(p for p in guarded["clubFrames"] if p["frame"] == 100)
after = next(p for p in guarded["clubFrames"] if p["frame"] == 102)
assert before["headValidUntilMs"] == 101 * 17 and after["headBreakBefore"] is True
assert before.get("handleValidUntilMs") is None and after.get("handleBreakBefore") is None
print("ambiguous candidates, source preservation and sampled overlay gaps passed")
