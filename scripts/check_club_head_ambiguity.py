import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.coach_pipeline import _ambiguous_club_frames, _build_overlay_payload, _select_best_track


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
    if i == 201:
        detections.append(detection("club_handle", .6, .7))
    if i == 202:
        detections.append(detection("club_handle", .6, .29))
    if i == 203:
        detections.append({"label": "club_handle", "conf": .9, "bbox": [2, 2, .02, .02]})
    if i == 204:
        detections.append(detection("grip", .6, .3))
    frames.append({"frame": None if i == 204 else i, "sourceTimeMs": i * 17, "_t_ms": i * 17, "detections": detections})

original = copy.deepcopy(frames)
ambiguous = _ambiguous_club_frames(frames, ['club_head'])
assert ambiguous == {101}
ambiguous_handles = _ambiguous_club_frames(frames, ['club_handle', 'grip'])
assert ambiguous_handles == {201, 204}
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
guarded_handles = _build_overlay_payload(None, heads, [p for p in handles if p['frame'] not in ambiguous_handles], frames)
before_handle = next(p for p in guarded_handles['clubFrames'] if p['frame'] == 200)
after_handle = next(p for p in guarded_handles['clubFrames'] if p['frame'] == 202)
assert before_handle['handleValidUntilMs'] == 201 * 17 and after_handle['handleBreakBefore'] is True
assert before_handle.get('headValidUntilMs') is None and after_handle.get('headBreakBefore') is None
print("ambiguous candidates, source preservation and sampled overlay gaps passed")
