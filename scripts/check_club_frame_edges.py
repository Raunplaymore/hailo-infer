"""A truncated detector box must be identified before its center is displayed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.coach_pipeline import _build_overlay_payload, _club_box_touches_frame_edge

point = {'x': .05, 'y': .5, 'w': .1, 'h': .1, 'frame': 0, 't': 0, 'conf': .9}
assert _club_box_touches_frame_edge(point) is True
for field, value in (('x', .95), ('y', .05), ('y', .95)):
    assert _club_box_touches_frame_edge({**point, 'x': .5, 'y': .5, field: value}) is True
safe = {**point, 'x': .5}
assert _club_box_touches_frame_edge(safe) is False
assert _club_box_touches_frame_edge({**safe, 'w': 0}) is None
assert _club_box_touches_frame_edge({**safe, 'w': float('nan')}) is None
overlay = _build_overlay_payload(None, [point, {**safe, 'frame': 1}], [{**safe, 'frame': 0}])
assert overlay['clubFrames'][0]['head']['frameEdge'] is True
assert overlay['clubFrames'][0]['handle']['frameEdge'] is False
assert overlay['clubFrames'][1]['head']['frameEdge'] is False
print('club frame edge checks passed')
