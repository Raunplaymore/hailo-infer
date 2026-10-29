"""Ambiguous V2 grips must not create displayed or compared shaft segments."""
import copy
import math
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services import coach_pipeline
from app.services.club_observation import observe


def fixture():
    frames, wrists = [], []
    for index in range(26):
        time = index * 20
        x, y = .5, .2 + (time - 240) ** 2 / 1_000_000
        angle = math.radians(30 + time * .3)
        head_x, head_y = x + .08 * math.cos(angle), y - .04 * math.sin(angle)
        detections = [
            {'label': 'club_head', 'conf': .9, 'bbox': [head_x - .01, head_y - .01, .02, .02]},
            {'label': 'club_handle', 'conf': .9, 'bbox': [x - .01, y - .01, .02, .02]},
        ]
        if index == 16:
            detections.append({'label': 'club_handle', 'conf': .7, 'bbox': [x + .03, y - .01, .02, .02]})
        if index == 17:
            detections.append({'label': 'club_handle', 'conf': .29, 'bbox': [x + .03, y - .01, .02, .02]})
        if index == 18:
            detections.append({'label': 'club_handle', 'conf': .9, 'bbox': [2, 2, .02, .02]})
        frames.append({'frame': index, 'sourceTimeMs': time, 'detections': detections})
        wrists.append({'frame': index, 't': time, 'x': x, 'y': y, 'conf': .9})
    return {'fps': 50, 'width': 1000, 'height': 2000, 'durationMs': 500,
            'viewpoint': 'down_the_line', 'handedness': 'right', 'frames': frames}, wrists


meta, wrists = fixture()
validation = {'status': 'usable', 'message': 'fixture', 'warnings': [], 'codes': [], 'metricAvailability': {}}
with patch.dict(os.environ, {'REFERENCE_COACHING_ENABLED': '0', 'VISIBLE_GRIP_TRACK_ENABLED': '0'}), \
     patch.object(coach_pipeline, '_wrist_track_from_body', return_value=wrists), \
     patch.object(coach_pipeline, '_segment_events', return_value=(0, 12, 23, 25, 'fixture', None)), \
     patch.object(coach_pipeline, '_validate_event_evidence', side_effect=lambda *a, **k: copy.deepcopy(validation)):
    legacy = coach_pipeline.analyze_meta(copy.deepcopy(meta), 'fixture', True)
    guarded = coach_pipeline.analyze_meta(copy.deepcopy(meta), 'fixture', True, withhold_ambiguous_club_handles=True)

legacy_rows = {r['frame']: r for r in legacy['overlay']['clubFrames']}
guarded_rows = {r['frame']: r for r in guarded['overlay']['clubFrames']}
assert 'head' in legacy_rows[16] and 'handle' in legacy_rows[16]
assert 'head' in guarded_rows[16] and 'handle' not in guarded_rows[16]
assert 'handle' in guarded_rows[17] and 'handle' in guarded_rows[18]
assert guarded_rows[15]['handleValidUntilMs'] == 320
assert 'headValidUntilMs' not in guarded_rows[15] and 'headBreakBefore' not in guarded_rows[17]
assert guarded['metrics']['trackingQuality']['ambiguousClubHandleFrames'] == 1
assert guarded['events'] == legacy['events']
legacy['events']['takeawayMs'] = guarded['events']['takeawayMs'] = 0
assert observe(legacy, 500)['phases'][-1]['status'] == 'observation'
assert observe(guarded, 500)['phases'][-1]['status'] == 'withheld'
assert meta['frames'][16]['detections'][1]['conf'] == .9
print('ambiguous grips withheld, neighboring points and primary events preserved')
