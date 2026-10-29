#!/usr/bin/env python3
"""Check pose clock alignment, missing observations and source identity."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.pose_overlay import read_pose_overlay
from app.services.body_pipeline import POSE_KEYPOINTS
from app.services.practice_review import load_practice_body


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); uploads = root / 'uploads'; data = root / 'data'
        uploads.mkdir(); data.mkdir(); source = uploads / 'video.mp4'; source.write_bytes(b'video')
        time.sleep(.01); body = data / 'pose.json'
        points = {'left_wrist': [.1, .2, .9], 'right_wrist': [.2, .3, .1],
                  'left_hip': [float('nan'), .4, .9], 'nose': [.3, .2], 'unknown': [.1, .2, .9]}
        raw = {'jobId': 'job', 'source': {'inputPath': str(source)}, 'meta': {'sampleStride': 1, 'sampledFrames': 4},
               'frames': [{'frameIndex': 0, 'timeMs': 0, 'keypoints': points, 'keypointSource': 'observed'},
                          {'frameIndex': 1, 'timeMs': 33, 'keypoints': None, 'keypointSource': 'missing'},
                          {'frameIndex': 2, 'timeMs': 66, 'keypoints': {'left_wrist': [.3, .2, .9]}, 'keypointSource': 'observed'},
                          {'frameIndex': 3, 'timeMs': 100, 'keypoints': points, 'keypointSource': 'interpolated'}]}
        body.write_text(json.dumps(raw)); settings = SimpleNamespace(upload_dir=uploads, data_dir=data)
        payload = SimpleNamespace(jobId='job', inputPath=str(source), bodyPath=str(body))
        clock = [0, 33.367, 66.733, 100.1]
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        before = (digest(source), digest(body))
        with patch('app.services.practice_review.source_frames', return_value=(30, clock)):
            result = read_pose_overlay(settings, payload)
            assert result['status'] == 'ready' and result['timeSource'] == 'video_pts'
            assert [f['timeMs'] for f in result['poseFrames']] == clock
            assert result['poseFrames'][0]['keypoints'] == {'left_wrist': [.1, .2, .9], 'right_wrist': [.2, .3, .1]}
            assert result['poseFrames'][1]['keypoints'] == {} and result['poseFrames'][3]['keypoints'] == {}
            assert (digest(source), digest(body)) == before
            other = copy.deepcopy(raw); other['jobId'] = 'other'; body.write_text(json.dumps(other))
            try: read_pose_overlay(settings, payload)
            except ValueError: pass
            else: raise AssertionError('different body job accepted')
            missing = copy.deepcopy(raw)
            for frame in missing['frames']: frame['keypoints'] = None; frame['keypointSource'] = 'missing'
            body.write_text(json.dumps(missing)); unavailable = read_pose_overlay(settings, payload)
            assert unavailable['status'] == 'unavailable' and unavailable['poseFrames'] == []
            body.write_text(json.dumps(raw))
            def changed(*args):
                aligned = load_practice_body(*args); body.write_text(json.dumps(raw) + ' '); return aligned
            with patch('app.services.pose_overlay.load_practice_body', side_effect=changed):
                try: read_pose_overlay(settings, payload)
                except ValueError: pass
                else: raise AssertionError('body changed during response accepted')
            with patch('app.services.pose_overlay.load_practice_body', return_value={'frames': [None] * 7201}):
                try: read_pose_overlay(settings, payload)
                except ValueError: pass
                else: raise AssertionError('unbounded pose response accepted')
            dense_points = {name: [.2,.3,.9] for name in POSE_KEYPOINTS}
            long = {'frames': [{'frameIndex': i, 'timeMs': i * 4, 'keypoints': dense_points}
                               for i in range(7200)]}
            long['frames'][3599]['keypoints'] = None
            with patch('app.services.pose_overlay.load_practice_body', return_value=long):
                result = read_pose_overlay(settings, payload)
                assert len(result['poseFrames']) == 7200 and result['poseFrames'][3599]['keypoints'] == {}
    print('pose overlay: actual frame clock, missing points, invalid data and identity guards passed')


if __name__ == '__main__': main()
