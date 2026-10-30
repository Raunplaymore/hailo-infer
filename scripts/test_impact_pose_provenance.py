import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.services import impact_review as review


class PoseProvenanceChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.settings = SimpleNamespace(upload_dir=root / 'uploads', data_dir=root / 'data', body_pose_target_fps=0)
        self.settings.upload_dir.mkdir(); self.settings.data_dir.mkdir()
        self.video = self.settings.upload_dir / 'video.mp4'; self.video.write_bytes(b'synthetic video')
        self.times = [0, 33, 67, 100]

    def pose(self, *, provided=False):
        body = {'meta': {'sampleStride': 1, 'sampledFrames': 4},
                'frames': [{'frameIndex': i, 'timeMs': i * 33} for i in range(4)]}
        if provided:
            body['source'] = {'inputPath': str(self.video)}
            body['meta'].update(sampleMode='target_fps', sampleTargetFps=30)
        return body

    def write_pose(self, kind):
        if kind == 'provided':
            time.sleep(.003)
            path = self.settings.data_dir / 'provided.json'
            path.write_text(json.dumps(self.pose(provided=True)))
            os.utime(path, ns=(self.video.stat().st_mtime_ns + 1_000_000,) * 2)
            return path, path
        digest = hashlib.sha256(self.video.read_bytes()).hexdigest()
        path = self.settings.data_dir / 'impact-review' / digest / 'body' / 'pose.json'
        path.parent.mkdir(parents=True); path.write_text(json.dumps(self.pose()))
        invalid = self.settings.data_dir / 'invalid.json'; invalid.write_text('{')
        return path, invalid

    def call(self, offered, predict):
        with patch.object(review, 'source_frames', return_value=(30, self.times)), \
             patch.object(review, 'analyze_body_video') as extract, \
             patch.object(review, 'predict', side_effect=predict):
            result = review.review_impact(self.settings, 'job', str(self.video), str(offered), False)
        extract.assert_not_called()
        return result

    def test_reports_the_file_actually_used(self):
        for kind in ('provided', 'isolated'):
            with self.subTest(kind=kind):
                actual, offered = self.write_pose(kind)
                result = self.call(offered, lambda *_: {'events': {'impactMs': 67}})
                expected = review.pose_stat_token(review.stat_identity(actual.stat()))
                self.assertEqual(result['poseProvenance'], {'version': 1, 'kind': kind, 'statToken': expected})
                actual.unlink()

    def test_replaced_pose_during_prediction_is_rejected(self):
        for kind in ('provided', 'isolated'):
            with self.subTest(kind=kind):
                actual, offered = self.write_pose(kind)
                def replace(*_):
                    replacement = actual.with_name('replacement.json')
                    replacement.write_text(json.dumps(self.pose(provided=kind == 'provided')))
                    replacement.replace(actual)
                    return {'events': {'impactMs': 67}}
                with self.assertRaisesRegex(ValueError, '자세'):
                    self.call(offered, replace)
                actual.unlink()

    def test_isolated_symlink_to_another_pose_is_rejected(self):
        actual, offered = self.write_pose('isolated')
        other = self.settings.data_dir / 'other.json'; other.write_text(json.dumps(self.pose()))
        actual.unlink(); actual.symlink_to(other)
        with self.assertRaisesRegex(ValueError, '자세'):
            self.call(offered, lambda *_: {'events': {'impactMs': 67}})

    def test_fresh_extractor_cannot_return_a_different_pose_file(self):
        other = self.settings.data_dir / 'other.json'; other.write_text(json.dumps(self.pose()))
        with patch.object(review, 'source_frames', return_value=(30, self.times)), \
             patch.object(review, 'analyze_body_video', return_value={'bodyPath': str(other)}), \
             patch.object(review, 'predict') as predict:
            with self.assertRaisesRegex(ValueError, '자세'):
                review.review_impact(self.settings, 'job', str(self.video))
            predict.assert_not_called()

    def test_pose_replaced_during_read_is_rejected(self):
        actual, _ = self.write_pose('isolated')
        original_read = Path.read_text
        def replace_after_read(path, *args, **kwargs):
            content = original_read(path, *args, **kwargs)
            replacement = path.with_name('replacement.json')
            replacement.write_text(json.dumps(self.pose()))
            replacement.replace(path)
            return content
        with patch.object(Path, 'read_text', autospec=True, side_effect=replace_after_read):
            with self.assertRaisesRegex(ValueError, '자세'):
                review.read_stable_pose(actual)

    def test_provided_symlink_retargeted_during_prediction_is_rejected(self):
        actual, _ = self.write_pose('provided')
        other = self.settings.data_dir / 'other.json'; other.write_bytes(actual.read_bytes())
        offered = self.settings.data_dir / 'offered.json'; offered.symlink_to(actual)
        def retarget(*_):
            offered.unlink(); offered.symlink_to(other)
            return {'events': {'impactMs': 67}}
        with self.assertRaisesRegex(ValueError, '자세'):
            self.call(offered, retarget)


if __name__ == '__main__':
    unittest.main()
