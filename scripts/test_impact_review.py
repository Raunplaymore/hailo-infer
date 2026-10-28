import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import Settings
from app.services import impact_review as review


def pose_body(count):
    return {'meta': {'sampleStride': 1, 'sampledFrames': count},
            'frames': [{'frameIndex': i, 'timeMs': i * 33} for i in range(count)]}


class ImpactReviewChecks(unittest.TestCase):
    def test_isolated_derivative_and_withheld_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = Settings()
            settings.upload_dir = Path(temp) / 'uploads'
            settings.data_dir = Path(temp) / 'data'
            settings.upload_dir.mkdir()
            source = settings.upload_dir / 'swing.mp4'
            source.write_bytes(b'source-video')
            body = Path(temp) / 'body.json'
            body.write_text(json.dumps(pose_body(4)))
            original = settings.data_dir
            def fake_body(**kwargs):
                self.assertNotEqual(kwargs['settings'].data_dir, original)
                self.assertTrue(kwargs['settings'].data_dir.is_relative_to(original / 'impact-review'))
                self.assertEqual(kwargs['settings'].body_pose_target_fps, 30)
                self.assertFalse(kwargs['force'])
                return {'bodyPath': str(body)}
            with patch.object(review, 'source_frames', return_value=(30, [0, 33.333, 66.667, 100])), \
                 patch.object(review, 'analyze_body_video', side_effect=fake_body), \
                 patch.object(review, 'predict', return_value={'events': {'impactMs': 67}}):
                result = review.review_impact(settings, 'job-1', str(source))
            self.assertEqual(result['candidate'], {'frameIndex': 2, 'timeMs': 66.667})
            self.assertIsNone(result['publicValue'])
            self.assertFalse(result['automaticCoachingReady'])
            self.assertFalse(result['appliedToPrimary'])
            self.assertEqual(settings.data_dir, original)
            self.assertEqual(source.read_bytes(), b'source-video')
            self.assertNotIn('topMs', json.dumps(result))

    def test_variable_frame_rate_uses_real_timestamps(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = Settings()
            settings.upload_dir = Path(temp)
            settings.data_dir = Path(temp) / "data"
            source = Path(temp) / "swing.mp4"
            source.write_bytes(b"video")
            body = Path(temp) / "body.json"
            body.write_text(json.dumps(pose_body(5)))
            original_body = body.read_bytes()
            with patch.object(review, "source_frames", return_value=(30, [0, 10, 20, 100, 133])), \
                 patch.object(review, "analyze_body_video", return_value={"bodyPath": str(body)}), \
                 patch.object(review, "predict") as predict:
                predict.return_value = {"events": {"impactMs": 67}}
                result = review.review_impact(settings, "job", str(source))
                self.assertEqual([f['timeMs'] for f in predict.call_args.args[0]['frames']], [0, 10, 20, 100, 133])
                self.assertEqual(body.read_bytes(), original_body)
                self.assertEqual(result["candidate"], {"frameIndex": 3, "timeMs": 100})
                self.assertEqual(result["timingRevision"], 3)
                for invalid in (-1, 134, float("nan")):
                    predict.return_value = {"events": {"impactMs": invalid}}
                    with self.assertRaises(ValueError):
                        review.review_impact(settings, "job", str(source))

    def test_dense_body_reuse_without_original_mutation(self):
        import os
        import time
        with tempfile.TemporaryDirectory() as temp:
            settings = Settings()
            settings.upload_dir = Path(temp) / "uploads"
            settings.data_dir = Path(temp) / "data"
            settings.upload_dir.mkdir(); settings.data_dir.mkdir()
            source = settings.upload_dir / "swing.mp4"; source.write_bytes(b"video")
            time.sleep(.01)
            body = settings.data_dir / "dense.json"
            content = pose_body(4)
            content['source'] = {"inputPath": str(source)}
            content['meta'].update({"sampleMode": "target_fps", "sampleTargetFps": 30})
            body.write_text(json.dumps(content)); os.utime(body, ns=(source.stat().st_mtime_ns+1000000,)*2)
            before = body.read_bytes()
            with patch.object(review, "source_frames", return_value=(30,[0,33.333,66.667,100])), \
                 patch.object(review, "analyze_body_video") as extract, \
                 patch.object(review, "predict", return_value={"events":{"impactMs":67}}):
                result = review.review_impact(settings,"job",str(source),str(body),False)
            extract.assert_not_called()
            self.assertFalse(result["inputScopeConfirmed"])
            self.assertEqual(before,body.read_bytes())

    def test_sampled_pose_uses_its_source_frame_times(self):
        original = {'meta': {'sampleStride': 2, 'sampledFrames': 3},
                    'frames': [{'frameIndex': i, 'timeMs': i * 33} for i in (0, 2, 4)]}
        aligned = review.align_body_times(original, [0, 10, 50, 83, 120])
        self.assertEqual([f['timeMs'] for f in aligned['frames']], [0, 50, 120])
        self.assertEqual([f['timeMs'] for f in original['frames']], [0, 66, 132])

    def test_corrupt_isolated_pose_is_rebuilt(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            settings = Settings(); settings.upload_dir = Path(temp) / 'uploads'; settings.upload_dir.mkdir()
            settings.data_dir = Path(temp) / 'data'
            source = settings.upload_dir / 'video.mp4'; source.write_bytes(b'original video')
            cache = settings.data_dir / 'impact-review' / hashlib.sha256(source.read_bytes()).hexdigest() / 'body' / 'pose.json'
            cache.parent.mkdir(parents=True); cache.write_text('{')
            def rebuild(**kwargs):
                self.assertTrue(kwargs['force'])
                cache.write_text(json.dumps(pose_body(4)))
                return {'bodyPath': str(cache)}
            with patch.object(review, 'source_frames', return_value=(30, [0, 33, 67, 100])), \
                 patch.object(review, 'analyze_body_video', side_effect=rebuild) as extract, \
                 patch.object(review, 'predict', return_value={'events': {'impactMs': 67}}):
                result = review.review_impact(settings, 'job', str(source))
            extract.assert_called_once()
            self.assertEqual(result['candidate']['frameIndex'], 2)
            self.assertEqual(source.read_bytes(), b'original video')

    def test_unmatched_pose_indices_do_not_reach_model(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = Settings(); settings.upload_dir = Path(temp); settings.data_dir = Path(temp) / 'data'
            source = Path(temp) / 'video.mp4'; source.write_bytes(b'video')
            body = Path(temp) / 'body.json'
            with patch.object(review, 'source_frames', return_value=(30, [0, 10, 20, 100])), \
                 patch.object(review, 'analyze_body_video', return_value={'bodyPath': str(body)}), \
                 patch.object(review, 'predict') as predict:
                for indices in ([0, 1, 1, 3], [0, 1, 2, 4], [0, 1, 2], [0, 1, 2, True]):
                    payload = pose_body(len(indices))
                    for frame, index in zip(payload['frames'], indices): frame['frameIndex'] = index
                    body.write_text(json.dumps(payload))
                    with self.assertRaises(ValueError): review.review_impact(settings, 'job', str(source))
                predict.assert_not_called()

    def test_source_frame_count_mismatch_is_rejected(self):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 30 if prop == review.cv2.CAP_PROP_FPS else 4
        frames = {'frames': [{'best_effort_timestamp_time': str(t)} for t in (0, .033, .066)]}
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'video.mp4'; source.write_bytes(b'video')
            with patch.object(review.cv2, 'VideoCapture', return_value=capture), \
                 patch.object(review.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(frames))):
                with self.assertRaises(ValueError): review.source_frames(source)
        capture.release.assert_called_once()

    def test_source_frame_missing_timestamp_is_rejected(self):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 30 if prop == review.cv2.CAP_PROP_FPS else 4
        frames = {'frames': [{}, {}, {}, {}]}
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'video.mp4'; source.write_bytes(b'video')
            with patch.object(review.cv2, 'VideoCapture', return_value=capture), \
                 patch.object(review.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(frames))):
                with self.assertRaises(ValueError): review.source_frames(source)

    def test_variable_frame_clock_over_30_seconds_is_rejected(self):
        capture = MagicMock(); capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 30 if prop == review.cv2.CAP_PROP_FPS else 4
        frames = {'frames': [{'best_effort_timestamp_time': str(t)} for t in (0, 1, 2, 31)]}
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'video.mp4'; source.write_bytes(b'video')
            with patch.object(review.cv2, 'VideoCapture', return_value=capture), \
                 patch.object(review.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(frames))):
                with self.assertRaises(ValueError): review.source_frames(source)

    def test_source_clock_cache_invalidation_and_copy(self):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 30 if prop == review.cv2.CAP_PROP_FPS else 4
        response = SimpleNamespace(stdout=json.dumps({'frames': [{'best_effort_timestamp_time': str(t)} for t in (0, .033, .066, .099)]}))
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'video.mp4'; source.write_bytes(b'video')
            with patch.object(review.cv2, 'VideoCapture', return_value=capture), \
                 patch.object(review.subprocess, 'run', return_value=response) as probe:
                _, times = review.source_frames(source)
                times[1] = 999
                self.assertEqual(review.source_frames(source)[1], [0, 33, 66, 99])
                self.assertEqual(probe.call_count, 1)
                source.write_bytes(b'changed video')
                self.assertEqual(review.source_frames(source)[1], [0, 33, 66, 99])
                self.assertEqual(probe.call_count, 2)

    def test_source_clock_cache_invalidates_same_size_and_mtime_replacement(self):
        import os
        capture = MagicMock(); capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 30 if prop == review.cv2.CAP_PROP_FPS else 4
        response = SimpleNamespace(stdout=json.dumps({'frames': [{'best_effort_timestamp_time': str(t)} for t in (0, .033, .066, .099)]}))
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'video.mp4'; source.write_bytes(b'video')
            fixed = 1_600_000_000_000_000_000; os.utime(source, ns=(fixed, fixed))
            before = source.stat()
            with patch.object(review.cv2, 'VideoCapture', return_value=capture), \
                 patch.object(review.subprocess, 'run', return_value=response) as probe:
                review.source_frames(source)
                replacement = Path(temp) / 'replacement.mp4'; replacement.write_bytes(b'other'); os.utime(replacement, ns=(fixed, fixed))
                replacement.replace(source)
                after = source.stat()
                self.assertEqual((before.st_size, before.st_mtime_ns), (after.st_size, after.st_mtime_ns))
                self.assertNotEqual(before.st_ino, after.st_ino)
                review.source_frames(source)
                self.assertEqual(probe.call_count, 2)

    def test_source_clock_not_cached_if_video_changes_during_probe(self):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 30 if prop == review.cv2.CAP_PROP_FPS else 4
        response = SimpleNamespace(stdout=json.dumps({'frames': [{'best_effort_timestamp_time': str(t)} for t in (0, .033, .066, .099)]}))
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'video.mp4'; source.write_bytes(b'video')
            def change_source(*args, **kwargs):
                source.write_bytes(b'changed video')
                return response
            with patch.object(review.cv2, 'VideoCapture', return_value=capture), \
                 patch.object(review.subprocess, 'run', side_effect=change_source):
                with self.assertRaises(ValueError): review.source_frames(source)

    def test_source_changed_during_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = Settings(); settings.upload_dir = Path(temp); settings.data_dir = Path(temp) / 'data'
            source = Path(temp) / 'video.mp4'; source.write_bytes(b'video')
            body = Path(temp) / 'pose.json'; body.write_text(json.dumps(pose_body(4)))
            def changing_source(*args):
                source.write_bytes(b'changed video')
                return {'events': {'impactMs': 67}}
            with patch.object(review, 'source_frames', return_value=(30, [0,33,67,100])), \
                 patch.object(review, 'analyze_body_video', return_value={'bodyPath': str(body)}), \
                 patch.object(review, 'predict', side_effect=changing_source):
                with self.assertRaisesRegex(ValueError, '변경'):
                    review.review_impact(settings, 'job', str(source))

    def test_replaced_video_does_not_reuse_old_pose(self):
        import os
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); settings = Settings(); settings.upload_dir = root / 'uploads'; settings.upload_dir.mkdir()
            settings.data_dir = root / 'data'; settings.data_dir.mkdir()
            source = settings.upload_dir / 'video.mp4'; source.write_bytes(b'video')
            fixed = 1_600_000_000_000_000_000; os.utime(source, ns=(fixed, fixed))
            cached = settings.data_dir / 'cached.json'; value = pose_body(4)
            value['source'] = {'inputPath': str(source)}; value['meta'].update(sampleMode='target_fps', sampleTargetFps=30)
            cached.write_text(json.dumps(value))
            replacement = settings.upload_dir / 'replacement.mp4'; replacement.write_bytes(b'other'); os.utime(replacement, ns=(fixed, fixed))
            replacement.replace(source)
            self.assertGreaterEqual(source.stat().st_ctime_ns, cached.stat().st_ctime_ns)
            fresh = settings.data_dir / 'fresh.json'; fresh.write_text(json.dumps(pose_body(4)))
            with patch.object(review, 'source_frames', return_value=(30, [0,33,67,100])), \
                 patch.object(review, 'analyze_body_video', return_value={'bodyPath': str(fresh)}) as extract, \
                 patch.object(review, 'predict', return_value={'events': {'impactMs': 67}}):
                review.review_impact(settings, 'job', str(source), str(cached), False)
            extract.assert_called_once()

    def test_outside_root_rejected_before_video_read(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'outside.mp4'
            source.write_bytes(b'video')
            settings = Settings()
            settings.upload_dir = Path(temp) / 'uploads'
            settings.data_dir = Path(temp) / 'data'
            with self.assertRaises(ValueError), patch.object(review, 'source_frames') as frames:
                review.review_impact(settings, 'job', str(source))
            frames.assert_not_called()

    def test_invalid_or_disappeared_body_falls_back_but_outside_body_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = Settings()
            settings.upload_dir = root / 'uploads'; settings.upload_dir.mkdir()
            settings.data_dir = root / 'data'; settings.data_dir.mkdir()
            source = settings.upload_dir / 'swing.mp4'; source.write_bytes(b'video')
            bad = settings.data_dir / 'bad.json'; bad.write_text('{')
            wrong_source = settings.data_dir / 'wrong-source.json'
            wrong_source.write_text(json.dumps({'source': {'inputPath': str(root / 'another.mp4')}, 'meta': {'sampleMode': 'target_fps', 'sampleTargetFps': 30}}))
            sparse = settings.data_dir / 'sparse.json'
            sparse.write_text(json.dumps({'source': {'inputPath': str(source)}, 'meta': {'sampleMode': 'target_fps', 'sampleTargetFps': 15}}))
            wrong_shape = settings.data_dir / 'wrong-shape.json'; wrong_shape.write_text('[]')
            wrong_frames = settings.data_dir / 'wrong-frames.json'
            wrong_frames.write_text(json.dumps({'source': {'inputPath': str(source)}, 'meta': {'sampleMode': 'target_fps', 'sampleTargetFps': 30, 'sampleStride': 1, 'sampledFrames': 1}, 'frames': [{'frameIndex': 1}]}))
            replacement = settings.data_dir / 'replacement.json'; replacement.write_text(json.dumps(pose_body(4)))
            outside = root / 'outside.json'; outside.write_text('{}')
            with patch.object(review, 'source_frames', return_value=(30, [0, 33.333, 66.667, 100])), \
                 patch.object(review, 'analyze_body_video', return_value={'bodyPath': str(replacement)}) as extract, \
                 patch.object(review, 'predict', return_value={'events': {'impactMs': 67}}):
                for path in (bad, settings.data_dir / 'missing.json', wrong_source, sparse, wrong_shape, wrong_frames):
                    result = review.review_impact(settings, 'job', str(source), str(path), False)
                    self.assertEqual(result['candidate'], {'frameIndex': 2, 'timeMs': 66.667})
                self.assertEqual(extract.call_count, 6)
                with self.assertRaises(ValueError):
                    review.review_impact(settings, 'job', str(source), str(outside), False)

    def test_api_confirmation_busy_and_error_cleanup(self):
        from fastapi import HTTPException
        from pydantic import ValidationError
        from app import main
        from app.schemas import ImpactReviewRequest
        with self.assertRaises(ValidationError):
            ImpactReviewRequest(jobId="../bad", inputPath="/tmp/video", confirmedInput=True)
        with self.assertRaises(ValidationError):
            ImpactReviewRequest(jobId="valid", inputPath="/tmp/video", confirmedInput=False)
        automatic = ImpactReviewRequest(jobId="valid", inputPath="/tmp/video", automatic=True)
        self.assertFalse(automatic.confirmedInput)
        payload = ImpactReviewRequest(jobId="valid", inputPath="/tmp/video", confirmedInput=True)
        main.REVIEW_LOCK.acquire()
        try:
            with self.assertRaises(HTTPException) as raised:
                main.impact_review(payload)
            self.assertEqual(raised.exception.status_code, 409)
        finally:
            main.REVIEW_LOCK.release()
        with patch.object(main, "review_impact", side_effect=ValueError("private path")):
            with self.assertRaises(HTTPException) as raised:
                main.impact_review(payload)
            self.assertEqual(raised.exception.status_code, 400)
            self.assertNotIn("private path", raised.exception.detail)
        self.assertFalse(main.REVIEW_LOCK.locked())

    def test_practice_routes_handle_video_probe_timeout(self):
        from fastapi import HTTPException
        import subprocess
        from app import main
        payload = SimpleNamespace()
        for name, handler in (('review_saved_body', main.practice_review),
                              ('preview_saved_body', main.practice_preview),
                              ('review_coaching', main.coaching_review)):
            with patch.object(main, name, side_effect=subprocess.TimeoutExpired('ffprobe', 30)):
                with self.assertRaises(HTTPException) as raised:
                    handler(payload)
                self.assertEqual(raised.exception.status_code, 422)

    def test_model_integrity_and_nonvideo_rejection(self):
        self.assertEqual(review.file_sha256(review.MODEL), review.MODEL_SHA256)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'invalid.mp4'
            source.write_bytes(b'not video')
            with self.assertRaises(ValueError):
                review.source_frames(source)


if __name__ == '__main__':
    unittest.main()
