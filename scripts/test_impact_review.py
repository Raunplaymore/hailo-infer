import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import Settings
from app.services import impact_review as review


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
            body.write_text('{}')
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
            body.write_text('{}')
            with patch.object(review, "source_frames", return_value=(30, [0, 10, 20, 100, 133])), \
                 patch.object(review, "analyze_body_video", return_value={"bodyPath": str(body)}), \
                 patch.object(review, "predict") as predict:
                predict.return_value = {"events": {"impactMs": 67}}
                result = review.review_impact(settings, "job", str(source))
                self.assertEqual(result["candidate"], {"frameIndex": 3, "timeMs": 100})
                self.assertEqual(result["timingRevision"], 2)
                for invalid in (-1, 134, float("nan")):
                    predict.return_value = {"events": {"impactMs": invalid}}
                    with self.assertRaises(ValueError):
                        review.review_impact(settings, "job", str(source))

    def test_dense_body_reuse_without_original_mutation(self):
        import os
        with tempfile.TemporaryDirectory() as temp:
            settings = Settings()
            settings.upload_dir = Path(temp) / "uploads"
            settings.data_dir = Path(temp) / "data"
            settings.upload_dir.mkdir(); settings.data_dir.mkdir()
            source = settings.upload_dir / "swing.mp4"; source.write_bytes(b"video")
            body = settings.data_dir / "dense.json"
            content = {"source": {"inputPath": str(source)}, "meta": {"sampleMode": "target_fps", "sampleTargetFps": 30}}
            body.write_text(json.dumps(content)); os.utime(body, ns=(source.stat().st_mtime_ns+1000000,)*2)
            before = body.read_bytes()
            with patch.object(review, "source_frames", return_value=(30,[0,33.333,66.667,100])), \
                 patch.object(review, "analyze_body_video") as extract, \
                 patch.object(review, "predict", return_value={"events":{"impactMs":67}}):
                result = review.review_impact(settings,"job",str(source),str(body),False)
            extract.assert_not_called()
            self.assertFalse(result["inputScopeConfirmed"])
            self.assertEqual(before,body.read_bytes())

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

    def test_model_integrity_and_nonvideo_rejection(self):
        self.assertEqual(review.file_sha256(review.MODEL), review.MODEL_SHA256)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'invalid.mp4'
            source.write_bytes(b'not video')
            with self.assertRaises(ValueError):
                review.source_frames(source)


if __name__ == '__main__':
    unittest.main()
