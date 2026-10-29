import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError
from fastapi import HTTPException
from app.main import analyze_club_sidecar
from app.schemas import ClubSidecarRequest
from app.services.coach_pipeline import CoachError, _load_body_artifact, analyze_meta


class ClubBodySourceTest(unittest.TestCase):
    def test_explicit_body_does_not_use_untracked_default(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DATA_DIR": directory}):
            root = Path(directory)
            (root / "body").mkdir()
            (root / "body" / "job.json").write_text(json.dumps({"source": "default"}))
            invalid = root / "invalid.json"
            invalid.write_text("{")
            valid = root / "valid.json"
            valid.write_text(json.dumps({"source": "explicit"}))
            for source in (None, str(root / "missing.json"), str(invalid)):
                self.assertIsNone(_load_body_artifact("job", source, allow_default=False))
                self.assertEqual(_load_body_artifact("job", source), {"source": "default"})
            self.assertEqual(_load_body_artifact("job", str(valid), allow_default=False), {"source": "explicit"})
            with self.assertRaises(CoachError) as error:
                analyze_meta({"frames": [{}]}, "job", False, str(invalid), allow_body_fallback=False)
            self.assertEqual(error.exception.code, "BODY_UNAVAILABLE")

    def test_endpoint_confirms_fallback_policy_and_preserves_legacy_default(self):
        with patch("app.main.load_meta", return_value={}), patch("app.main.analyze_meta", return_value={}) as analyze:
            strict = analyze_club_sidecar(ClubSidecarRequest(jobId="job", metaPath="meta", allowBodyFallback=False))
            self.assertFalse(strict.bodyFallbackAllowed)
            self.assertFalse(analyze.call_args.kwargs["allow_body_fallback"])
            self.assertTrue(analyze.call_args.kwargs["withhold_ambiguous_club_heads"])
            self.assertTrue(analyze.call_args.kwargs["withhold_ambiguous_club_handles"])
            legacy = analyze_club_sidecar(ClubSidecarRequest(jobId="job", metaPath="meta"))
            self.assertTrue(legacy.bodyFallbackAllowed)
            self.assertTrue(analyze.call_args.kwargs["allow_body_fallback"])
        with self.assertRaises(ValidationError):
            ClubSidecarRequest(jobId="job", metaPath="meta", allowBodyFallback="false")
        with patch("app.main.load_meta", return_value={}), patch("app.main.analyze_meta", side_effect=CoachError("BODY_UNAVAILABLE", "missing")):
            with self.assertRaises(HTTPException) as error:
                analyze_club_sidecar(ClubSidecarRequest(jobId="job", metaPath="meta", allowBodyFallback=False))
            self.assertEqual(error.exception.detail["errorCode"], "BODY_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
