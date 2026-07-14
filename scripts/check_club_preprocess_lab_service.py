#!/usr/bin/env python3
"""Regression checks for the web-facing, lab-only service boundary."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.services import club_preprocess_lab as lab_service


def test_web_lab_boundary() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        settings = Settings()
        settings.upload_dir = root / "uploads"
        settings.data_dir = root / "data"
        source = settings.upload_dir / "swing.mp4"
        body = settings.data_dir / "body" / "shot.json"
        source.parent.mkdir(parents=True)
        body.parent.mkdir(parents=True)
        source.write_bytes(b"video")
        body.write_text("{}", encoding="utf-8")

        original_run = lab_service.run

        def fake_run(args):
            workspace = Path(args.workspace)
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "score.json").write_text(
                json.dumps({"decision": "no_candidate", "results": {}, "candidates": [], "guardrail": "review"}),
                encoding="utf-8",
            )
            return 0

        lab_service.run = fake_run
        try:
            result = lab_service.run_club_preprocess_lab(settings, "shot", str(source), str(body))
        finally:
            lab_service.run = original_run

        assert result["ok"] is True
        assert result["labOnly"] is True
        assert result["report"]["decision"] == "no_candidate"
        try:
            lab_service.run_club_preprocess_lab(settings, "shot", "/tmp/outside.mp4", str(body))
        except lab_service.ClubPreprocessLabError:
            pass
        else:
            raise AssertionError("outside source path must be rejected")


if __name__ == "__main__":
    test_web_lab_boundary()
    print("club preprocess lab service checks passed")
