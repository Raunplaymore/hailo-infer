"""Explicit, non-production club preprocessing experiment runner."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scripts.club_preprocess_lab import is_within, run

from app.core.config import Settings

_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ClubPreprocessLabError(Exception):
    pass


def run_club_preprocess_lab(
    settings: Settings,
    job_id: str,
    input_path: str,
    body_path: str | None,
) -> dict[str, Any]:
    if not _SAFE_JOB_ID.fullmatch(job_id):
        raise ClubPreprocessLabError("invalid lab job ID")
    source = Path(input_path).expanduser().resolve()
    if not is_within(source, settings.upload_dir):
        raise ClubPreprocessLabError("source video must be below the upload directory")
    resolved_body: str | None = None
    if body_path:
        body = Path(body_path).expanduser().resolve()
        if not is_within(body, settings.data_dir / "body"):
            raise ClubPreprocessLabError("body artifact must be below the body data directory")
        resolved_body = str(body)
    workspace = settings.data_dir / "labs" / "club-preprocess" / f"web-{job_id}"
    args = argparse.Namespace(
        source=str(source),
        body=resolved_body,
        workspace=str(workspace),
        job_prefix=f"lab-{job_id}",
        camera_url=settings.camera_base_url,
        upload_dir=str(settings.upload_dir),
        timeout=180,
        model=None,
    )
    run(args)
    score_path = workspace / "score.json"
    if not score_path.is_file():
        raise ClubPreprocessLabError("lab completed without a score report")
    report = json.loads(score_path.read_text(encoding="utf-8"))
    return {
        "ok": True,
        "jobId": job_id,
        "labOnly": True,
        "report": report,
        "scorePath": str(score_path),
    }
