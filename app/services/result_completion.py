"""Completion rules shared by job creation and result retrieval."""

from __future__ import annotations

from typing import Any


def is_complete_coach_result(result: dict[str, Any] | None, analysis_version: str) -> bool:
    if not isinstance(result, dict):
        return False
    if "analysis" in result or "progress" in result:
        return False
    if result.get("ok") is not True:
        return False
    result_version = result.get("analysisVersion")
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    if result_version != analysis_version and meta.get("analysisVersion") != analysis_version:
        return False
    metrics = result.get("metrics")
    events = result.get("events")
    if not isinstance(metrics, dict) or not isinstance(events, dict):
        return False
    has_event = any(
        events.get(key) is not None
        for key in ("addressMs", "topMs", "impactMs", "finishMs", "address", "top", "impact", "finish")
    )
    event_validation = result.get("eventValidation") if isinstance(result.get("eventValidation"), dict) else {}
    events_withheld_for_quality = event_validation.get("status") == "withheld"
    has_metric = any(
        metrics.get(key) is not None
        for key in ("tempo", "swingPlane", "impactStability", "shaftPlane", "backswing", "trackingQuality")
    )
    return (has_event or events_withheld_for_quality) and has_metric
