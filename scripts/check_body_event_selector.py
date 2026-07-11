#!/usr/bin/env python3
"""Regression checks for the pose body event selector."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.body_event_selector import (  # noqa: E402
    early_top_cluster_candidate,
    feature_vote_events_from_clusters,
    state_machine_score,
)


def test_state_machine_rejects_unrefined_early_top() -> None:
    clusters = [
        {"t": 50.0, "weight": 9.6, "count": 9, "sources": ["early"]},
        {"t": 260.0, "weight": 9.8, "count": 8, "sources": ["top"]},
        {"t": 522.0, "weight": 12.1, "count": 12, "sources": ["impact"]},
        {"t": 767.0, "weight": 7.6, "count": 7, "sources": ["finish"]},
    ]

    early_events, early_debug = early_top_cluster_candidate(clusters)
    if not early_events:
        raise AssertionError("early top candidate should exist for regression fixture")
    if early_events["topMs"] != 50:
        raise AssertionError(f"expected unrefined early top at 50 ms, got {early_events}")

    fallback = feature_vote_events_from_clusters(clusters, 50.0, use_offset_model=True)
    if not fallback.get("available"):
        raise AssertionError("offset fallback candidate should exist for regression fixture")
    fallback_events = fallback["events"]
    if fallback_events["topMs"] <= 120:
        raise AssertionError(f"fallback must move top out of the impossible early window: {fallback_events}")

    early_score, early_reasons = state_machine_score(
        early_events,
        early_debug,
        "feature-vote-early-top-cluster",
    )
    fallback_score, fallback_reasons = state_machine_score(
        fallback_events,
        fallback.get("debug", {}),
        "feature-vote-offset-fallback",
    )

    if "top_gap_too_short_without_refinement" not in early_reasons:
        raise AssertionError(f"early candidate must be explicitly penalized, got {early_reasons}")
    if fallback_reasons:
        raise AssertionError(f"fallback should be a clean state-machine sequence, got {fallback_reasons}")
    if not fallback_score < early_score:
        raise AssertionError(
            f"state machine must prefer fallback top over early top: early={early_score}, fallback={fallback_score}"
        )


def test_state_machine_accepts_refined_compact_top() -> None:
    events = {"addressMs": 0, "topMs": 115, "impactMs": 310, "finishMs": 650}
    score, reasons = state_machine_score(
        events,
        {"topRefined": True},
        "feature-vote-early-top-cluster",
    )
    if not score < 3.0:
        raise AssertionError(f"refined compact top should remain usable, score={score}, reasons={reasons}")
    if "top_gap_too_short_without_refinement" in reasons:
        raise AssertionError(f"refined compact top should not get unrefined penalty: {reasons}")


if __name__ == "__main__":
    test_state_machine_rejects_unrefined_early_top()
    test_state_machine_accepts_refined_compact_top()
    print("body event selector checks passed")
