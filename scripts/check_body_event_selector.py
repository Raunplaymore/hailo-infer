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
    phase_evidence_from_body,
    select_body_events,
    state_machine_score,
)
from app.services.swing_phase_decoder import MIN_PHASE_MS, decode_swing_phases  # noqa: E402
from app.services.coach_pipeline import _body_selector_is_operational  # noqa: E402


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


def test_forward_decoder_enforces_phase_order_and_duration() -> None:
    def row(time_ms: int, phase: str) -> dict:
        scores = {
            "ready": 0.0,
            "backswing": 0.0,
            "top": 0.0,
            "downswing": 0.0,
            "impact_candidate": 0.0,
            "follow_through": 0.0,
            "finish": 0.0,
        }
        scores[phase] = 1.0
        return {"timeMs": time_ms, "scores": scores}

    result = decode_swing_phases(
        [
            row(0, "ready"),
            row(100, "backswing"),
            row(200, "top"),
            row(240, "top"),
            row(320, "downswing"),
            row(400, "impact_candidate"),
            row(430, "follow_through"),
            row(540, "follow_through"),
            row(600, "finish"),
        ]
    )
    if not result.get("available"):
        raise AssertionError(f"expected a complete forward phase path, got {result}")
    events = result["events"]
    if not events["addressMs"] < events["topMs"] < events["impactMs"] < events["finishMs"]:
        raise AssertionError(f"decoder must emit monotonic events, got {events}")
    phases = [item["phase"] for item in result["phasePath"]]
    if phases != ["ready", "backswing", "top", "downswing", "impact_candidate", "follow_through", "finish"]:
        raise AssertionError(f"decoder must not skip or reorder phases, got {phases}")
    phase_times = {item["phase"]: item["timeMs"] for item in result["phasePath"]}
    for before, after in zip(phases, phases[1:]):
        if phase_times[after] - phase_times[before] < MIN_PHASE_MS[before]:
            raise AssertionError(f"decoder must respect {before} minimum duration, got {phase_times}")
    if result.get("confidence", 0.0) <= 0:
        raise AssertionError(f"complete decoder path must include confidence, got {result}")


def test_forward_decoder_rejects_finish_before_swing() -> None:
    def row(time_ms: int) -> dict:
        return {
            "timeMs": time_ms,
            "scores": {
                "ready": 0.1,
                "backswing": 0.0,
                "top": 0.0,
                "downswing": 0.0,
                "impact_candidate": 0.0,
                "follow_through": 0.0,
                "finish": 1.0,
            },
        }

    result = decode_swing_phases([row(index * 40) for index in range(10)])
    if result.get("available"):
        raise AssertionError(f"finish-only evidence must not become a valid swing, got {result}")


def test_pose_club_and_roi_motion_are_exposed_as_evidence() -> None:
    frames = []
    for index, x in enumerate((0.30, 0.38, 0.48, 0.55, 0.50, 0.42, 0.36, 0.31, 0.28)):
        frames.append(
            {
                "timeMs": index * 60,
                "keypoints": {
                    "left_wrist": [x, 0.42, 0.9],
                    "right_wrist": [x + 0.04, 0.44, 0.9],
                },
                "roiMotion": {
                    "upper": {"magnitude": 0.002 + index * 0.0001},
                    "torso": {"magnitude": 0.001},
                    "lower": {"magnitude": 0.0005},
                },
            }
        )
    club_x = (0.20, 0.21, 0.23, 0.27, 0.33, 0.40, 0.45, 0.48, 0.50)
    club = [{"t": index * 60, "x": x, "y": 0.5, "conf": 0.8} for index, x in enumerate(club_x)]
    evidence = phase_evidence_from_body({"frames": frames}, club)
    if len(evidence) != len(frames):
        raise AssertionError(f"expected pose evidence for every fixture frame, got {len(evidence)}")
    sources = evidence[4].get("sources", {})
    if not sources.get("pose", 0.0) > 0 or not sources.get("club", 0.0) > 0 or not sources.get("roiMotion", 0.0) > 0:
        raise AssertionError(f"pose, club, and ROI motion must all contribute evidence, got {sources}")


def test_confident_single_wrist_preserves_early_evidence() -> None:
    frames = [
        {
            "timeMs": index * 33,
            "keypoints": {
                "left_wrist": [0.45, 0.60, 0.08],
                "right_wrist": [0.55 + index * 0.01, 0.60, 0.91],
            },
        }
        for index in range(7)
    ]
    evidence = phase_evidence_from_body({"frames": frames})
    if not evidence or evidence[0]["timeMs"] != 0:
        raise AssertionError(f"weak opposite wrist must not discard the address window, got {evidence}")


def test_selector_uses_forward_decoder_for_complete_pose_club_roi_sequence() -> None:
    wrist_x = (0.30, 0.31, 0.35, 0.40, 0.47, 0.55, 0.55, 0.46, 0.34, 0.25, 0.19, 0.18, 0.18, 0.18, 0.18)
    club_x = (0.10, 0.11, 0.13, 0.16, 0.20, 0.25, 0.30, 0.38, 0.48, 0.54, 0.56, 0.565, 0.565, 0.565, 0.565)
    roi_motion = (0.0001, 0.0002, 0.0005, 0.001, 0.0015, 0.0001, 0.0002, 0.001, 0.002, 0.003, 0.001, 0.0001, 0.0001, 0.0001, 0.0001)
    frames = [
        {
            "timeMs": index * 60,
            "keypoints": {
                "left_wrist": [x, 0.42, 0.9],
                "right_wrist": [x + 0.04, 0.44, 0.9],
            },
            "roiMotion": {
                "upper": {"magnitude": roi_motion[index]},
                "torso": {"magnitude": roi_motion[index] / 3},
                "lower": {"magnitude": roi_motion[index] / 5},
            },
        }
        for index, x in enumerate(wrist_x)
    ]
    club = [{"t": index * 60, "x": x, "y": 0.5, "conf": 0.8} for index, x in enumerate(club_x)]
    result = select_body_events({"frames": frames}, club)
    if result.get("method") != "forward-phase-decoder":
        raise AssertionError(f"expected forward decoder to win over legacy candidates, got {result}")
    events = result.get("events", {})
    if not events.get("addressMs") < events.get("topMs") < events.get("impactMs") < events.get("finishMs"):
        raise AssertionError(f"forward selector events must remain monotonic, got {events}")


def test_pipeline_only_uses_confident_forward_decoder() -> None:
    confident = {"available": True, "method": "forward-phase-decoder", "confidence": 0.8}
    weak = {"available": True, "method": "forward-phase-decoder", "confidence": 0.29}
    if not _body_selector_is_operational("face_on", confident):
        raise AssertionError("confident forward decoder must be usable for either camera view")
    if _body_selector_is_operational("down_the_line", weak):
        raise AssertionError("weak forward decoder must not override club timing")


if __name__ == "__main__":
    test_state_machine_rejects_unrefined_early_top()
    test_state_machine_accepts_refined_compact_top()
    test_forward_decoder_enforces_phase_order_and_duration()
    test_forward_decoder_rejects_finish_before_swing()
    test_pose_club_and_roi_motion_are_exposed_as_evidence()
    test_confident_single_wrist_preserves_early_evidence()
    test_selector_uses_forward_decoder_for_complete_pose_club_roi_sequence()
    test_pipeline_only_uses_confident_forward_decoder()
    print("body event selector checks passed")
