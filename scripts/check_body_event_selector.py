#!/usr/bin/env python3
"""Regression checks for the pose body event selector."""

from __future__ import annotations

import json
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
from app.services.coach_pipeline import (  # noqa: E402
    _body_selector_is_operational,
    _filter_track_by_bbox,
    _filter_track_by_wrist,
    _impact_stability,
    _refine_late_body_impact,
    _normalize_times,
    _validate_event_evidence,
)
from app.services.result_completion import is_complete_coach_result  # noqa: E402


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


def test_finish_adjacent_impact_uses_wrist_candidate_without_club_head() -> None:
    fixture = json.loads(
        (ROOT / "fixtures/event_labels/2a9e707c-62bd-45ec-9527-d3ce9189677c.json").read_text()
    )
    labels = fixture["labels"]
    observed = fixture["observed"]
    events = {
        "addressMs": labels["addressMs"],
        "topMs": labels["topMs"],
        "impactMs": observed["decoderImpactMs"],
        "finishMs": labels["finishMs"],
    }
    wrist_impact = {"t": observed["wristImpactMs"]}
    refined, changed = _refine_late_body_impact(events, wrist_impact, observed["confirmedClubHeadFrames"])
    if not changed or refined.get("impactMs") != observed["wristImpactMs"]:
        raise AssertionError(f"late production impact must move to the wrist candidate, got {refined}")
    if abs(refined["impactMs"] - labels["impactMs"]) > fixture["toleranceMs"]:
        raise AssertionError(f"refined impact must fall inside the hand-labeled video window, got {refined}")
    if refined.get("impactRefinedFromMs") != observed["decoderImpactMs"]:
        raise AssertionError(f"refinement must preserve the decoder candidate for audit, got {refined}")

    trusted, changed = _refine_late_body_impact(events, wrist_impact, 3)
    if changed or trusted.get("impactMs") != observed["decoderImpactMs"]:
        raise AssertionError(f"confirmed club-head evidence must retain decoder impact, got {trusted}")

    plausible = {"addressMs": 0, "topMs": 567, "impactMs": 900, "finishMs": 1467}
    retained, changed = _refine_late_body_impact(plausible, {"t": 800}, 0)
    if changed or retained.get("impactMs") != 900:
        raise AssertionError(f"a plausible decoder impact must not be rewritten, got {retained}")

    validation = _validate_event_evidence(
        body_events=refined,
        wrist_top={"t": labels["topMs"]},
        wrist_impact=wrist_impact,
        club_head_track=[],
        club_handle_track=[],
        club_track=[],
        body_selector_confidence=0.59,
    )
    impact_quality = validation.get("eventQuality", {}).get("impact", {})
    if impact_quality.get("status") != "reference" or impact_quality.get("source") != "pose_wrist_refinement":
        raise AssertionError(f"refined pose impact must remain visible as reference evidence, got {validation}")
    if validation.get("metricAvailability", {}).get("impact") != "withheld":
        raise AssertionError(f"pose reference impact must not unlock impact coaching, got {validation}")


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


def test_event_validation_withholds_conflicting_or_missing_club_evidence() -> None:
    conflict = _validate_event_evidence(
        body_events={"topMs": 2164, "impactMs": 2431},
        wrist_top={"t": 533},
        wrist_impact={"t": 699},
        club_head_track=[],
    )
    if conflict.get("status") != "withheld":
        raise AssertionError(f"conflicting sources must withhold event metrics, got {conflict}")
    expected = {"CLUB_TRACK_INSUFFICIENT_AT_IMPACT"}
    if not expected.issubset(set(conflict.get("codes", []))):
        raise AssertionError(f"expected explicit failure codes, got {conflict}")
    if conflict.get("warnings") != ["POSE_EVENT_SOURCE_DIVERGENCE"]:
        raise AssertionError(f"pose-derived disagreement must be a deduplicated warning, got {conflict}")

    usable = _validate_event_evidence(
        body_events={"topMs": 500, "impactMs": 700},
        wrist_top={"t": 520},
        wrist_impact={"t": 715},
        club_head_track=[
            {"t": 620, "x": 0.4, "y": 0.5, "conf": 0.8},
            {"t": 690, "x": 0.5, "y": 0.5, "conf": 0.8},
            {"t": 740, "x": 0.6, "y": 0.5, "conf": 0.8},
        ],
    )
    if usable.get("status") != "usable":
        raise AssertionError(f"consistent evidence must remain usable, got {usable}")


def test_production_reference_impact_fixture_is_partial_not_abandoned() -> None:
    fixture_path = ROOT / "fixtures" / "event_validation" / "c04d1b58-026d-490f-939d-80c52ccc7781.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    tracks = fixture["acceptedTracks"]
    evidence = _validate_event_evidence(
        body_events=fixture["bodyEvents"],
        wrist_top=fixture["legacyWristEvents"]["top"],
        wrist_impact=fixture["legacyWristEvents"]["impact"],
        club_head_track=tracks["head"],
        club_handle_track=tracks["handle"],
        club_track=tracks["club"],
        body_selector_confidence=fixture["bodySelectorConfidence"],
    )
    expected = fixture["expected"]
    if evidence.get("status") != expected["status"]:
        raise AssertionError(f"production fixture must retain reference events, got {evidence}")
    if evidence.get("codes") != [expected["code"]]:
        raise AssertionError(f"reference-only impact must have one stable reason code, got {evidence}")
    if evidence.get("warnings") != [expected["warning"]]:
        raise AssertionError(f"pose source divergence must be diagnostic only, got {evidence}")
    if evidence.get("eventQuality", {}).get("impact", {}).get("status") != expected["impactQuality"]:
        raise AssertionError(f"impact must be exposed as reference, got {evidence}")
    if evidence.get("metricAvailability", {}).get("tempo") != expected["tempoAvailability"]:
        raise AssertionError(f"reference impact must not unlock tempo, got {evidence}")


def test_pose_events_remain_partial_when_impact_is_withheld() -> None:
    evidence = _validate_event_evidence(
        body_events={"addressMs": 0, "topMs": 900, "impactMs": 1200, "finishMs": 1700},
        wrist_top={"t": 910},
        wrist_impact={"t": 1210},
        club_head_track=[],
        club_handle_track=[],
        club_track=[],
        body_selector_confidence=0.82,
    )
    if evidence.get("status") != "partial":
        raise AssertionError(f"usable pose events must not become a fully withheld analysis, got {evidence}")
    event_quality = evidence.get("eventQuality", {})
    for event_key in ("address", "top", "finish"):
        if event_quality.get(event_key, {}).get("status") != "reference":
            raise AssertionError(f"{event_key} must remain a pose reference event, got {evidence}")
    if event_quality.get("impact", {}).get("status") != "withheld":
        raise AssertionError(f"impact must stay withheld without club evidence, got {evidence}")
    if any(value != "withheld" for value in evidence.get("metricAvailability", {}).values()):
        raise AssertionError(f"pose-only evidence must not unlock club-dependent metrics, got {evidence}")


def test_quality_withheld_result_is_still_complete() -> None:
    withheld = {
        "ok": True,
        "analysisVersion": "hailo-coach-service7-v11",
        "events": {"addressMs": None, "topMs": None, "impactMs": None, "finishMs": None},
        "metrics": {"trackingQuality": {"label": "weak"}},
        "eventValidation": {"status": "withheld", "codes": ["CLUB_TRACK_INSUFFICIENT"]},
    }
    if not is_complete_coach_result(withheld, "hailo-coach-service7-v11"):
        raise AssertionError("quality-withheld result must be visible as a completed analysis")


def test_meta_timeline_prefers_video_frame_clock_over_inference_clock() -> None:
    fixture_path = ROOT / "fixtures" / "event_labels" / "957e5457-4d13-46bf-88c6-65c467af8487.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    timeline = fixture["timeline"]
    last_frame = int(timeline["lastFrame"])
    raw_last_ms = float(timeline["rawInferenceLastTimeMs"])
    frames = [
        {
            "frame": None if timeline.get("frameFieldMissing") else frame_index,
            "timeMs": round(frame_index * raw_last_ms / last_frame),
        }
        for frame_index in range(last_frame + 1)
    ]
    times = _normalize_times(frames, float(timeline["fps"]), float(timeline["durationMs"]))
    expected_last_ms = float(timeline["expectedVideoLastTimeMs"])
    if abs((times[-1] or 0) - expected_last_ms) > 2.0:
        raise AssertionError(f"frame/FPS clock must repair this golden video's compressed inference timestamps, got {times[-1]}")

    for event_key, frame_index in timeline["poseEventFrames"].items():
        expected_ms = float(fixture["labels"][event_key])
        actual_ms = times[int(frame_index)]
        if actual_ms is None or abs(actual_ms - expected_ms) > float(fixture["toleranceMs"]):
            raise AssertionError(
                f"golden pose event {event_key} must remain on the video frame clock, expected {expected_ms}, got {actual_ms}"
            )

    last_club_frame = max(timeline["clubHeadFrames"])
    impact_ms = float(fixture["labels"]["impactMs"])
    club_impact_gap_ms = impact_ms - float(times[last_club_frame] or 0)
    if club_impact_gap_ms < float(timeline["minimumClubImpactGapMs"]):
        raise AssertionError(
            f"golden club track must retain its post-impact gap, expected >= {timeline['minimumClubImpactGapMs']}, got {club_impact_gap_ms}"
        )
    raw_last_club_ms = round(last_club_frame * raw_last_ms / last_frame)
    if abs(raw_last_club_ms - impact_ms) <= float(fixture["toleranceMs"]):
        raise AssertionError("compressed inference clock must not make the sparse club track appear impact-aligned")

    club_head_track = [
        {"t": times[frame_index], "conf": 0.5}
        for frame_index in timeline["clubHeadFrames"]
    ]
    evidence = _validate_event_evidence(
        body_events=fixture["labels"],
        wrist_top=None,
        wrist_impact=None,
        club_head_track=club_head_track,
    )
    if evidence.get("status") != "withheld" or "CLUB_TRACK_INSUFFICIENT_AT_IMPACT" not in evidence.get("codes", []):
        raise AssertionError(f"golden fixture must retain the real club-head coverage limitation, got {evidence}")


def test_wrist_gate_rejects_static_background_club_candidate() -> None:
    wrists = [
        {"frame": 33, "x": 0.544, "y": 0.590},
        {"frame": 66, "x": 0.545, "y": 0.537},
    ]
    background = [{"frame": 33, "x": 0.783, "y": 0.219, "w": 0.04, "h": 0.032, "conf": 0.51}]
    plausible = [{"frame": 66, "x": 0.632, "y": 0.267, "w": 0.04, "h": 0.03, "conf": 0.53}]
    shaft_sized_handle = [{"frame": 66, "x": 0.52, "y": 0.54, "w": 0.285, "h": 0.21, "conf": 0.62}]
    if _filter_track_by_wrist(
        background, wrists, max_distance=0.32, max_area=0.02, max_width=0.18, max_height=0.18
    ):
        raise AssertionError("static background head must not become a club track")
    if not _filter_track_by_wrist(
        plausible, wrists, max_distance=0.32, max_area=0.02, max_width=0.18, max_height=0.18
    ):
        raise AssertionError("plausible moving head candidate should remain available")
    if _filter_track_by_wrist(
        shaft_sized_handle, wrists, max_distance=0.22, max_area=0.06, max_width=0.22, max_height=0.18
    ):
        raise AssertionError("shaft-sized handle box must not become a grip track")
    if _filter_track_by_bbox(
        [{"w": 0.43, "h": 0.46}], max_area=0.012, max_width=0.14, max_height=0.14
    ):
        raise AssertionError("person-sized ball box must not become a ball track")


def test_point_only_impact_stability_does_not_fail_fusion() -> None:
    point_track = [
        {"t": 0, "x": 0.50, "y": 0.50, "conf": 0.9},
        {"t": 33, "x": 0.51, "y": 0.49, "conf": 0.9},
        {"t": 66, "x": 0.52, "y": 0.50, "conf": 0.9},
    ]
    label, score = _impact_stability(point_track, 1)
    if label != "unstable" or score != 0.0:
        raise AssertionError(f"point-only impact stability must remain conservative, got {(label, score)}")


if __name__ == "__main__":
    test_state_machine_rejects_unrefined_early_top()
    test_state_machine_accepts_refined_compact_top()
    test_finish_adjacent_impact_uses_wrist_candidate_without_club_head()
    test_forward_decoder_enforces_phase_order_and_duration()
    test_forward_decoder_rejects_finish_before_swing()
    test_pose_club_and_roi_motion_are_exposed_as_evidence()
    test_confident_single_wrist_preserves_early_evidence()
    test_selector_uses_forward_decoder_for_complete_pose_club_roi_sequence()
    test_pipeline_only_uses_confident_forward_decoder()
    test_event_validation_withholds_conflicting_or_missing_club_evidence()
    test_production_reference_impact_fixture_is_partial_not_abandoned()
    test_pose_events_remain_partial_when_impact_is_withheld()
    test_quality_withheld_result_is_still_complete()
    test_meta_timeline_prefers_video_frame_clock_over_inference_clock()
    test_wrist_gate_rejects_static_background_club_candidate()
    test_point_only_impact_stability_does_not_fail_fusion()
    print("body event selector checks passed")
