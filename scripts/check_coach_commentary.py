#!/usr/bin/env python3
"""Regression checks for rule-based golf coach commentary."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.coach_commentary import build_coach_comments, build_coach_finding_debug


def assert_contains(comments: list[str], text: str) -> None:
    if not any(text in comment for comment in comments):
        raise AssertionError(f"missing text {text!r} in comments: {comments}")


def assert_not_contains(comments: list[str], text: str) -> None:
    if any(text in comment for comment in comments):
        raise AssertionError(f"unexpected text {text!r} in comments: {comments}")


def keys(debug: list[dict[str, object]]) -> list[object]:
    return [item["key"] for item in debug]


def run_fast_low_confidence_case() -> None:
    comments = build_coach_comments(
        {"backswingMs": 352, "downswingMs": 175, "ratio": 2.01},
        {"label": "flat", "confidence": 0.27, "angleDeg": 15.3, "source": "club_box_proxy"},
        {"label": "adequate", "score": 0.61, "clubTravelRatio": 0.13, "handTravelRatio": 0.13, "source": "pose_wrist"},
        {"label": "unstable", "score": 0},
        {"label": "unknown"},
        {"label": "weak", "score": 0.11, "personFrames": 0, "ballFrames": 0},
        {"launchDirection": "unknown"},
        {"label": "outside-in", "confidence": 0.2, "source": "club_box_endpoint"},
    )
    debug = build_coach_finding_debug(
        {"backswingMs": 352, "downswingMs": 175, "ratio": 2.01},
        {"label": "flat", "confidence": 0.27, "angleDeg": 15.3, "source": "club_box_proxy"},
        {"label": "adequate", "score": 0.61, "clubTravelRatio": 0.13, "handTravelRatio": 0.13, "source": "pose_wrist"},
        {"label": "unstable", "score": 0},
        {"label": "unknown"},
        {"label": "weak", "score": 0.11, "personFrames": 0, "ballFrames": 0},
        {"launchDirection": "unknown"},
        {"label": "outside-in", "confidence": 0.2, "source": "club_box_endpoint"},
    )

    assert len(comments) == 6
    assert_contains(comments, "빠른 전환, 낮은 샤프트, 임팩트 불안정")
    assert_contains(comments, "클럽이 몸 뒤에 남은 상태")
    assert_contains(comments, "임팩트 주변 클럽 위치 변동")
    assert_contains(comments, "club_handle 없이 bbox로 근사")
    assert_contains(comments, "추적 품질이 낮습니다")
    assert_not_contains(comments, "템포가 2.01:1로 빠른 편")
    assert_not_contains(comments, "다운스윙 샤프트가 낮고 뒤에 남는 편")
    assert_not_contains(comments, "outside-in")
    assert debug[0]["key"] == "pattern_late_club_release"
    assert debug[0]["priority"] == "1순위 패턴"
    assert "펌프 드릴" in str(debug[0]["drill"])
    assert "몸 앞" in str(debug[0]["checkpoint"])
    assert debug[1]["key"] == "impact_unstable"
    assert "tempo_fast" in keys(debug)
    assert "shaft_flat" in keys(debug)
    assert "ball_missing" in keys(debug)


def run_stable_neutral_case() -> None:
    comments = build_coach_comments(
        {"backswingMs": 430, "downswingMs": 140, "ratio": 3.07},
        {"label": "neutral", "confidence": 0.55, "angleDeg": 48.0, "source": "head_handle"},
        {"label": "adequate", "score": 0.97, "clubTravelRatio": 0.4, "source": "club_motion"},
        {"label": "stable", "score": 0.82},
        {"label": "ready"},
        {"label": "fair", "score": 0.42, "personFrames": 20, "ballFrames": 4},
        {"launchDirection": "center"},
        {"label": "inside-out", "confidence": 0.42, "source": "hybrid"},
    )

    assert len(comments) == 5
    assert_contains(comments, "템포는 3.07:1로 사용할 수 있는 범위")
    assert_contains(comments, "샤프트 각도는 2D 기준 중립 범위")
    assert_contains(comments, "클럽 경로가 inside-out")
    assert_contains(comments, "임팩트 주변 클럽 위치는 비교적 안정적")
    assert_not_contains(comments, "추적 품질이 낮습니다")


def run_short_steep_case() -> None:
    comments = build_coach_comments(
        {"backswingMs": 120, "downswingMs": 120, "ratio": 1.0},
        {"label": "steep", "confidence": 0.62, "angleDeg": 66.0, "source": "head_handle"},
        {"label": "short", "score": 0.2, "clubTravelRatio": 0.04, "source": "club_motion"},
        {"label": "unstable", "score": 0.35},
        {"label": "ready"},
        {"label": "fair", "score": 0.4, "personFrames": 12, "ballFrames": 2},
        {"launchDirection": "right"},
        {"label": "outside-in", "confidence": 0.5, "source": "hybrid"},
    )
    debug = build_coach_finding_debug(
        {"backswingMs": 120, "downswingMs": 120, "ratio": 1.0},
        {"label": "steep", "confidence": 0.62, "angleDeg": 66.0, "source": "head_handle"},
        {"label": "short", "score": 0.2, "clubTravelRatio": 0.04, "source": "club_motion"},
        {"label": "unstable", "score": 0.35},
        {"label": "ready"},
        {"label": "fair", "score": 0.4, "personFrames": 12, "ballFrames": 2},
        {"launchDirection": "right"},
        {"label": "outside-in", "confidence": 0.5, "source": "hybrid"},
    )

    assert_contains(comments, "전환이 급합니다")
    assert_contains(comments, "세워진 샤프트와 outside-in 경로")
    assert_contains(comments, "백스윙이 작고 전환 템포도 빠릅니다")
    assert_not_contains(comments, "백스윙 크기가 작게 잡힙니다")
    assert_not_contains(comments, "다운스윙 샤프트가 세워지는 편")
    assert_not_contains(comments, "클럽 경로가 outside-in")
    assert debug[0]["key"] == "pattern_over_the_top"
    assert debug[0]["priority"] == "1순위 패턴"
    assert "오른팔 수건" in str(debug[0]["drill"])
    assert "다운스윙 첫 1/3" in str(debug[0]["checkpoint"])
    assert "pattern_rushed_short_swing" in keys(debug)
    assert "shaft_steep" in keys(debug)
    assert "path_outside_in" in keys(debug)


if __name__ == "__main__":
    run_fast_low_confidence_case()
    run_stable_neutral_case()
    run_short_steep_case()
    print("coach commentary checks passed")
