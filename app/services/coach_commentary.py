"""Rule-based coaching commentary for fused golf swing metrics.

The comments here intentionally avoid absolute diagnosis when the upstream
signals are weak. Each comment follows: observation -> likely implication ->
actionable checkpoint.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        num = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return num if math.isfinite(num) else default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _with_caution(comment: str, confidence: float, source: str = "") -> str:
    if confidence >= 0.35:
        return comment
    suffix = " 현재 검출 신뢰도가 낮아 참고 신호로만 보세요."
    if source == "club_box_proxy":
        suffix = " club_handle 없이 bbox로 근사한 값이라 참고 신호로만 보세요."
    return f"{comment}{suffix}"


def _tempo_comment(tempo: Dict[str, object]) -> str | None:
    ratio = _safe_float(tempo.get("ratio"), 0.0)
    backswing_ms = _safe_int(tempo.get("backswingMs"), 0)
    downswing_ms = _safe_int(tempo.get("downswingMs"), 0)
    if ratio <= 0 or backswing_ms <= 0 or downswing_ms <= 0:
        return None

    if ratio < 1.7:
        return (
            f"템포가 {ratio}:1로 전환이 급합니다. 탑에서 바로 손으로 내려치기보다 "
            "하체-몸통-팔 순서가 느껴지도록 탑에서 반 박자 멈춘 뒤 다운스윙을 시작해 보세요."
        )
    if ratio < 2.4:
        return (
            f"템포가 {ratio}:1로 빠른 편입니다. 리듬 자체는 쓸 수 있지만, 다운스윙 시작 때 "
            "손목 릴리스가 먼저 풀리지 않도록 가슴이 공 쪽으로 돌아오는 느낌을 우선하세요."
        )
    if ratio <= 3.6:
        return (
            f"템포는 {ratio}:1로 사용할 수 있는 범위입니다. 지금은 템포를 크게 바꾸기보다 "
            "탑 이후 임팩트까지 같은 리듬으로 반복되는지 확인하는 쪽이 우선입니다."
        )
    if ratio <= 4.5:
        return (
            f"템포가 {ratio}:1로 백스윙 대비 다운스윙이 짧습니다. 백스윙을 더 키우기보다 "
            "전환 이후 체중 이동과 회전이 끊기지 않게 연결하는 연습이 필요합니다."
        )
    return (
        f"템포가 {ratio}:1로 백스윙 시간이 과하게 깁니다. 탑에서 정지한 뒤 다시 치는 패턴이면 "
        "스윙 전체 리듬이 끊길 수 있으니 3/4 스윙으로 연속 리듬을 먼저 맞추세요."
    )


def _backswing_comment(backswing: Dict[str, object]) -> str | None:
    label = str(backswing.get("label") or "")
    source = str(backswing.get("source") or "")
    travel = _safe_float(backswing.get("handTravelRatio") or backswing.get("clubTravelRatio"), 0.0)

    if label == "short":
        return (
            f"백스윙 크기가 작게 잡힙니다(travel {travel:.2f}). 손만 작게 드는 패턴이면 파워와 "
            "다운스윙 여유가 줄어드니, 왼쪽 어깨가 턱 밑으로 들어오는 3/4 회전부터 확인하세요."
        )
    if label == "low_top":
        return (
            "탑 위치가 낮게 잡힙니다. 팔을 억지로 높이기보다 어깨 회전 폭과 손목 코킹이 "
            "함께 만들어지는지 정면/측면 캡처로 확인하세요."
        )
    if label == "high_top":
        return (
            "탑 위치가 높게 잡힙니다. 오버스윙이면 다운스윙 타이밍이 늦어질 수 있으니 "
            "왼팔이 지면과 평행을 지난 직후 멈추는 축소 스윙으로 기준점을 잡으세요."
        )
    if label == "adequate":
        if source == "pose_wrist":
            return "백스윙 크기는 손목 추적 기준으로 충분합니다. 다음 우선순위는 크기보다 전환 순서와 임팩트 재현성입니다."
        return "백스윙 크기는 클럽 추적 기준으로 충분합니다. 더 크게 만들기보다 같은 탑 위치를 반복하는 쪽이 좋습니다."
    return None


def _shaft_comment(shaft_plane: Dict[str, object]) -> str | None:
    label = str(shaft_plane.get("label") or "")
    confidence = _safe_float(shaft_plane.get("confidence"), 0.0)
    source = str(shaft_plane.get("source") or "")
    angle = shaft_plane.get("angleDeg")
    angle_text = f"{angle}도" if angle is not None else "각도 미확정"

    if label == "flat":
        return _with_caution(
            f"다운스윙 샤프트가 낮고 뒤에 남는 편입니다({angle_text}). 클럽이 몸 뒤에 갇히면 "
            "임팩트 직전 손으로 급히 맞추게 되므로, 전환 때 손을 내리기보다 가슴 회전으로 클럽을 앞으로 끌고 오세요.",
            confidence,
            source,
        )
    if label == "steep":
        return _with_caution(
            f"다운스윙 샤프트가 세워지는 편입니다({angle_text}). 손과 팔이 먼저 앞으로 나가면 "
            "깎아 치는 궤도가 되기 쉬우니, 오른팔이 몸 앞에 붙은 상태로 내려오는지 확인하세요.",
            confidence,
            source,
        )
    if label == "neutral":
        return _with_caution(
            f"다운스윙 샤프트 각도는 2D 기준 중립 범위입니다({angle_text}). 현재는 샤프트 플레인보다 "
            "임팩트 위치와 페이스/경로 안정성을 우선 확인하세요.",
            confidence,
            source,
        )
    return "샤프트 플레인은 club_head와 handle 동시 추적이 부족해 판단하지 않습니다. 이 항목은 촬영/검출 품질 개선 후 다시 보세요."


def _swing_plane_comment(swing_plane: Dict[str, object]) -> str | None:
    label = str(swing_plane.get("label") or "")
    confidence = _safe_float(swing_plane.get("confidence"), 0.0)
    source = str(swing_plane.get("source") or "")
    if confidence < 0.28:
        return None
    if label == "outside-in":
        return _with_caution(
            "클럽 경로가 outside-in 쪽으로 보입니다. 슬라이스/풀성 구질이 동반된다면 다운스윙 초반 손이 공 쪽으로 튀어나오는지 확인하세요.",
            confidence,
            source,
        )
    if label == "inside-out":
        return _with_caution(
            "클럽 경로가 inside-out 쪽으로 보입니다. 푸시/훅성 구질이 동반된다면 클럽이 너무 뒤에서 늦게 나오는지 확인하세요.",
            confidence,
            source,
        )
    return None


def _impact_comment(impact_stability: Dict[str, object]) -> str | None:
    label = str(impact_stability.get("label") or "")
    score = _safe_float(impact_stability.get("score"), 0.0)
    if label == "stable":
        return f"임팩트 주변 클럽 위치는 비교적 안정적입니다(score {score:.2f}). 이 패턴을 유지하면서 시작 방향과 거리 편차를 확인하세요."
    if label == "unstable":
        return (
            f"임팩트 주변 클럽 위치 변동이 큽니다(score {score:.2f}). 공을 맞히는 순간보다 임팩트 전후 "
            "3프레임의 손목 릴리스와 몸 회전이 끊기는지 먼저 보세요."
        )
    return None


def _quality_comments(tracking: Dict[str, object], ball: Dict[str, object], readiness: Dict[str, object]) -> List[str]:
    comments: List[str] = []
    if readiness.get("label") == "not_ready":
        comments.append("어드레스 준비 상태가 불안정하게 감지됩니다. 분석 시작 구간을 어드레스 이후로 맞추면 이벤트 정확도가 올라갑니다.")
    if tracking.get("label") == "weak":
        comments.append("추적 품질이 낮습니다. 코멘트는 우선순위 참고용이며, 전신이 프레임 안에 들어오고 클럽 헤드가 배경과 분리되게 촬영하면 정확도가 올라갑니다.")
    if _safe_int(tracking.get("personFrames"), 0) == 0:
        comments.append("person detection은 없지만 pose keypoint를 사용 중입니다. 골반/어깨 회전 코칭은 pose 안정화 후 더 강하게 판단할 수 있습니다.")
    if _safe_int(tracking.get("ballFrames"), 0) == 0 and str(ball.get("launchDirection") or "unknown") == "unknown":
        comments.append("공 검출이 없어 실제 구질과 출발 방향은 판단하지 않습니다. 현재 코멘트는 몸-클럽 움직임 기준입니다.")
    return comments


def build_coach_comments(
    tempo: Dict[str, object],
    shaft_plane: Dict[str, object],
    backswing: Dict[str, object],
    impact_stability: Dict[str, object],
    readiness: Dict[str, object],
    tracking: Dict[str, object],
    ball: Dict[str, object],
    swing_plane: Dict[str, object],
    *,
    limit: int = 6,
) -> List[str]:
    comments: List[str] = []

    for candidate in (
        _tempo_comment(tempo),
        _backswing_comment(backswing),
        _shaft_comment(shaft_plane),
        _swing_plane_comment(swing_plane),
        _impact_comment(impact_stability),
    ):
        if candidate:
            comments.append(candidate)

    comments.extend(_quality_comments(tracking, ball, readiness))

    deduped: List[str] = []
    seen = set()
    for comment in comments:
        key = comment.strip()
        if not key or key in seen:
            continue
        deduped.append(key)
        seen.add(key)
    return deduped[:limit]
