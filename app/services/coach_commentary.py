"""Rule-based coaching commentary for fused golf swing metrics.

The comments here intentionally avoid absolute diagnosis when the upstream
signals are weak. Each comment follows: observation -> likely implication ->
actionable checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CoachFinding:
    key: str
    category: str
    severity: str
    confidence: float
    evidence: str
    interpretation: str
    action: str
    caution: Optional[str] = None

    def comment(self) -> str:
        suffix = f" {self.caution}" if self.caution else ""
        return f"{self.evidence} {self.interpretation} {self.action}{suffix}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "category": self.category,
            "severity": self.severity,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
            "interpretation": self.interpretation,
            "action": self.action,
            "caution": self.caution,
        }


SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "info": 3}
CATEGORY_ORDER = {
    "tempo": 0,
    "backswing": 1,
    "shaft_plane": 2,
    "swing_path": 3,
    "impact": 4,
    "quality": 5,
}


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


def _caution(confidence: float, source: str = "") -> Optional[str]:
    if confidence >= 0.35:
        return None
    suffix = "현재 검출 신뢰도가 낮아 참고 신호로만 보세요."
    if source == "club_box_proxy":
        suffix = "club_handle 없이 bbox로 근사한 값이라 참고 신호로만 보세요."
    return suffix


def _tempo_finding(tempo: Dict[str, object]) -> Optional[CoachFinding]:
    ratio = _safe_float(tempo.get("ratio"), 0.0)
    backswing_ms = _safe_int(tempo.get("backswingMs"), 0)
    downswing_ms = _safe_int(tempo.get("downswingMs"), 0)
    if ratio <= 0 or backswing_ms <= 0 or downswing_ms <= 0:
        return None

    if ratio < 1.7:
        return CoachFinding(
            key="tempo_rushed_transition",
            category="tempo",
            severity="high",
            confidence=0.75,
            evidence=f"템포가 {ratio}:1로 전환이 급합니다.",
            interpretation="탑에서 바로 손으로 내려치면 하체-몸통-팔-클럽 순서가 무너지기 쉽습니다.",
            action="탑에서 반 박자 멈춘 뒤 하체와 가슴 회전으로 다운스윙을 시작해 보세요.",
        )
    if ratio < 2.4:
        return CoachFinding(
            key="tempo_fast",
            category="tempo",
            severity="medium",
            confidence=0.72,
            evidence=f"템포가 {ratio}:1로 빠른 편입니다.",
            interpretation="리듬 자체는 쓸 수 있지만 전환에서 손목 릴리스가 먼저 풀릴 가능성이 있습니다.",
            action="다운스윙 시작 때 손보다 가슴이 공 쪽으로 돌아오는 느낌을 우선하세요.",
        )
    if ratio <= 3.6:
        return CoachFinding(
            key="tempo_usable",
            category="tempo",
            severity="info",
            confidence=0.68,
            evidence=f"템포는 {ratio}:1로 사용할 수 있는 범위입니다.",
            interpretation="지금은 템포 자체보다 탑 이후 임팩트까지의 재현성이 더 중요합니다.",
            action="같은 탑 위치와 같은 다운스윙 리듬이 반복되는지 먼저 확인하세요.",
        )
    if ratio <= 4.5:
        return CoachFinding(
            key="tempo_slow_transition",
            category="tempo",
            severity="medium",
            confidence=0.7,
            evidence=f"템포가 {ratio}:1로 백스윙 대비 다운스윙이 짧습니다.",
            interpretation="탑에서 리듬이 멈추면 다운스윙을 급하게 보상하기 쉽습니다.",
            action="백스윙을 더 키우기보다 전환 이후 체중 이동과 회전이 끊기지 않게 연결하세요.",
        )
    return CoachFinding(
        key="tempo_excessively_slow",
        category="tempo",
        severity="high",
        confidence=0.7,
        evidence=f"템포가 {ratio}:1로 백스윙 시간이 과하게 깁니다.",
        interpretation="탑에서 정지한 뒤 다시 치는 패턴이면 스윙 전체 리듬이 끊길 수 있습니다.",
        action="3/4 스윙으로 백스윙과 다운스윙이 이어지는 연속 리듬을 먼저 맞추세요.",
    )


def _backswing_finding(backswing: Dict[str, object]) -> Optional[CoachFinding]:
    label = str(backswing.get("label") or "")
    source = str(backswing.get("source") or "")
    travel = _safe_float(backswing.get("handTravelRatio") or backswing.get("clubTravelRatio"), 0.0)
    confidence = _safe_float(backswing.get("score"), 0.45)

    if label == "short":
        return CoachFinding(
            key="backswing_short",
            category="backswing",
            severity="medium",
            confidence=confidence,
            evidence=f"백스윙 크기가 작게 잡힙니다(travel {travel:.2f}).",
            interpretation="손만 작게 드는 패턴이면 파워와 다운스윙 여유가 줄어듭니다.",
            action="왼쪽 어깨가 턱 밑으로 들어오는 3/4 회전부터 확인하세요.",
        )
    if label == "low_top":
        return CoachFinding(
            key="backswing_low_top",
            category="backswing",
            severity="medium",
            confidence=confidence,
            evidence="탑 위치가 낮게 잡힙니다.",
            interpretation="팔 높이만 낮은 문제가 아니라 어깨 회전과 손목 코킹이 같이 부족할 수 있습니다.",
            action="팔을 억지로 높이기보다 어깨 회전 폭과 손목 코킹이 함께 만들어지는지 확인하세요.",
        )
    if label == "high_top":
        return CoachFinding(
            key="backswing_high_top",
            category="backswing",
            severity="medium",
            confidence=confidence,
            evidence="탑 위치가 높게 잡힙니다.",
            interpretation="오버스윙이면 다운스윙 타이밍이 늦어질 수 있습니다.",
            action="왼팔이 지면과 평행을 지난 직후 멈추는 축소 스윙으로 기준점을 잡으세요.",
        )
    if label == "adequate":
        source_text = "손목 추적" if source == "pose_wrist" else "클럽 추적"
        if source == "pose_wrist":
            action = "다음 우선순위는 크기보다 전환 순서와 임팩트 재현성입니다."
        else:
            action = "더 크게 만들기보다 같은 탑 위치를 반복하는 쪽이 좋습니다."
        return CoachFinding(
            key="backswing_adequate",
            category="backswing",
            severity="info",
            confidence=confidence,
            evidence=f"백스윙 크기는 {source_text} 기준으로 충분합니다.",
            interpretation="현재 관측에서는 백스윙 크기를 주요 문제로 보지 않습니다.",
            action=action,
        )
    return None


def _shaft_finding(shaft_plane: Dict[str, object]) -> CoachFinding:
    label = str(shaft_plane.get("label") or "")
    confidence = _safe_float(shaft_plane.get("confidence"), 0.0)
    source = str(shaft_plane.get("source") or "")
    angle = shaft_plane.get("angleDeg")
    angle_text = f"{angle}도" if angle is not None else "각도 미확정"

    if label == "flat":
        return CoachFinding(
            key="shaft_flat",
            category="shaft_plane",
            severity="medium",
            confidence=confidence,
            evidence=f"다운스윙 샤프트가 낮고 뒤에 남는 편입니다({angle_text}).",
            interpretation="클럽이 몸 뒤에 갇히면 임팩트 직전 손으로 급히 맞추게 됩니다.",
            action="전환 때 손을 내리기보다 가슴 회전으로 클럽을 앞으로 끌고 오세요.",
            caution=_caution(confidence, source),
        )
    if label == "steep":
        return CoachFinding(
            key="shaft_steep",
            category="shaft_plane",
            severity="medium",
            confidence=confidence,
            evidence=f"다운스윙 샤프트가 세워지는 편입니다({angle_text}).",
            interpretation="손과 팔이 먼저 앞으로 나가면 깎아 치는 궤도가 되기 쉽습니다.",
            action="오른팔이 몸 앞에 붙은 상태로 내려오는지 확인하세요.",
            caution=_caution(confidence, source),
        )
    if label == "neutral":
        return CoachFinding(
            key="shaft_neutral",
            category="shaft_plane",
            severity="info",
            confidence=confidence,
            evidence=f"다운스윙 샤프트 각도는 2D 기준 중립 범위입니다({angle_text}).",
            interpretation="현재 관측에서는 샤프트 플레인을 주요 문제로 보지 않습니다.",
            action="임팩트 위치와 페이스/경로 안정성을 우선 확인하세요.",
            caution=_caution(confidence, source),
        )
    return CoachFinding(
        key="shaft_unknown",
        category="shaft_plane",
        severity="info",
        confidence=0.0,
        evidence="샤프트 플레인은 club_head와 handle 동시 추적이 부족해 판단하지 않습니다.",
        interpretation="bbox나 단일 점만으로는 샤프트 방향을 확정하기 어렵습니다.",
        action="이 항목은 촬영/검출 품질 개선 후 다시 보세요.",
    )


def _swing_plane_finding(swing_plane: Dict[str, object]) -> Optional[CoachFinding]:
    label = str(swing_plane.get("label") or "")
    confidence = _safe_float(swing_plane.get("confidence"), 0.0)
    source = str(swing_plane.get("source") or "")
    if confidence < 0.28:
        return None
    if label == "outside-in":
        return CoachFinding(
            key="path_outside_in",
            category="swing_path",
            severity="medium",
            confidence=confidence,
            evidence="클럽 경로가 outside-in 쪽으로 보입니다.",
            interpretation="슬라이스/풀성 구질이 동반된다면 손과 클럽이 바깥에서 들어오는 패턴일 수 있습니다.",
            action="다운스윙 초반 손이 공 쪽으로 튀어나오는지 확인하세요.",
            caution=_caution(confidence, source),
        )
    if label == "inside-out":
        return CoachFinding(
            key="path_inside_out",
            category="swing_path",
            severity="medium",
            confidence=confidence,
            evidence="클럽 경로가 inside-out 쪽으로 보입니다.",
            interpretation="푸시/훅성 구질이 동반된다면 클럽이 너무 뒤에서 늦게 나오는 패턴일 수 있습니다.",
            action="전환 이후 클럽이 몸 앞 공간으로 돌아오는지 확인하세요.",
            caution=_caution(confidence, source),
        )
    return None


def _impact_finding(impact_stability: Dict[str, object]) -> Optional[CoachFinding]:
    label = str(impact_stability.get("label") or "")
    score = _safe_float(impact_stability.get("score"), 0.0)
    if label == "stable":
        return CoachFinding(
            key="impact_stable",
            category="impact",
            severity="info",
            confidence=score,
            evidence=f"임팩트 주변 클럽 위치는 비교적 안정적입니다(score {score:.2f}).",
            interpretation="현재 관측에서는 임팩트 재현성을 주요 문제로 보지 않습니다.",
            action="이 패턴을 유지하면서 시작 방향과 거리 편차를 확인하세요.",
        )
    if label == "unstable":
        return CoachFinding(
            key="impact_unstable",
            category="impact",
            severity="high" if score < 0.25 else "medium",
            confidence=max(0.25, 1.0 - score),
            evidence=f"임팩트 주변 클럽 위치 변동이 큽니다(score {score:.2f}).",
            interpretation="릴리스 타이밍이나 몸 회전이 임팩트 구간에서 끊길 가능성이 있습니다.",
            action="공을 맞히는 순간보다 임팩트 전후 3프레임의 손목 릴리스와 몸 회전을 먼저 보세요.",
        )
    return None


def _quality_findings(tracking: Dict[str, object], ball: Dict[str, object], readiness: Dict[str, object]) -> List[CoachFinding]:
    findings: List[CoachFinding] = []
    if readiness.get("label") == "not_ready":
        findings.append(
            CoachFinding(
                key="readiness_not_ready",
                category="quality",
                severity="medium",
                confidence=_safe_float(readiness.get("confidence"), 0.3),
                evidence="어드레스 준비 상태가 불안정하게 감지됩니다.",
                interpretation="스윙 시작 전 움직임이 섞이면 이벤트 기준점이 흔들릴 수 있습니다.",
                action="분석 시작 구간을 어드레스 이후로 맞추면 이벤트 정확도가 올라갑니다.",
            )
        )
    if tracking.get("label") == "weak":
        findings.append(
            CoachFinding(
                key="tracking_weak",
                category="quality",
                severity="medium",
                confidence=_safe_float(tracking.get("score"), 0.0),
                evidence="추적 품질이 낮습니다.",
                interpretation="코멘트는 우선순위 참고용이며 세부 진단은 흔들릴 수 있습니다.",
                action="전신이 프레임 안에 들어오고 클럽 헤드가 배경과 분리되게 촬영하면 정확도가 올라갑니다.",
            )
        )
    if _safe_int(tracking.get("personFrames"), 0) == 0:
        findings.append(
            CoachFinding(
                key="person_detection_missing",
                category="quality",
                severity="info",
                confidence=0.2,
                evidence="person detection은 없지만 pose keypoint를 사용 중입니다.",
                interpretation="전신 bbox 기반 보정 없이 keypoint 중심으로 몸 움직임을 봅니다.",
                action="골반/어깨 회전 코칭은 pose 안정화 후 더 강하게 판단할 수 있습니다.",
            )
        )
    if _safe_int(tracking.get("ballFrames"), 0) == 0 and str(ball.get("launchDirection") or "unknown") == "unknown":
        findings.append(
            CoachFinding(
                key="ball_missing",
                category="quality",
                severity="info",
                confidence=0.0,
                evidence="공 검출이 없어 실제 구질과 출발 방향은 판단하지 않습니다.",
                interpretation="현재 시스템은 ball flight law까지 확정하지 못합니다.",
                action="현재 코멘트는 몸-클럽 움직임 기준으로 해석하세요.",
            )
        )
    return findings


def _rank_findings(findings: List[CoachFinding]) -> List[CoachFinding]:
    return sorted(
        findings,
        key=lambda finding: (
            SEVERITY_ORDER.get(finding.severity, 9),
            CATEGORY_ORDER.get(finding.category, 9),
            -finding.confidence,
        ),
    )


def build_coach_findings(
    tempo: Dict[str, object],
    shaft_plane: Dict[str, object],
    backswing: Dict[str, object],
    impact_stability: Dict[str, object],
    readiness: Dict[str, object],
    tracking: Dict[str, object],
    ball: Dict[str, object],
    swing_plane: Dict[str, object],
) -> List[CoachFinding]:
    findings: List[CoachFinding] = []
    for candidate in (
        _tempo_finding(tempo),
        _backswing_finding(backswing),
        _shaft_finding(shaft_plane),
        _swing_plane_finding(swing_plane),
        _impact_finding(impact_stability),
    ):
        if candidate:
            findings.append(candidate)
    findings.extend(_quality_findings(tracking, ball, readiness))
    return _rank_findings(findings)


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
    comments = [finding.comment() for finding in build_coach_findings(
        tempo,
        shaft_plane,
        backswing,
        impact_stability,
        readiness,
        tracking,
        ball,
        swing_plane,
    )]

    deduped: List[str] = []
    seen = set()
    for comment in comments:
        key = comment.strip()
        if not key or key in seen:
            continue
        deduped.append(key)
        seen.add(key)
    return deduped[:limit]


def build_coach_finding_debug(
    tempo: Dict[str, object],
    shaft_plane: Dict[str, object],
    backswing: Dict[str, object],
    impact_stability: Dict[str, object],
    readiness: Dict[str, object],
    tracking: Dict[str, object],
    ball: Dict[str, object],
    swing_plane: Dict[str, object],
    *,
    limit: int = 8,
) -> List[Dict[str, object]]:
    return [
        finding.to_dict()
        for finding in build_coach_findings(
            tempo,
            shaft_plane,
            backswing,
            impact_stability,
            readiness,
            tracking,
            ball,
            swing_plane,
        )[:limit]
    ]
