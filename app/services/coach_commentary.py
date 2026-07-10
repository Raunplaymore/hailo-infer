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
    priority: str = "확인"
    drill: Optional[str] = None
    checkpoint: Optional[str] = None
    caution: Optional[str] = None

    def comment(self) -> str:
        parts = [f"[{self.priority}] {self.evidence}", self.interpretation, self.action]
        if self.drill:
            parts.append(f"드릴: {self.drill}")
        if self.checkpoint:
            parts.append(f"체크: {self.checkpoint}")
        if self.caution:
            parts.append(self.caution)
        return " ".join(parts)

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "category": self.category,
            "severity": self.severity,
            "confidence": round(self.confidence, 2),
            "priority": self.priority,
            "evidence": self.evidence,
            "interpretation": self.interpretation,
            "action": self.action,
            "drill": self.drill,
            "checkpoint": self.checkpoint,
            "caution": self.caution,
        }


SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "info": 3}
CATEGORY_ORDER = {
    "pattern": 0,
    "fusion": 1,
    "tempo": 2,
    "backswing": 3,
    "body": 4,
    "shaft_plane": 5,
    "swing_path": 6,
    "impact": 7,
    "quality": 8,
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
            priority="전환 순서",
            drill="탑에서 1초 정지 후 50% 속도로 하체-가슴-팔 순서로 내려오기 5회.",
            checkpoint="다운스윙 첫 움직임이 손이 아니라 골반/가슴 회전으로 시작되는지 봅니다.",
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
            priority="리듬 조절",
            drill="3/4 스윙으로 백스윙은 그대로, 다운스윙 시작만 70% 속도로 낮춰 반복.",
            checkpoint="임팩트 직전 손목이 급히 뒤집히지 않고 몸 앞에서 릴리스되는지 확인합니다.",
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
            priority="유지",
            checkpoint="반복 촬영에서 top-impact 간격이 크게 흔들리지 않는지 봅니다.",
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
            priority="연결 리듬",
            drill="연속 빈스윙 3회 뒤 바로 공 치기. 탑에서 정지하지 않는 느낌을 유지합니다.",
            checkpoint="탑 이후 몸 회전이 멈춘 뒤 팔만 내려오는 구간이 있는지 확인합니다.",
        )
    return CoachFinding(
        key="tempo_excessively_slow",
        category="tempo",
        severity="high",
        confidence=0.7,
        evidence=f"템포가 {ratio}:1로 백스윙 시간이 과하게 깁니다.",
        interpretation="탑에서 정지한 뒤 다시 치는 패턴이면 스윙 전체 리듬이 끊길 수 있습니다.",
        action="3/4 스윙으로 백스윙과 다운스윙이 이어지는 연속 리듬을 먼저 맞추세요.",
        priority="연결 리듬",
        drill="메트로놈 느낌으로 하나-둘에 백스윙, 셋에 임팩트까지 연결.",
        checkpoint="탑에서 클럽이 멈춘 뒤 다시 출발하는지 봅니다.",
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
            priority="백스윙 크기",
            drill="공 없이 3/4 탑 위치를 만들고 2초 유지한 뒤 내려오기.",
            checkpoint="손만 올라가지 않고 어깨 회전과 손목 코킹이 함께 만들어지는지 봅니다.",
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
            priority="탑 위치",
            drill="오른팔 접힘과 손목 코킹을 만든 3/4 백스윙 정지 드릴.",
            checkpoint="탑에서 손 위치가 오른쪽 어깨보다 지나치게 낮지 않은지 확인합니다.",
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
            priority="탑 위치",
            drill="L-to-L 하프스윙으로 탑 위치를 줄인 뒤 같은 피니시까지 회전.",
            checkpoint="탑에서 클럽이 목표 방향으로 과하게 넘어가지 않는지 봅니다.",
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
            priority="유지",
            checkpoint="크기를 더 키우기보다 같은 탑 위치가 반복되는지 확인합니다.",
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
            priority="클럽 위치",
            drill="펌프 드릴: 탑에서 허리 높이까지 2번 내렸다가 세 번째에 치기.",
            checkpoint="다운스윙 중간에 그립과 클럽헤드가 몸 뒤가 아니라 오른쪽 허벅지 앞 공간에 있는지 봅니다.",
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
            priority="클럽 위치",
            drill="오른팔 겨드랑이에 수건을 끼고 하프스윙으로 임팩트까지 회전.",
            checkpoint="다운스윙 초반 손이 공 쪽으로 튀어나오지 않는지 봅니다.",
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
            priority="유지",
            checkpoint="샤프트보다 임팩트 전후 클럽헤드 위치 변동을 우선 봅니다.",
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
        priority="품질",
        checkpoint="club_head와 club_handle이 동시에 잡히는 프레임 수를 먼저 늘립니다.",
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
            priority="스윙 경로",
            drill="볼 뒤 안쪽에 헤드커버를 두고 하프스윙으로 안쪽 공간에서 내려오기.",
            checkpoint="공 기준 바깥쪽에서 클럽이 내려오는 궤적이 줄어드는지 봅니다.",
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
            priority="스윙 경로",
            drill="허리 높이 펌프 동작에서 클럽헤드를 몸 앞에 둔 뒤 회전으로 임팩트.",
            checkpoint="클럽이 과하게 뒤에서 접근해 손이 늦게 따라오는지 봅니다.",
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
            priority="유지",
            checkpoint="연속 샷에서 임팩트 프레임의 손/클럽 위치가 크게 바뀌지 않는지 봅니다.",
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
            priority="임팩트 재현성",
            drill="티 위에 공을 두고 50% 속도 펀치샷으로 임팩트 후 손이 목표 쪽으로 낮게 지나가게 반복.",
            checkpoint="임팩트 전후 클럽헤드가 갑자기 튀거나 손목이 급히 뒤집히는지 확인합니다.",
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
                priority="촬영 품질",
                checkpoint="address로 표시된 프레임이 실제 셋업 이후인지 확인합니다.",
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
                priority="촬영 품질",
                checkpoint="클럽 헤드가 천장/그물/매트와 겹치지 않고 프레임 안에 남는지 확인합니다.",
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
                priority="품질",
                checkpoint="현재는 골반/어깨 회전 진단보다 클럽-손목 기반 코멘트를 우선합니다.",
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
                priority="범위 제한",
                checkpoint="슬라이스/훅 확정 코멘트는 공 출발 방향과 커브 정보가 있을 때만 강화합니다.",
            )
        )
    return findings


def _body_findings(body_metrics: Dict[str, object]) -> List[CoachFinding]:
    findings: List[CoachFinding] = []
    head = body_metrics.get("headStability")
    if isinstance(head, dict):
        label = str(head.get("label") or "")
        movement = _safe_float(head.get("movementRatio"), 0.0)
        confidence = _safe_float(head.get("confidence"), 0.0)
        if label == "unstable" and confidence >= 0.35:
            findings.append(
                CoachFinding(
                    key="head_unstable",
                    category="body",
                    severity="medium",
                    confidence=confidence,
                    evidence=f"머리 기준점 이동이 크게 잡힙니다(movement {movement:.2f}).",
                    interpretation="스윙 중 머리와 상체 축이 많이 흔들리면 최저점과 임팩트 위치가 함께 흔들릴 수 있습니다.",
                    action="백스윙에서 머리를 고정하려고 버티기보다, 가슴 회전은 허용하되 상체 중심이 공 쪽/뒤쪽으로 밀리지 않는지 보세요.",
                    priority="축 안정성",
                    drill="벽 앞 섀도우 스윙: 이마와 벽 사이 간격을 유지하면서 50% 속도로 백스윙-임팩트 반복.",
                    checkpoint="address부터 impact까지 코/머리 위치가 어깨폭의 절반 이상 이동하지 않는지 확인합니다.",
                    caution="2D 코 keypoint 기반 참고 신호입니다.",
                )
            )
        elif label == "stable" and confidence >= 0.35:
            findings.append(
                CoachFinding(
                    key="head_stable",
                    category="body",
                    severity="info",
                    confidence=confidence,
                    evidence=f"머리 기준점은 비교적 안정적입니다(movement {movement:.2f}).",
                    interpretation="현재 관측에서는 상체 축 흔들림을 주요 문제로 보지 않습니다.",
                    action="축을 더 고정하려 하기보다 회전 순서와 임팩트 재현성을 우선하세요.",
                    priority="유지",
                    checkpoint="머리 고정보다 가슴과 골반 회전이 막히지 않는지 같이 봅니다.",
                )
            )

    shoulder = body_metrics.get("shoulderTurnProxy")
    if isinstance(shoulder, dict):
        label = str(shoulder.get("label") or "")
        delta = shoulder.get("deltaDeg")
        confidence = _safe_float(shoulder.get("confidence"), 0.0)
        if label == "limited" and confidence >= 0.3:
            findings.append(
                CoachFinding(
                    key="shoulder_turn_limited",
                    category="body",
                    severity="medium",
                    confidence=confidence,
                    evidence=f"어깨 회전 proxy 변화가 작게 잡힙니다({delta}도).",
                    interpretation="실제 회전이 부족하면 백스윙 크기와 다운스윙 여유가 같이 줄어들 수 있습니다.",
                    action="팔을 더 드는 보상보다 왼쪽 어깨가 턱 밑으로 들어오는 몸통 회전을 먼저 확인하세요.",
                    priority="몸통 회전",
                    drill="클럽을 가슴에 대고 어깨선이 목표 반대쪽으로 돌아가는 3/4 회전 드릴.",
                    checkpoint="탑에서 손 높이보다 어깨선 변화가 먼저 늘어나는지 봅니다.",
                    caution="2D shoulder keypoint proxy라 실제 회전각 확정값은 아닙니다.",
                )
            )
    return findings


def _fusion_findings(fusion_metrics: Dict[str, object]) -> List[CoachFinding]:
    findings: List[CoachFinding] = []
    release = fusion_metrics.get("releaseTiming")
    if isinstance(release, dict):
        label = str(release.get("label") or "")
        confidence = _safe_float(release.get("confidence"), 0.0)
        if label == "late_proxy" and confidence >= 0.25:
            findings.append(
                CoachFinding(
                    key="release_late_proxy",
                    category="fusion",
                    severity="medium",
                    confidence=confidence,
                    evidence="릴리스 타이밍 proxy가 늦은 쪽으로 잡힙니다.",
                    interpretation="클럽이 몸 뒤에 남은 상태에서 임팩트 직전 손으로 맞추는 보상이 생길 수 있습니다.",
                    action="전환 이후 클럽을 몸 앞 공간에 두고 회전으로 임팩트를 통과하는 느낌을 먼저 만드세요.",
                    priority="릴리스 타이밍",
                    drill="허리 높이 펌프 드릴에서 그립과 클럽헤드가 오른쪽 허벅지 앞 공간을 지나게 만들기.",
                    checkpoint="임팩트 직전 클럽헤드가 손 뒤에서 급히 따라오는지 확인합니다.",
                    caution="공/페이스 데이터가 없는 tempo-shaft-impact proxy입니다.",
                )
            )
        elif label == "early_or_cast_proxy" and confidence >= 0.3:
            findings.append(
                CoachFinding(
                    key="release_cast_proxy",
                    category="fusion",
                    severity="medium",
                    confidence=confidence,
                    evidence="릴리스 타이밍 proxy가 손/팔 선행 쪽으로 잡힙니다.",
                    interpretation="전환 초반 손이 먼저 나가면 샤프트가 세워지고 바깥에서 덮이는 궤도가 되기 쉽습니다.",
                    action="손을 공 쪽으로 던지기보다 오른팔을 몸 앞에 둔 채 가슴 회전으로 내려오세요.",
                    priority="릴리스 타이밍",
                    drill="오른팔 수건 하프스윙으로 전환 첫 1/3에서 손이 튀어나가지 않게 반복.",
                    checkpoint="다운스윙 초반 그립이 공 쪽으로 먼저 밀리지 않는지 봅니다.",
                    caution="공/페이스 데이터가 없는 tempo-shaft-impact proxy입니다.",
                )
            )

    sequencing = fusion_metrics.get("sequencing")
    if isinstance(sequencing, dict):
        label = str(sequencing.get("label") or "")
        confidence = _safe_float(sequencing.get("confidence"), 0.0)
        if label == "rushed_transition_proxy" and confidence >= 0.35:
            findings.append(
                CoachFinding(
                    key="sequence_rushed_proxy",
                    category="fusion",
                    severity="medium",
                    confidence=confidence,
                    evidence="시퀀싱 proxy가 빠른 전환을 가리킵니다.",
                    interpretation="탑 이후 하체-몸통-팔-클럽 순서가 만들어지기 전에 손이 내려올 가능성이 있습니다.",
                    action="스윙 속도를 줄이고 탑에서 하체/가슴 회전이 먼저 시작되는 느낌을 확인하세요.",
                    priority="전환 순서",
                    drill="탑 정지 1초 후 50% 속도로 골반-가슴-팔 순서만 확인하는 빈스윙.",
                    checkpoint="다운스윙 첫 움직임이 손이 아니라 몸 회전인지 봅니다.",
                    caution="2D pose/tempo 기반 proxy입니다.",
                )
            )
        elif label == "arms_dominant_proxy" and confidence >= 0.3:
            findings.append(
                CoachFinding(
                    key="sequence_arms_dominant_proxy",
                    category="fusion",
                    severity="medium",
                    confidence=confidence,
                    evidence="시퀀싱 proxy가 손/팔 우세 패턴을 가리킵니다.",
                    interpretation="회전 여유가 생기기 전에 팔이 먼저 내려오면 임팩트 전후 재현성이 흔들릴 수 있습니다.",
                    action="팔을 더 빠르게 쓰기보다 3/4 백스윙에서 몸통 회전과 손 위치를 먼저 맞추세요.",
                    priority="전환 순서",
                    drill="3/4 탑 정지 후 가슴이 공 쪽으로 돌아온 다음 손이 내려오는 순서 드릴.",
                    checkpoint="탑에서 손 높이보다 어깨선 변화와 가슴 회전 시작이 먼저 보이는지 확인합니다.",
                    caution="2D pose/tempo 기반 proxy입니다.",
                )
            )
    return findings


def _finding_by_key(findings: List[CoachFinding], key: str) -> Optional[CoachFinding]:
    return next((finding for finding in findings if finding.key == key), None)


def _composite_findings(findings: List[CoachFinding]) -> List[CoachFinding]:
    composites: List[CoachFinding] = []
    keys = {finding.key for finding in findings}

    if (
        "impact_unstable" in keys
        and "shaft_flat" in keys
        and ("tempo_fast" in keys or "tempo_rushed_transition" in keys)
    ):
        impact = _finding_by_key(findings, "impact_unstable")
        shaft = _finding_by_key(findings, "shaft_flat")
        tempo = _finding_by_key(findings, "tempo_fast") or _finding_by_key(findings, "tempo_rushed_transition")
        confidence = min(0.8, max(0.35, ((impact.confidence if impact else 0.4) + (shaft.confidence if shaft else 0.3) + (tempo.confidence if tempo else 0.5)) / 3.0))
        composites.append(
            CoachFinding(
                key="pattern_late_club_release",
                category="pattern",
                severity="high",
                confidence=confidence,
                evidence="빠른 전환, 낮은 샤프트, 임팩트 불안정이 함께 나타납니다.",
                interpretation="클럽이 몸 뒤에 남은 상태에서 임팩트 직전 손으로 맞추는 보상 패턴일 수 있습니다.",
                action="첫 처방은 스피드를 줄이고, 탑에서 가슴 회전으로 클럽을 몸 앞 공간에 가져온 뒤 릴리스하는 50% 스윙입니다.",
                priority="1순위 패턴",
                drill="50% 속도 펌프 드릴: 탑-허리높이-탑-허리높이-임팩트 순서로 클럽을 몸 앞에 두고 치기.",
                checkpoint="임팩트 직전 클럽헤드가 손 뒤에서 급히 따라오지 않고, 손과 클럽이 몸 앞에서 같이 지나가는지 봅니다.",
                caution=_caution(min(shaft.confidence if shaft else 0.0, confidence), "club_box_proxy" if shaft and shaft.caution else ""),
            )
        )

    if (
        "impact_unstable" in keys
        and "shaft_steep" in keys
        and ("path_outside_in" in keys or "tempo_rushed_transition" in keys)
    ):
        impact = _finding_by_key(findings, "impact_unstable")
        shaft = _finding_by_key(findings, "shaft_steep")
        path = _finding_by_key(findings, "path_outside_in")
        confidence = min(0.82, max(0.4, ((impact.confidence if impact else 0.4) + (shaft.confidence if shaft else 0.5) + (path.confidence if path else 0.5)) / 3.0))
        composites.append(
            CoachFinding(
                key="pattern_over_the_top",
                category="pattern",
                severity="high",
                confidence=confidence,
                evidence="세워진 샤프트와 outside-in 경로, 임팩트 불안정이 같이 보입니다.",
                interpretation="다운스윙 초반 손과 상체가 먼저 덮이면서 클럽이 바깥에서 들어오는 패턴일 수 있습니다.",
                action="오른팔을 몸 앞에 붙인 채 하프스윙으로 내려오고, 손보다 몸통 회전이 먼저 시작되는지 확인하세요.",
                priority="1순위 패턴",
                drill="오른팔 수건 하프스윙: 오른팔이 몸에서 떨어지지 않게 하고 가슴 회전으로 임팩트까지 연결.",
                checkpoint="다운스윙 첫 1/3에서 손이 공 쪽으로 나가거나 샤프트가 급히 세워지는지 봅니다.",
                caution=_caution(min(shaft.confidence if shaft else 0.0, path.confidence if path else 0.0), ""),
            )
        )

    if "backswing_short" in keys and ("tempo_fast" in keys or "tempo_rushed_transition" in keys):
        backswing = _finding_by_key(findings, "backswing_short")
        tempo = _finding_by_key(findings, "tempo_fast") or _finding_by_key(findings, "tempo_rushed_transition")
        confidence = min(0.78, max(0.35, ((backswing.confidence if backswing else 0.3) + (tempo.confidence if tempo else 0.5)) / 2.0))
        composites.append(
            CoachFinding(
                key="pattern_rushed_short_swing",
                category="pattern",
                severity="medium",
                confidence=confidence,
                evidence="백스윙이 작고 전환 템포도 빠릅니다.",
                interpretation="충분한 회전이 만들어지기 전에 다운스윙이 시작되어 손 위주로 맞추는 패턴일 수 있습니다.",
                action="공을 치기 전 3/4 백스윙 위치를 먼저 만들고, 그 위치에서 같은 리듬으로 내려오는 반복 드릴을 우선하세요.",
                priority="2순위 패턴",
                drill="3/4 탑 정지 드릴: 탑 위치를 먼저 만들고 1초 정지 후 70% 속도로 피니시까지 회전.",
                checkpoint="백스윙을 크게 만들려고 팔만 드는 보상이 아니라 어깨 회전 폭이 늘어나는지 봅니다.",
            )
        )

    return composites


def _rank_findings(findings: List[CoachFinding]) -> List[CoachFinding]:
    return sorted(
        findings,
        key=lambda finding: (
            SEVERITY_ORDER.get(finding.severity, 9),
            CATEGORY_ORDER.get(finding.category, 9),
            -finding.confidence,
        ),
    )


def _suppress_redundant_summary_findings(findings: List[CoachFinding]) -> List[CoachFinding]:
    keys = {finding.key for finding in findings}
    suppressed = set()
    if "pattern_late_club_release" in keys:
        suppressed.update({"tempo_fast", "tempo_rushed_transition", "shaft_flat", "release_late_proxy"})
    if "pattern_over_the_top" in keys:
        suppressed.update({"shaft_steep", "path_outside_in", "release_cast_proxy"})
    if "pattern_rushed_short_swing" in keys:
        suppressed.update({"backswing_short", "sequence_arms_dominant_proxy", "sequence_rushed_proxy"})
    return [finding for finding in findings if finding.key not in suppressed]


def build_coach_findings(
    tempo: Dict[str, object],
    shaft_plane: Dict[str, object],
    backswing: Dict[str, object],
    impact_stability: Dict[str, object],
    readiness: Dict[str, object],
    tracking: Dict[str, object],
    ball: Dict[str, object],
    swing_plane: Dict[str, object],
    body_metrics: Optional[Dict[str, object]] = None,
    fusion_metrics: Optional[Dict[str, object]] = None,
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
    findings.extend(_body_findings(body_metrics or {}))
    findings.extend(_fusion_findings(fusion_metrics or {}))
    findings.extend(_quality_findings(tracking, ball, readiness))
    findings.extend(_composite_findings(findings))
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
    body_metrics: Optional[Dict[str, object]] = None,
    fusion_metrics: Optional[Dict[str, object]] = None,
    *,
    limit: int = 6,
) -> List[str]:
    comments = [
        finding.comment()
        for finding in _suppress_redundant_summary_findings(
            build_coach_findings(
                tempo,
                shaft_plane,
                backswing,
                impact_stability,
                readiness,
                tracking,
                ball,
                swing_plane,
                body_metrics or {},
                fusion_metrics or {},
            )
        )
    ]

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
    body_metrics: Optional[Dict[str, object]] = None,
    fusion_metrics: Optional[Dict[str, object]] = None,
    *,
    limit: int = 8,
    suppress_redundant: bool = False,
) -> List[Dict[str, object]]:
    findings = build_coach_findings(
        tempo,
        shaft_plane,
        backswing,
        impact_stability,
        readiness,
        tracking,
        ball,
        swing_plane,
        body_metrics or {},
        fusion_metrics or {},
    )
    if suppress_redundant:
        findings = _suppress_redundant_summary_findings(findings)
    return [
        finding.to_dict()
        for finding in findings[:limit]
    ]
