"""Forward-only swing phase decoder.

The decoder intentionally knows nothing about pose, club, or optical-flow
implementations.  Callers provide one evidence row per timestamp, making the
same decoder usable for pose-relative movement, club tracking, and dynamic ROI
motion without turning any individual signal into an event classifier.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

PHASES = (
    "ready",
    "backswing",
    "top",
    "downswing",
    "impact_candidate",
    "follow_through",
    "finish",
)

MIN_PHASE_MS = {
    "ready": 0.0,
    "backswing": 90.0,
    "top": 25.0,
    "downswing": 70.0,
    "impact_candidate": 15.0,
    "follow_through": 90.0,
    "finish": 0.0,
}
MIN_ADVANCE_SCORE = 0.18


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score(row: Dict[str, Any], phase: str) -> float:
    scores = row.get("scores")
    if not isinstance(scores, dict):
        return 0.0
    return _clamp(_number(scores.get(phase)))


def _time(row: Dict[str, Any], index: int) -> float:
    return _number(row.get("timeMs"), index * 33.333)


def decode_swing_phases(evidence: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose the highest-scoring monotonic full-swing phase path.

    A state can remain in its current phase or move exactly one phase forward.
    The transition is available only after the current phase has lasted its
    minimum duration.  This prevents Finish-before-Top predictions without
    imposing a single backswing or downswing tempo on every player.
    """

    rows = [row for row in evidence if isinstance(row, dict)]
    if len(rows) < len(PHASES):
        return {"available": False, "reason": "insufficient_evidence_frames", "phasePath": []}

    rows.sort(key=lambda row: _time(row, 0))
    # phase -> (total score, entered time, previous phase, path)
    states: Dict[int, Tuple[float, float, Optional[int], List[int]]] = {
        0: (_score(rows[0], "ready"), _time(rows[0], 0), None, [0])
    }

    for index, row in enumerate(rows[1:], start=1):
        now = _time(row, index)
        next_states: Dict[int, Tuple[float, float, Optional[int], List[int]]] = {}
        for phase_index, (total, entered_at, _previous, path) in states.items():
            phase = PHASES[phase_index]
            stay = (total + _score(row, phase), entered_at, phase_index, [*path, phase_index])
            best = next_states.get(phase_index)
            if best is None or stay[0] > best[0]:
                next_states[phase_index] = stay

            if phase_index >= len(PHASES) - 1:
                continue
            if now - entered_at < MIN_PHASE_MS[phase]:
                continue
            next_phase_index = phase_index + 1
            next_phase = PHASES[next_phase_index]
            next_score = _score(row, next_phase)
            # A transition is evidence-led.  Without this floor a long stream
            # of neutral frames could eventually manufacture a complete swing.
            if next_score < MIN_ADVANCE_SCORE:
                continue
            advance = (total + next_score, now, phase_index, [*path, next_phase_index])
            best = next_states.get(next_phase_index)
            if best is None or advance[0] > best[0]:
                next_states[next_phase_index] = advance
        states = next_states

    final = states.get(len(PHASES) - 1)
    if not final:
        return {"available": False, "reason": "incomplete_phase_path", "phasePath": []}

    total, _entered_at, _previous, path = final
    changes: List[Dict[str, Any]] = []
    for index, phase_index in enumerate(path):
        if index == 0 or phase_index != path[index - 1]:
            changes.append({"phase": PHASES[phase_index], "timeMs": round(_time(rows[index], index))})
    times = {item["phase"]: item["timeMs"] for item in changes}
    missing = [phase for phase in PHASES if phase not in times]
    if missing:
        return {"available": False, "reason": "missing_phase_transitions", "phasePath": changes}

    coverage = sum(1 for row in rows if isinstance(row.get("scores"), dict)) / float(len(rows))
    confidence = _clamp((total / float(len(rows))) * coverage)
    return {
        "available": True,
        "events": {
            "addressMs": times["ready"],
            "topMs": times["top"],
            "impactMs": times["impact_candidate"],
            "finishMs": times["finish"],
        },
        "confidence": round(confidence, 3),
        "phasePath": changes,
        "debug": {
            "decoder": "forward-phase-v1",
            "phaseCount": len(PHASES),
            "evidenceFrames": len(rows),
            "minimumPhaseMs": MIN_PHASE_MS,
            "minimumAdvanceScore": MIN_ADVANCE_SCORE,
        },
    }
