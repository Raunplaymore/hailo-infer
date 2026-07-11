#!/usr/bin/env python3
"""Print readable coach finding previews for representative metric samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.coach_commentary import build_coach_comments, build_coach_finding_debug


MetricCase = Dict[str, Any]


CASES: dict[str, MetricCase] = {
    "low_tracking_late_release": {
        "description": "Weak tracking, fast-ish tempo, flat shaft, unstable impact. Similar to the IMG_5082 debug flow.",
        "tempo": {"backswingMs": 352, "downswingMs": 175, "ratio": 2.01},
        "shaft_plane": {"label": "flat", "confidence": 0.27, "angleDeg": 15.3, "source": "club_box_proxy"},
        "backswing": {
            "label": "adequate",
            "score": 0.61,
            "clubTravelRatio": 0.13,
            "handTravelRatio": 0.13,
            "source": "pose_wrist",
        },
        "impact_stability": {"label": "unstable", "score": 0},
        "readiness": {"label": "unknown"},
        "tracking": {"label": "weak", "score": 0.11, "personFrames": 0, "ballFrames": 0},
        "ball": {"launchDirection": "unknown"},
        "swing_plane": {"label": "outside-in", "confidence": 0.2, "source": "club_box_endpoint"},
        "body_metrics": {},
        "fusion_metrics": {"releaseTiming": {"label": "late_proxy", "confidence": 0.5}},
    },
    "neutral_stable": {
        "description": "Fair tracking with neutral shaft, usable tempo, and stable impact.",
        "tempo": {"backswingMs": 430, "downswingMs": 140, "ratio": 3.07},
        "shaft_plane": {"label": "neutral", "confidence": 0.55, "angleDeg": 48.0, "source": "head_handle"},
        "backswing": {"label": "adequate", "score": 0.97, "clubTravelRatio": 0.4, "source": "club_motion"},
        "impact_stability": {"label": "stable", "score": 0.82},
        "readiness": {"label": "ready"},
        "tracking": {"label": "fair", "score": 0.42, "personFrames": 20, "ballFrames": 4},
        "ball": {"launchDirection": "center"},
        "swing_plane": {"label": "inside-out", "confidence": 0.42, "source": "hybrid"},
        "body_metrics": {},
        "fusion_metrics": {},
    },
    "over_the_top_proxy": {
        "description": "Rushed transition, steep shaft, outside-in path, and unstable impact.",
        "tempo": {"backswingMs": 120, "downswingMs": 120, "ratio": 1.0},
        "shaft_plane": {"label": "steep", "confidence": 0.62, "angleDeg": 66.0, "source": "head_handle"},
        "backswing": {"label": "short", "score": 0.2, "clubTravelRatio": 0.04, "source": "club_motion"},
        "impact_stability": {"label": "unstable", "score": 0.35},
        "readiness": {"label": "ready"},
        "tracking": {"label": "fair", "score": 0.4, "personFrames": 12, "ballFrames": 2},
        "ball": {"launchDirection": "right"},
        "swing_plane": {"label": "outside-in", "confidence": 0.5, "source": "hybrid"},
        "body_metrics": {},
        "fusion_metrics": {},
    },
    "stuck_inside_release": {
        "description": "Flat shaft, inside-out path, and unstable impact. Similar to a club stuck behind the body pattern.",
        "tempo": {"backswingMs": 435, "downswingMs": 134, "ratio": 3.25},
        "shaft_plane": {"label": "flat", "confidence": 0.45, "angleDeg": 30.1, "source": "club_box_proxy"},
        "backswing": {"label": "adequate", "score": 0.97, "clubTravelRatio": 0.41, "source": "club_motion"},
        "impact_stability": {"label": "unstable", "score": 0.22},
        "readiness": {"label": "unknown"},
        "tracking": {"label": "fair", "score": 0.42, "personFrames": 0, "ballFrames": 0},
        "ball": {"launchDirection": "unknown"},
        "swing_plane": {"label": "inside-out", "confidence": 0.45, "source": "hybrid"},
        "body_metrics": {},
        "fusion_metrics": {},
    },
}


def _findings(case: MetricCase) -> list[dict[str, object]]:
    return build_coach_finding_debug(
        case["tempo"],
        case["shaft_plane"],
        case["backswing"],
        case["impact_stability"],
        case["readiness"],
        case["tracking"],
        case["ball"],
        case["swing_plane"],
        case.get("body_metrics") or {},
        case.get("fusion_metrics") or {},
        suppress_redundant=True,
    )


def _comments(case: MetricCase) -> list[str]:
    return build_coach_comments(
        case["tempo"],
        case["shaft_plane"],
        case["backswing"],
        case["impact_stability"],
        case["readiness"],
        case["tracking"],
        case["ball"],
        case["swing_plane"],
        case.get("body_metrics") or {},
        case.get("fusion_metrics") or {},
    )


def print_case(name: str, case: MetricCase) -> None:
    print(f"\n## {name}")
    print(case["description"])
    for index, finding in enumerate(_findings(case), start=1):
        confidence = finding.get("confidence")
        priority = finding.get("priority")
        key = finding.get("key")
        print(f"\n{index}. [{priority}] {key} confidence={confidence}")
        print(f"   evidence: {finding.get('evidence')}")
        print(f"   action: {finding.get('action')}")
        if finding.get("drill"):
            print(f"   drill: {finding.get('drill')}")
        if finding.get("checkpoint"):
            print(f"   checkpoint: {finding.get('checkpoint')}")
        if finding.get("caution"):
            print(f"   caution: {finding.get('caution')}")

    print("\n   summary comments:")
    for comment in _comments(case):
        print(f"   - {comment}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview coach findings for representative metric samples.")
    parser.add_argument("case", nargs="*", choices=sorted(CASES), help="Case names to print. Defaults to all.")
    parser.add_argument("--json", action="store_true", help="Print structured JSON instead of readable text.")
    args = parser.parse_args()

    selected = args.case or sorted(CASES)
    if args.json:
        payload = {
            name: {
                "description": CASES[name]["description"],
                "findings": _findings(CASES[name]),
                "comments": _comments(CASES[name]),
            }
            for name in selected
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    for name in selected:
        print_case(name, CASES[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
