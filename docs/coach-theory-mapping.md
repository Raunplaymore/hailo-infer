# Coach Theory Mapping

This document defines how `coach_commentary.py` turns observed swing events and
metrics into coaching findings. The goal is not to claim a full 3D diagnosis;
the current system produces prioritized, evidence-bounded coaching prompts from
2D pose, club tracking, and event timing.

## Core Principles

1. Event timing is the backbone.
   - Address, top, impact, and finish must be detected before tempo or sequence
     comments are meaningful.
   - If event detection is weak, comments must be phrased as reference signals,
     not final diagnosis.

2. Kinematic sequence is treated as a proxy.
   - A full golf sequence is pelvis, thorax, arms, then club.
   - Current inputs do not provide robust 3D pelvis/thorax angular velocity, so
     sequence comments are only proxy findings based on tempo, pose, and club
     motion.

3. Ball flight laws are out of scope unless ball tracking is present.
   - Without launch direction, curve, and face information, the system should not
     say "slice", "hook", "open face", or "closed face" as a final diagnosis.
   - Path comments remain 2D club-path tendencies.

4. Composite patterns outrank isolated symptoms.
   - A combined pattern such as fast transition + flat shaft + unstable impact is
     more useful than three separate comments.
   - The UI should show the composite first and suppress redundant sub-findings
     in the summary view.

5. Tracking quality caps diagnostic strength.
   - Swing findings can still be useful when tracking is weak, but their
     confidence must be capped and the wording must remain provisional.
   - Quality findings themselves are not capped, because they explain why the
     rest of the analysis should be treated carefully.

## Finding Map

| Finding | Signals | Theory rationale | Required caution |
| --- | --- | --- | --- |
| `pattern_late_club_release` | fast/rushed tempo, flat shaft, unstable impact | Club may stay behind the body and require a late hand compensation near impact. | Mark as proxy when shaft uses `club_box_proxy` or tracking is weak. |
| `pattern_over_the_top` | steep shaft, outside-in path or rushed transition, unstable impact | Hands/upper body may dominate transition, moving the club over the plane. | Avoid ball-flight claims unless ball data exists. |
| `pattern_rushed_short_swing` | short backswing and fast/rushed tempo | Downswing starts before rotation and hand/club travel have enough time to organize. | Confirm backswing source, because pose wrist and club box have different reliability. |
| `release_late_proxy` | flat shaft and unstable impact | Release timing may be late because the club is still behind the body. | State that face/ball data is missing. |
| `release_cast_proxy` | steep shaft and unstable impact | Early hand/arm action can steepen the shaft and destabilize impact. | State that face/ball data is missing. |
| `sequence_rushed_proxy` | very low tempo ratio or short top-to-impact interval | Transition may begin before pelvis/thorax/arms/club order can form. | State that this is a 2D pose/tempo proxy. |
| `sequence_arms_dominant_proxy` | fast tempo plus short backswing or limited shoulder proxy | Arm motion may lead before the torso creates space. | State that this is a 2D pose/tempo proxy. |

## Metric Interpretation

### Tempo

`tempo.ratio = backswingMs / downswingMs`.

- `< 1.7`: rushed transition.
- `1.7 - 2.4`: fast, but usable with transition caution.
- `2.4 - 3.6`: usable range.
- `> 3.6`: slow/paused transition risk.

These thresholds are coaching heuristics. They should be evaluated against user
videos, not treated as universal biomechanics constants.

### Shaft Plane

The system prefers `club_head + club_handle`. If only club boxes are available,
`club_box_proxy` must be treated as lower confidence.

- `flat`: club is low/behind in 2D downswing view.
- `steep`: shaft is upright or moving over plane in 2D view.
- `neutral`: no strong shaft-plane issue in current observation.

### Impact Stability

Impact stability is a repeatability proxy around the impact event. It does not
measure ball compression, face angle, or strike location.

### Body Metrics

Head stability and shoulder turn are 2D pose proxies:

- `headStability`: nose movement normalized by shoulder width.
- `shoulderTurnProxy`: shoulder line angle change in the camera plane.

These should never be described as exact head movement, shoulder turn degree, or
X-factor.

### Quality Gate

Before ranking final coach findings, non-quality findings are adjusted by
`trackingQuality`:

- `tracking.label == weak` or score `< 0.25`: cap confidence at `0.30`.
- score `< 0.50`: cap confidence at `0.55`.
- score `>= 0.50`: no tracking cap.

When the cap applies, non-quality findings must include caution text that the
finding is a reference signal for repeated confirmation, not a final diagnosis.
This prevents strong-looking pattern labels from overclaiming when club head,
handle, ball, or person tracking is sparse.

## Validation Expectations

`scripts/check_coach_commentary.py` must cover:

- late release composite suppression of redundant tempo/shaft findings;
- over-the-top composite suppression of redundant shaft/path findings;
- short backswing plus rushed tempo composite;
- body pose findings for unstable head and limited shoulder proxy;
- fusion proxy findings for rushed sequence and cast/hand-leading release;
- preservation of `theory` rationale in structured `coachFindings`.
- tracking quality caps low-confidence swing findings and adds caution text.

`pi_web/scripts/check-analysis-normalization.mjs` must cover:

- `coachFindings` preservation;
- `theory` preservation;
- `metrics.fusion` preservation;
- event timing and progress preservation.

## Review Workflow

Use `scripts/check_coach_commentary.py` for pass/fail regression coverage.
Use `scripts/preview_coach_findings.py` when changing coaching language or
priority rules, because it prints representative findings in the same order a
coach or user would read them:

```bash
python3 scripts/preview_coach_findings.py
python3 scripts/preview_coach_findings.py low_tracking_late_release --json
```

The preview cases should include at least one weak-tracking sample, one stable
neutral sample, and one strong pattern sample. If a new user video reveals a
recurring failure mode, add a reduced metric case here before tuning rules.

## References

- GolfDB event sequencing: https://arxiv.org/abs/1903.06528
- Tempo timing research by Grober: https://arxiv.org/abs/physics/0611291
- Downswing timing and release phase by Grober: https://arxiv.org/abs/1001.1137
