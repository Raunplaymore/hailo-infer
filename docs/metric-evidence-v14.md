# Metric Evidence v14

## Purpose

v14 separates three concepts that were previously mixed:

- a detector or pose observation,
- a derived numeric value,
- a validated golf claim that may drive coaching.

A value may exist internally without being safe to show as a diagnosis. The public contract uses
`eventValidation.metricEvidence` as the source of truth, with `confirmed`, `reference`, and
`withheld` states.

## Public metric policy

| Metric | v14 state | Reason |
| --- | --- | --- |
| Tempo duration/ratio | `reference` | Pose phase timing describes elapsed time but does not prove body-part sequencing. |
| Shaft plane | `confirmed` only with paired head/handle evidence | Requires at least 8 paired samples and confidence at least 0.35. |
| Backswing size | `reference` | Single-camera wrist or club displacement is useful only as a 2D observation. |
| Impact stability | `withheld` | Point spread inside one swing is motion, not repeatability across swings. |
| Swing path | `withheld` | Target line, viewpoint, handedness, and camera calibration are not yet normalized. |
| Ball launch | `withheld` | Ball identity continuity and calibrated flight coordinates are not yet validated. |
| Body proxies | `reference` | 2D head/shoulder/hip values remain visible for inspection but do not create corrective coaching. |

The previous raw path, impact-spread, ball, and fusion calculations remain available under
`debug.unvalidatedMetrics` for offline comparison. They must not be rendered as user-facing
diagnoses.

## Quality semantics

`analysisQuality.score` is an observation coverage index. It combines club observation coverage,
pose phase-path coverage, and pose coverage. It is not a probability that a diagnosis is correct,
and the UI must not label it as model confidence.

The legacy top-level `confidence` field mirrors this score during the additive contract migration.

## Coaching gate

Only evidence explicitly admitted by the metric policy may generate a coaching finding:

- pose-based tempo does not generate sequencing or rushed-transition advice;
- impact stability, path, ball flight, release, and sequencing do not generate findings;
- 2D negative body proxies do not generate findings;
- a positive pose-based backswing-size observation may be shown as a low-risk maintenance note;
- capture/readiness and tracking quality may generate recording-quality guidance.

## Release checks

- Every metric has its own evidence state and reason code.
- A globally usable event path does not automatically confirm every metric.
- `score` is never formatted as metric confidence in the front end.
- Withheld impact, path, and ball values have no public numeric payload.
- The UI labels observation coverage separately from per-metric evidence.
- Raw unvalidated values are debug-only.

## Next validation milestone

Build a labeled replay set covering camera viewpoint, handedness, address/top/impact/finish, club
head and handle identity, and ball identity. A metric can move from `reference` or `withheld` only
after reporting held-out error and calibration by capture condition. Temporal tracking should be
evaluated against that set before its candidates are allowed to confirm a golf claim.
