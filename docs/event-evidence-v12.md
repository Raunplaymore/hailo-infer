# Event Evidence v12

## Runtime flow

`pi_camera` writes normalized service7 detections and the video frame clock. `pi_service` adds the
body artifact path and passes both artifacts to `hailo-infer`. `coach_pipeline.py` separates body
pose, accepted club observations, event selection, event validation, metrics, and commentary.
`pi_web` normalizes the additive `eventValidation` fields and labels reference events explicitly.

The single source of truth for whether an event or metric may be shown is now
`eventValidation.eventQuality` and `eventValidation.metricAvailability`. Summary and coaching text
are generated only after these gates have been applied.

## Production regression

Job `c04d1b58-026d-490f-939d-80c52ccc7781` reproduced the v11 regression locally:

- pose coverage: 96/96
- club head: 45 raw, 16 geometry-accepted (35.6%)
- handle: 55 raw, 9 geometry-accepted (16.4%)
- club box: 56 raw, 55 geometry-accepted (98.2%)
- v11 result: 0/4 visible events, duplicate `POSE_CLUB_EVENT_CONFLICT`, all event metrics withheld
- v11 summary: leaked the pre-gate `tempo 3.44:1`

The forward phase decoder selected top 1835 ms and impact 2369 ms with confidence 0.699. The
legacy wrist extrema selected top 634 ms and impact 801 ms. Both are pose-derived, so their
disagreement is a diagnostic warning rather than independent evidence of failure. Around the body
impact, accepted head evidence at 2202/2236 ms and accepted club evidence at 2336/2402/2436/2469
ms bracket the candidate. This is enough for a reference event, not enough to unlock confirmed
tempo, impact, path, or shaft metrics.

## Physical interpretation of detector rules

- `club_handle`: a person-sized or shaft-sized box is not a grip observation. The bbox-size reject
  remains a hard reject.
- `club_head`: a fixed small wrist radius is not a complete physical model because the head is one
  shaft length from the hands. Wrist distance remains useful against static background candidates,
  but a distance failure must not erase independent pose events.
- `club`: its box is a proxy observation. It may support temporal bracketing and trajectory
  continuity, but cannot confirm impact or shaft direction by itself.
- `golf_ball`, `person`, and readiness classes are excluded from event confirmation.

No detector threshold was relaxed in v12. The accepted/rejected counts for the production fixture
remain unchanged, limiting false-positive regression.

## Evidence states

- `absent`: no raw observation
- `rejected`: raw observations exist but fail object geometry
- `reference`: pose or accepted proxy observations support a timestamp but cannot unlock dependent
  metrics
- `confirmed`: independent club-head evidence satisfies the impact window

The overall validation state is `usable`, `partial`, or `withheld`. `partial` exposes the available
pose reference events while all dependent metrics remain withheld; it does not require a reference
impact. `withheld` is reserved for cases where the pose phase path itself is not usable.

## Shaft interpolation boundary

Missing head/handle points may be interpolated for visualization only when both endpoints bracket
the gap, the gap is at most three video frames, velocity direction is consistent, and the implied
shaft length stays inside a robust per-swing range. Interpolated points must never create the two
independent samples required for a confirmed shaft or impact metric. Longer or one-sided gaps stay
reference-only.

## Minimum, recommended, and model work

The v12 minimum fix deduplicates pose-source divergence, admits temporally bracketed club proxy
evidence as reference-only, preserves body findings, and removes withheld numbers from summary/UI.

The recommended next step is a multi-candidate temporal tracker: retain more than the top detection
per class, score predicted position, acceleration, wrist-relative shaft length, direction continuity,
and class confidence, then solve the best path across frames. Confidence calibration must be measured
against labeled frame sequences rather than chosen from one job.

The long-term model step is a club-only detector when a labeled evaluation set shows either head or
handle confirmed coverage below 60% in the top/downswing/impact interval, or when more than 20% of
accepted observations remain static-background identities after temporal tracking. The service7
person/ball/readiness classes should not be expanded as a workaround.

## Release criteria

- v11 baseline on the Pi: 7/7 stored v11 results were fully withheld.
- production regression fixture: 4/4 events restored as reference, 0 event-dependent metrics
  incorrectly unlocked, duplicate conflict codes reduced to one warning.
- detector safety: head/handle/club accepted counts remain unchanged by the minimum fix.
- summary safety: no tempo, impact, path, or shaft number is emitted when its availability is
  `withheld`.
- pose-only safety: usable pose address/top/finish ends as `partial`, while impact and every
  club-dependent metric remain `withheld` when club evidence is absent.
- rollout target: fully withheld rate below 40% on at least 20 representative swings, with zero
  confirmed event metrics from a reference-only impact.
