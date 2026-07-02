# Event Segmentation Experiment Plan

Goal: build a full-body golf swing event engine from repeatable evidence, not one-off threshold tuning.

## Baseline

Run the current service and wrist-only event logic against every labeled fixture:

```bash
python3 scripts/replay_events.py --allow-missing
```

When Pi artifacts are available locally or on the Pi, omit `--allow-missing` so missing data fails the run.

## Labels

Start with 10 videos that include:

- clean face-on full-body swings
- down-the-line swings
- known failures where top is marked in follow-through
- low club-detection quality cases
- clips with stable pose but sparse club detections

Each fixture must label:

- `addressMs`
- `topMs`
- `impactMs`
- `finishMs`

Initial pass tolerance is `80ms`. Tighten to one sampled frame after the event engine stabilizes.

## Experiments

1. Dense pose event time preservation
   - Do not remap pose-wrist event times to sparse club/motion indices.
   - Compare final `events.*Ms` against `wrist raw`.

2. Wrist source separation
   - Score `left_wrist`, `right_wrist`, and `weighted_midpoint` independently.
   - Penalize candidates that require source switching during a phase.

3. Top detection
   - Use the first backswing local peak after address.
   - Reject peaks after the hand/shoulder has entered follow-through.
   - Require a subsequent downswing drop or direction reversal.

4. Impact detection
   - Use the first post-top return through the address/ball zone.
   - Blend club head or club box evidence only as confirmation, not the primary clock.

5. Finish detection
   - Require impact to be known first.
   - Look for post-impact motion settling or final follow-through hold.
   - Do not force finish to the final frame.

## Promotion Rule

Do not deploy a heuristic change unless:

- every labeled fixture runs through `scripts/replay_events.py`
- top/impact/finish error is within tolerance on known failure fixtures
- no previously passing fixture regresses beyond tolerance
- debug output clearly reports which source selected each event
