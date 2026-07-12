# Event Label Fixtures

Hand-labeled swing event fixtures for local replay experiments.

Each fixture points at saved Pi artifacts and defines the expected event timing:

```json
{
  "jobId": "uuid",
  "toleranceMs": 80,
  "labels": {
    "addressMs": 0,
    "topMs": 391,
    "impactMs": 503,
    "finishMs": 838
  },
  "artifacts": {
    "metaPath": "/tmp/job.debug.meta.json",
    "bodyPath": "/home/ray/data/body/job.json",
    "analysisPath": "/home/ray/data/analysis/job.json"
  }
}
```

Run:

```bash
python3 scripts/replay_events.py
python3 scripts/replay_events.py --allow-missing
python3 scripts/replay_events.py --diagnostics fixtures/event_labels/<jobId>.json
python3 scripts/replay_events.py fixtures/event_labels/<jobId>.json
```

`957e5457-4d13-46bf-88c6-65c467af8487` is also a self-contained regression
for the video-frame clock. Run `python3 scripts/check_body_event_selector.py`
to verify that its pose event frames remain aligned to the video timeline and
that its sparse club track is not mistaken for impact evidence. Full replay
still requires the saved Pi meta and body artifacts referenced by the fixture.

When the Pi artifacts are present, run the stronger end-to-end verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_golden_957_runtime.py
```

It checks the artifact checksums, restores the video frame clock from the
missing-frame meta, confirms the stored pose event labels, and verifies that
current false-positive club/ball detections are withheld from coaching.

The first target is not model accuracy. It is repeatable scoring so event logic changes can be
compared against known failures before deployment.
