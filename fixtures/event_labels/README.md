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
python3 scripts/replay_events.py fixtures/event_labels/<jobId>.json
```

The first target is not model accuracy. It is repeatable scoring so event logic changes can be
compared against known failures before deployment.
