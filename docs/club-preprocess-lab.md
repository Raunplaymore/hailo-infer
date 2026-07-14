# Club preprocessing lab

This is an offline experiment. It does not change `hailo-infer` jobs, Pi service orchestration, NAS archive behavior, HEF files, or coach output.

## Run one complete comparison

Run this one command on the Pi after the lab has been deployed. The source remains untouched; derived videos are written only to the lab directory and `/home/ray/uploads/lab/`. It calls the camera metadata endpoint only: it never submits an analysis job, creates coach output, or archives anything to NAS.

```bash
cd /home/ray/hailo-infer && .venv/bin/python scripts/club_preprocess_lab.py run --source /home/ray/uploads/VIDEO.mp4 --body /home/ray/data/body/JOB.json --workspace /home/ray/data/labs/club-preprocess/JOB --job-prefix lab-JOB
```

It creates the following variants and performs one isolated Hailo metadata pass for each:

- `source`: original input reference
- `contrast.mp4`: conservative CLAHE lightness correction
- `wrist-roi.mp4`: static, pose-derived wrist/arm crop enlarged to the original resolution

## Manual mode

`prepare` and `score` remain available when you want to generate or inspect a single stage separately. Do not send lab job IDs to the analysis API.

```bash
cd /home/ray/hailo-infer && .venv/bin/python scripts/club_preprocess_lab.py score --meta source=/tmp/lab-source.meta.json --meta contrast=/tmp/lab-contrast.meta.json --meta wrist-roi=/tmp/lab-wrist-roi.meta.json --output /home/ray/data/labs/club-preprocess/JOB/score.json
```

The score weights `club_head`, `club_handle`, and simultaneous head+handle frames. A variant is only a *candidate* when it gains at least 0.08 score, two club-head frames, and does not lose paired frames. It still requires visual review for false positives and ROI coordinate reprojection before any production proposal.
