# pose-tcn-v1

Frozen human-review candidate from recovery stages 2–5 (2026-10-26).

- SHA256: `70b338b519565f870ed1fce044a0453f828773e53b9fa9d024391b3b4cf57733`
- NumPy NPZ, 23,268 parameters, 88,561 bytes; no PyTorch or Hailo compilation required for inference.
- Existing training-permitted footage only: 43 unique training clips and 5 validation clips. The eight evaluation clips were excluded from fitting and checkpoint selection.
- Inference implementation: `scripts/pose_event_model.py`, unchanged from the frozen experiment.
- Expose only an impact reference candidate for human video review. No calibrated accuracy/confidence claim, no automatic coaching. Other event predictions are not returned by the review API.
- Evidence: sibling hailo-model artifact `artifacts/recovery/stages2-5-2026-10-26/`.
