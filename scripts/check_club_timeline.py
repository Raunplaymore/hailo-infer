"""Source PTS must survive FPS heuristics and stay separate from legacy clocks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.coach_pipeline import _normalize_times

frames = [{"frame": i, "t": t * 4, "sourceTimeMs": t} for i, t in enumerate([0, 17, 51, 68])]
assert _normalize_times(frames, 30, 100) == [0, 17, 51, 68]
assert _normalize_times([{"t": 100, "sourceTimeMs": 0}], 60, 17) == [0]
for bad in (None, float("nan"), -1, 0):
    invalid = [dict(f) for f in frames]
    invalid[1]["sourceTimeMs"] = bad
    assert _normalize_times(invalid, 30, 100) == [i * 1000 / 30 for i in range(4)]
print("club source PTS checks passed")
