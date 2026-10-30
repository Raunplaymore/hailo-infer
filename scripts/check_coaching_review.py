"""Regression checks for silhouette evidence, source identity and bounded work."""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services import coaching_review as review


def samples(shift=8):
    base = [{"timeMs": t, "edgePx": 80, "y": .6, "uncertaintyPx": 1,
             "baseline": True, "cameraStable": True} for t in (100, 134, 168)]
    return base + [{"timeMs": t, "edgePx": 80 + (shift if t >= 700 else 0), "y": .6,
                    "uncertaintyPx": 1, "baseline": False, "cameraStable": True} for t in range(500, 1101, 60)]


def summarize(rows):
    return review.summarize_observation(rows, "job", 1000, 270, 480, 1)


incomplete = {**review.unavailable('job', 'SILHOUETTE_UNRELIABLE'),
              'excludedFrames': [{'timeMs': i * 60, 'reason': 'CAMERA_MOTION_OR_BACKGROUND_UNCLEAR'} for i in range(9)]}
assert review._exclusion_reason(incomplete)['reason'] == 'CAMERA_MOTION_OR_BACKGROUND_UNCLEAR'
assert incomplete['reason'] == 'SILHOUETTE_UNRELIABLE', 'do not change saved cache data in memory'
mixed = {**incomplete, 'excludedFrames': [{'reason': 'CAMERA_MOTION_OR_BACKGROUND_UNCLEAR'}] * 4 + [{'reason': 'SILHOUETTE_UNRELIABLE'}] * 4}
assert review._exclusion_reason(mixed)['reason'] == 'PHASE_COVERAGE_INSUFFICIENT'
assert review._exclusion_reason({**incomplete, 'excludedFrames': incomplete['excludedFrames'][:6]})['reason'] == 'SILHOUETTE_UNRELIABLE'
timing = {**incomplete, 'excludedFrames': [{'reason': 'VIDEO_TIMING_UNAVAILABLE'}] * 7}
assert review._exclusion_reason(timing)['reason'] == 'VIDEO_TIMING_UNAVAILABLE'
with tempfile.TemporaryDirectory() as temporary:
    cache = Path(temporary) / 'cached.json'
    cache.write_text(json.dumps({'identity': {'fixture': 1}, 'result': incomplete}))
    before = cache.read_bytes()
    assert review._read_cache(cache, {'fixture': 1}, 'job')['reason'] == 'CAMERA_MOTION_OR_BACKGROUND_UNCLEAR'
    assert cache.read_bytes() == before, 'correct the displayed reason without rewriting the old cache'


positive = summarize(samples())
assert positive["evidence"]["observation"] == "toward_ball"
assert positive["diagnosisStatus"] == "unvalidated" and "practice" not in positive
assert "검사한 구간" in positive["summary"] and "추정한 공 방향" in positive["summary"]
assert positive["evidence"]["focusMs"] <= 960
assert positive["evidence"]["endMs"] == 920, "replay ends at the last measured pre-impact sample"
assert positive["evidence"]["endMs"] < positive["evidence"]["impactMs"]
assert summarize(samples(1))["evidence"]["observation"] == "no_clear_shift"
late = samples(0)
for row in late:
    if row["timeMs"] >= 980:
        row["edgePx"] += 12
assert summarize(late)["evidence"]["observation"] == "no_clear_shift", "post-impact change is not early movement"
single = samples(0)
single[6]["edgePx"] += 12
assert summarize(single)["evidence"]["observation"] == "no_clear_shift", "single-frame spike is not sustained evidence"
mirrored = samples()
for row in mirrored:
    row["edgePx"] = 270-row["edgePx"]
mirror_result = review.summarize_observation(mirrored, "job", 1000, 270, 480, -1)
assert mirror_result["evidence"]["observation"] == "toward_ball"
assert mirror_result["evidence"]["shiftPx"] == positive["evidence"]["shiftPx"]
translated = samples(0)
for row in translated:
    dx = row["timeMs"] / 100
    row.update(transform=[[1,0,dx],[0,1,-5]])
translation_result = summarize(translated)
assert translation_result["reason"] == "CAMERA_BACKGROUND_AMBIGUOUS", "a decisive background correction needs independent camera evidence"
for row in translated:
    if row["timeMs"] >= 700:
        row["edgePx"] += 8
assert summarize(translated)["reason"] == "CAMERA_BACKGROUND_AMBIGUOUS"
constant_transform = samples()
for row in constant_transform:
    row["transform"] = [[1,0,7],[0,1,-5]]
assert summarize(constant_transform)["evidence"]["observation"] == "toward_ball", "a constant offset must not hide relative movement"
for dx in (-5, 5):
    moving_background = samples(0)
    for row in moving_background:
        if not row["baseline"]:
            row.update(edgePx=80-dx, transform=[[1,0,dx],[0,1,0]])
    assert summarize(moving_background)["reason"] == "CAMERA_BACKGROUND_AMBIGUOUS", "a stationary person cannot inherit background motion"
for corrected_edge, dx in ((83, -1), (82, 1)):
    borderline_correction = samples(0)
    for row in borderline_correction:
        if not row["baseline"]:
            row.update(edgePx=corrected_edge, transform=[[1,0,dx],[0,1,0]])
    assert summarize(borderline_correction)["reason"] == "CAMERA_BACKGROUND_AMBIGUOUS", "a small correction cannot change the movement verdict"
mirrored_correction = samples(0)
for row in mirrored_correction:
    if not row["baseline"]:
        row.update(edgePx=77, transform=[[1,0,1],[0,1,0]])
assert review.summarize_observation(mirrored_correction, "job", 1000, 270, 480, -1)["reason"] == "CAMERA_BACKGROUND_AMBIGUOUS"
shared_window = samples(0)
for row in shared_window:
    if not row["baseline"]:
        row.update(edgePx=83, transform=[[1,0,-1 if row["timeMs"] < 620 else 0],[0,1,0]])
assert summarize(shared_window)["evidence"]["focusMs"] == 740, "replay a sustained window shared by both assumptions"
disjoint_windows = samples(0)
for row in disjoint_windows:
    if not row["baseline"]:
        early = row["timeMs"] < 740
        row.update(edgePx=83 if early else 82, transform=[[1,0,-1 if early else 1],[0,1,0]])
assert summarize(disjoint_windows)["reason"] == "CAMERA_BACKGROUND_AMBIGUOUS", "different positive windows do not corroborate a replay"
for dx, expected in ((2, "ready"), (2.1, "unavailable")):
    borderline = samples(0)
    for row in borderline:
        if not row["baseline"]:
            row["transform"] = [[1,0,dx],[0,1,0]]
    assert summarize(borderline)["status"] == expected, "correction dependence uses the existing noise limit"
for key, value in [("cameraStable", False), ("edgePx", float("nan")), ("y", 1), ("uncertaintyPx", 6)]:
    bad = samples(); bad[5][key] = value
    assert summarize(bad)["status"] == "unavailable"
assert summarize(samples()[:6] + samples()[7:])["status"] == "ready", "one missing sample may retain phase coverage"
assert summarize(samples()[1:])["status"] == "unavailable", "all three baseline observations are required"
assert summarize([s for s in samples() if s["timeMs"] < 800])["status"] == "unavailable", "missing approach to impact cannot pass"
bad = samples(); bad[0]["edgePx"] += 10
assert summarize(bad)["reason"] == "ADDRESS_UNSTABLE"
assert summarize([s for s in samples() if not 650 < s["timeMs"] < 850])["status"] == "unavailable"
assert summarize(samples() + [samples()[0]])["status"] == "unavailable"
height_changed = samples()
for row in height_changed:
    if not row["baseline"] and row["timeMs"] <= 960:
        row["y"] += 16 / 480
assert summarize(height_changed)["reason"] == "PELVIS_HEIGHT_UNCERTAIN"
height_at_limit = samples()
for row in height_at_limit:
    if not row["baseline"]:
        row["y"] += 6 / 480
assert summarize(height_at_limit)["status"] == "ready", "the sampling-band boundary remains usable"
after_impact_height = samples()
for row in after_impact_height:
    if row["timeMs"] >= 980:
        row["y"] += 16 / 480
assert summarize(after_impact_height)["status"] == "ready", "post-impact height is outside the observed interval"
address_height = samples()
address_height[0]["y"] += 16 / 480
assert summarize(address_height)["reason"] == "PELVIS_HEIGHT_UNCERTAIN"

points = {f"{side}_{joint}": [x, .4 if joint == "shoulder" else .6, .99] for joint, x in (("hip", .4), ("shoulder", .48), ("wrist", .55)) for side in ("left", "right")}
body = {"frames": [{"timeMs": t, "keypointSource": "observed", "keypoints": copy.deepcopy(points)} for t in (70,100,130)]}
assert review._direction(body, 100) == 1
for frame in body["frames"]:
    for point in frame["keypoints"].values():
        point[0] = 1-point[0]
assert review._direction(body, 100) == -1
for frame in body["frames"]:
    frame["keypoints"]["left_wrist"][2] = .3
assert review._direction(body, 100) == -1, "a visible near wrist can calibrate direction while far wrist is occluded"
for frame in body["frames"]:
    frame["keypoints"]["right_wrist"][2] = .3
assert review._direction(body, 100) is None
for frames in (None, "bad", [None, {"keypointSource": "observed", "timeMs": 100, "keypoints": "bad"}]):
    assert review._direction({"frames": frames}, 100) is None

with patch('app.services.practice_review.source_frames', return_value=(30, [0,33,67,100])), tempfile.TemporaryDirectory() as temp:
    root = Path(temp); upload = root/"uploads"; data = root/"data"
    upload.mkdir(); data.mkdir()
    source = upload/"video.mp4"; source.write_bytes(b"test-video")
    import time
    time.sleep(.01)
    body_path = data/"body.json"
    body_path.write_text(json.dumps({"jobId": "job", "source": {"inputPath": str(source)},
        "meta": {"durationMs": 1500, "sampleStride": 1, "sampledFrames": 4},
        "frames": [{"frameIndex": i, "timeMs": i * 33} for i in range(4)]}))
    settings = SimpleNamespace(data_dir=data, upload_dir=upload)
    payload = SimpleNamespace(jobId="job", inputPath=str(source), bodyPath=str(body_path), addressMs=100,
                              comparisonMs=900, impactMs=1000, viewpoint="down_the_line")
    with patch.object(review, "_observe", return_value=positive.copy()) as observe:
        first = review.review_coaching(settings, payload)
        assert review.review_coaching(settings, payload) == first
        assert observe.call_count == 1, "a repeat read must not run MediaPipe again"
        cached_path = data / "coaching-review" / "job.json"
        cached_before = cached_path.read_bytes()
        updated_guide = {"version": "movement-guide-v1", "status": "unavailable", "testMarker": "fresh"}
        with patch.object(review, "build_movement_guide", return_value=updated_guide):
            assert review.review_coaching(settings, payload)["movementGuide"] == updated_guide
        assert observe.call_count == 1 and cached_path.read_bytes() == cached_before, "refresh guide without rerunning or rewriting silhouette"
        with patch.object(review, "CACHE_REVISION", review.CACHE_REVISION + 1):
            review.review_coaching(settings, payload)
            assert observe.call_count == 2, "algorithm change invalidates old rejection cache"
        observe.reset_mock()
        payload.impactMs = 1010
        review.review_coaching(settings, payload)
        assert observe.call_count == 1, "changed phase invalidates cache"
        body_path.write_text(body_path.read_text()+" ")
        review.review_coaching(settings, payload)
        assert observe.call_count == 2, "changed pose artifact invalidates cache"
        payload.impactMs = 1020
        review.REVIEW_LOCK.acquire()
        try:
            try:
                review.review_coaching(settings, payload)
                raise AssertionError("busy work must not start")
            except review.CoachingBusy:
                pass
        finally:
            review.REVIEW_LOCK.release()
        assert observe.call_count == 2
    with patch.object(review, "_observe", return_value=review.unavailable("job", "ANALYSIS_RUNTIME_UNAVAILABLE")) as observe:
        assert review.review_coaching(settings, payload)["reason"] == "ANALYSIS_RUNTIME_UNAVAILABLE"
        review.review_coaching(settings, payload)
        assert observe.call_count == 2, "transient runtime failures must remain retryable"
    original_load = review.load_practice_body
    def change_during_load(*args):
        value = original_load(*args)
        body_path.write_text(body_path.read_text()+" ")
        return value
    with patch.object(review, "load_practice_body", side_effect=change_during_load), patch.object(review, "_observe") as observe:
        try:
            review.review_coaching(settings, payload)
            raise AssertionError("pose changed while reading")
        except ValueError:
            pass
        assert observe.call_count == 0
    def change_source(_body, _payload):
        source.write_bytes(b"changed-source")
        return positive.copy()
    with patch.object(review, "_observe", side_effect=change_source):
        try:
            review.review_coaching(settings, payload)
            raise AssertionError("must not cache a result for a changed input")
        except ValueError:
            pass
    assert not list(data.rglob("*.tmp"))
    payload.jobId = "../outside"
    try:
        review.review_coaching(settings, payload)
        raise AssertionError("path traversal accepted")
    except ValueError:
        pass

try:
    import cv2
    import numpy as np
except ImportError:
    print("OpenCV checks unavailable on this Python; run on Pi candidate environment")
else:
    gray = np.random.default_rng(71).integers(0, 256, (480,270), dtype="uint8")
    mask = np.zeros((480,270), dtype="float32")
    motion = review._background_motion(gray, gray.copy(), mask, mask)
    assert motion is not None and np.allclose(motion[0], [[1,0,0],[0,1,0]], atol=.01)
    moved = cv2.warpAffine(gray, np.float32([[1,0,7],[0,1,0]]), (270,480))
    motion = review._background_motion(gray, moved, mask, mask)
    assert motion is not None and abs(motion[0][0,2]-7)<.1 and abs(motion[0][1,2])<.1
    far = cv2.warpAffine(gray, np.float32([[1,0,20],[0,1,0]]), (270,480))
    assert review._background_motion(gray, far, mask, mask) is None
    for angle, scale in [(.6,1),(0,1.01)]:
        transform = cv2.getRotationMatrix2D((135,240),angle,scale)
        changed = cv2.warpAffine(gray, transform, (270,480))
        motion = review._background_motion(gray, changed, mask, mask)
        assert motion is not None, (angle, scale)
        assert np.max(np.abs(motion[0]-transform)) < .2
    for angle, scale in [(2,1),(0,1.04)]:
        changed = cv2.warpAffine(gray, cv2.getRotationMatrix2D((135,240),angle,scale), (270,480))
        assert review._background_motion(gray, changed, mask, mask) is None
    assert review._background_motion(gray, gray, np.ones_like(mask), mask) is None
    assert review._background_motion(np.zeros_like(gray), np.zeros_like(gray), mask, mask) is None
    deformed = gray.copy()
    deformed[:240] = np.roll(gray[:240], 3, axis=1)
    deformed[240:] = np.roll(gray[240:], -3, axis=1)
    assert review._background_motion(gray, deformed, mask, mask) is None, "inconsistent background cannot be corrected as camera motion"
    mask[100:400,80:150] = 1
    hips = [SimpleNamespace(x=.42,y=.6,visibility=.99) for _ in range(33)]
    result = SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=hips), segmentation_mask=mask)
    assert review._silhouette(result, 1, 270, 480)["edgePx"] == 80
    assert review._silhouette(result, -1, 270, 480)["edgePx"] == 149
    sloped_mask = np.zeros((480,270), dtype="float32")
    cv2.fillPoly(sloped_mask, [np.array([[40,100],[120,400],[190,400],[190,100]], dtype=np.int32)], 1)
    def fixed_contour(y):
        points = [SimpleNamespace(x=.46,y=y,visibility=.99) for _ in range(33)]
        return review._silhouette(SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=points),
                                                  segmentation_mask=sloped_mask), 1, 270, 480)
    base, shifted_height = fixed_contour(.6), fixed_contour(.6+16/480)
    assert base and shifted_height and shifted_height["edgePx"] > base["edgePx"]
    stationary = [{**base,"timeMs":t,"baseline":True,"cameraStable":True} for t in (100,134,168)]
    stationary += [{**shifted_height,"timeMs":t,"baseline":False,"cameraStable":True} for t in range(500,921,60)]
    assert summarize(stationary)["reason"] == "PELVIS_HEIGHT_UNCERTAIN", "height alone must not become a forward movement observation"
    transform = cv2.getRotationMatrix2D((135,240), .6, 1.01)
    transform[:,2] += [2,-3]
    xy = transform @ np.array([.42*270,.6*480,1])
    moving_result = SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=[SimpleNamespace(x=xy[0]/270,y=xy[1]/480,visibility=.99) for _ in range(33)]), segmentation_mask=cv2.warpAffine(mask,transform,(270,480)))
    stabilized = review._stabilize_pose(moving_result,transform,270,480)
    assert abs(review._silhouette(stabilized,1,270,480)["edgePx"]-80) <= 1, "camera rotation/scale must not become pelvis shift"
    transformed_samples = samples(0)
    for row in transformed_samples:
        row["transform"] = transform.tolist()
    observed = summarize(transformed_samples)["evidence"]
    assert observed["shiftPx"] == 0
    point = observed["points"][-1]
    assert point["referenceTop"][0] != point["referenceBottom"][0], "reference line rotates with image"
    assert np.allclose([point["x"]*270,point["y"]*480],transform@np.array([80,.6*480,1]),atol=.001)
    hips[23].visibility = .2
    assert review._silhouette(result, 1, 270, 480) is None
    hips[23].visibility = .99
    mask[100:400,190:240] = 1
    assert review._silhouette(result, 1, 270, 480) is None, "second large foreground component must be rejected"
    print("background translation, texture, foreground exclusion and silhouette component checks passed")

    import mediapipe as mp
    assert review.CACHE_REVISION == 9, "the correction-consistency check must not reuse prior silhouette results"
    class FakeCapture:
        def __init__(self, mode):
            self.mode, self.time, self.seeks = mode, 0, []
        def set(self, _property, value):
            self.time = value; self.seeks.append(value)
            if self.mode == "seek" and value == 100:
                raise cv2.error("reference seek failed")
        def get(self, _property):
            if self.mode == "nonfinite" and self.time == 100:
                return float("nan")
            offsets = {"timing": 51, "boundary": 20}
            return self.time + (offsets.get(self.mode, 0) if self.time == 100 else 0)
        def read(self):
            if self.mode == "decode" and self.time == 100:
                return False, None
            if self.mode == "empty" and self.time == 100:
                return True, None
            return True, np.full((20, 20, 3), int(self.time) % 251, dtype="uint8")
        def release(self):
            pass
    class FakePose:
        def __init__(self, mode): self.mode = mode
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def process(self, frame):
            if self.mode == "pose_error" and frame[0,0,0] == 100:
                raise RuntimeError("reference pose unavailable")
            if self.mode == "mask_error" and frame[0,0,0] == 100:
                class FailingMask:
                    def copy(self): raise RuntimeError("reference mask unavailable")
                return SimpleNamespace(pose_landmarks=object(), segmentation_mask=FailingMask())
            missing = self.mode == "pose" and frame[0, 0, 0] == 100
            return SimpleNamespace(pose_landmarks=None if missing else object(),
                                   segmentation_mask=None if missing else np.zeros((20,20), dtype="float32"))
    def reference_test(mode, impact):
        capture = FakeCapture(mode); references = []
        def background(first, _frame, _first_mask, _mask):
            references.append(int(first[0,0])); return np.array([[1.,0.,0.],[0.,1.,0.]]), 0.
        payload = SimpleNamespace(jobId="test", inputPath="/tmp/source.mp4", addressMs=0, impactMs=impact)
        body = {"meta": {"durationMs": 1500}, "frames": []}
        result = {**review.unavailable("test", "fixture"), "status": "ready"}
        with patch.object(cv2, "VideoCapture", return_value=capture), \
             patch.object(mp.solutions.pose, "Pose", return_value=FakePose(mode)), \
             patch.object(review, "_direction", return_value=1), \
             patch.object(review, "_background_motion", side_effect=background), \
             patch.object(review, "_stabilize_pose", side_effect=lambda pose, *_args: pose), \
             patch.object(review, "_silhouette", return_value={"edgePx": 8., "y": .5, "uncertaintyPx": 1.}), \
             patch.object(review, "summarize_observation", return_value=result) as summarize_observation:
            assert review._observe(body, payload)["status"] == "ready"
        samples = summarize_observation.call_args.args[0]
        assert [s["timeMs"] for s in samples if s["baseline"]] == [0,34,68], "reference choice must not move the baseline samples"
        return capture.seeks, references
    seeks, references = reference_test("valid", 1000)
    assert seeks[0] == 100 and references[0] == 100, "use the nearby frame for background registration"
    for mode in ("decode", "empty", "timing", "nonfinite", "pose", "seek", "pose_error", "mask_error"):
        seeks, references = reference_test(mode, 1000)
        assert seeks[0] == 100 and references[0] == 0, f"{mode} failure must preserve the original reference"
    seeks, references = reference_test("boundary", 610)
    assert seeks[0] == 100 and references[0] == 0, "actual reference must remain before the pre-impact interval"
    seeks, references = reference_test("valid", 590)
    assert seeks[0] == 0 and references[0] == 0, "never use a reference inside the pre-impact interval"
    print("nearby registration reference, original baseline and fallback checks passed")

print("coaching evidence timing, noise, mirror, calibration, source identity, caching and concurrency checks passed")
