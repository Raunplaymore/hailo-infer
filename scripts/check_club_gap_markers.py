"""Keep detector gaps across overlay sampling without changing observed points."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.coach_pipeline import _build_overlay_payload


def fixture(count, head_missing=(), handle_missing=(), head_low=()):
    frames = [{'frame': i, 'sourceTimeMs': i * 17} for i in range(count)]
    def track(missing, low=()):
        return [{'frame':i, 't':i * 17, 'x':.2, 'y':.4, 'conf':.2 if i in low else .9}
                for i in range(count) if i not in missing]
    return frames, track(set(head_missing), set(head_low)), track(set(handle_missing))


class ClubGapMarkerTests(unittest.TestCase):
    def test_missing_both_marks_next_observation_without_changing_points(self):
        frames, head, handle = fixture(4, head_missing=[1], handle_missing=[1])
        inputs = copy.deepcopy((frames, head, handle))
        result = _build_overlay_payload(None, head, handle, frames)['clubFrames']
        self.assertEqual([row['frame'] for row in result],[0,2,3])
        self.assertEqual(result[0]['headValidUntilMs'],17)
        self.assertEqual(result[0]['handleValidUntilMs'],17)
        self.assertTrue(result[1]['headBreakBefore'])
        self.assertTrue(result[1]['handleBreakBefore'])
        old = _build_overlay_payload(None,head,handle)['clubFrames']
        self.assertEqual([{k:v for k,v in row.items() if not k.endswith(('BreakBefore','ValidUntilMs'))} for row in result],old)
        self.assertEqual((frames,head,handle),inputs)

    def test_downsampling_preserves_head_and_both_missing_gaps(self):
        frames, head, handle = fixture(721, head_low=[101], head_missing=[300], handle_missing=[300])
        result = _build_overlay_payload(None,head,handle,frames)['clubFrames']
        self.assertLessEqual(len(result),360)
        first = next(row for row in result if row['frame'] > 101)
        previous = next(row for row in reversed(result) if row['frame'] < 101)
        self.assertEqual(previous['headValidUntilMs'],101*17)
        self.assertNotIn('handleValidUntilMs',previous)
        self.assertTrue(first['headBreakBefore'])
        self.assertNotIn('handleBreakBefore',first)
        second = next(row for row in result if row['frame'] > 300)
        self.assertTrue(second['headBreakBefore'])
        self.assertTrue(second['handleBreakBefore'])

    def test_downsampling_complete_observations_does_not_make_gaps(self):
        frames, head, handle = fixture(721)
        result = _build_overlay_payload(None,head,handle,frames)['clubFrames']
        self.assertLessEqual(len(result),360)
        self.assertFalse(any(k.endswith(('BreakBefore','ValidUntilMs')) for row in result for k in row))

    def test_ends_visibility_at_last_missing_frame_even_without_later_point(self):
        frames, head, handle = fixture(4, head_missing=[3], handle_missing=[3])
        result = _build_overlay_payload(None,head,handle,frames)['clubFrames']
        self.assertEqual(result[-1]['headValidUntilMs'],51)
        self.assertEqual(result[-1]['handleValidUntilMs'],51)

    def test_unknown_or_inconsistent_source_clock_does_not_invent_markers(self):
        frames, head, handle = fixture(4, head_missing=[1], handle_missing=[1])
        for broken in ('missing','duplicate_time','duplicate_frame','mismatch','boolean'):
            source, heads = copy.deepcopy(frames), copy.deepcopy(head)
            if broken == 'missing': source[1].pop('sourceTimeMs')
            elif broken == 'duplicate_time': source[1]['sourceTimeMs']=0
            elif broken == 'duplicate_frame': source[1]['frame']=0
            elif broken == 'mismatch': heads[1]['t']=999
            else: source[1]['sourceTimeMs']=True
            result = _build_overlay_payload(None,heads,handle,source)['clubFrames']
            self.assertFalse(any(k.endswith(('BreakBefore','ValidUntilMs')) for row in result for k in row),broken)


if __name__ == '__main__':
    unittest.main()
