import copy
import math
import random
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.club_observation import observe, inclination, inclination_bounds, matched_height_levels


def sample():
    frames=[]
    for t in range(0,501,20):
        a=math.radians(30+t*.3);x=.5;y=.2+(t-240)**2/1_000_000
        frames.append({'frame':t//20,'timeMs':t,'handle':{'x':x,'y':y,'confidence':.9},'head':{'x':x+.08*math.cos(a),'y':y-.04*math.sin(a),'confidence':.9}})
    return {'jobId':'test','meta':{'width':1000,'height':2000},'events':{'takeawayMs':0},'overlay':{'clubTimeSource':'video_pts','clubFrames':frames,'poseFrames':[]}}


class ObservationsTest(unittest.TestCase):
    def test_pixels_not_stretched_normalized_coordinates(self):
        result=observe(sample(),500)
        self.assertAlmostEqual(result['anchors'][0]['observation']['shaftDirectionDeg'],30)
        self.assertFalse(result['automaticCoachingReady'])
        self.assertEqual(result['anchors'][1]['requestedTimeMs'],240)
        self.assertTrue(all(p['status']=='observation' for p in result['phases']))
    def test_matched_height_uses_real_frames_on_opposite_sides(self):
        result=observe(sample(),500);pair=result['matchedGripHeight']
        self.assertIsNotNone(pair)
        self.assertLess(pair['backswing']['timeMs'],240)
        self.assertGreater(pair['downswing']['timeMs'],240)
        self.assertLess(abs(pair['backswing']['handle']['y']-pair['downswing']['handle']['y']),.04)
    def test_gap_does_not_create_continuous_rotation(self):
        a=sample()
        for f in a['overlay']['clubFrames']:
            if 300<=f['timeMs']<=400:f['head']['confidence']=.1
        phase=observe(a,500)['phases'][-1]
        self.assertEqual(phase['status'],'withheld');self.assertIsNone(phase['shaftRotation2dDeg'])
    def test_invalid_clock_and_outside_impact(self):
        for mutation in ('clock','order','outside'):
            a=copy.deepcopy(sample())
            if mutation=='clock':a['overlay']['clubTimeSource']=None
            if mutation=='order':a['overlay']['clubFrames'][1]['timeMs']=0
            with self.assertRaises(ValueError):observe(a,1000 if mutation=='outside' else 500)
    def test_angle_wrap_is_not_a_full_rotation(self):
        a=sample()
        for f in a['overlay']['clubFrames']:
            r=math.radians(170+f['timeMs']*.05);g=f['handle'];f['head'].update(x=g['x']+.08*math.cos(r),y=g['y']-.04*math.sin(r))
        self.assertAlmostEqual(observe(a,500)['phases'][0]['shaftRotation2dDeg'],12)

class SensitivityTest(unittest.TestCase):
    def test_inclination_folds_vertical_and_horizontal(self):
        self.assertEqual(inclination(179),1)
        self.assertEqual(inclination(-91),89)
        self.assertEqual(inclination(270),90)

    def test_short_segment_has_no_direction_bound(self):
        self.assertEqual(inclination_bounds({'projectedLengthPx':10,'shaftDirectionDeg':30},5),[0,90])
        with self.assertRaises(ValueError):inclination_bounds({'projectedLengthPx':10},-1)

    def test_disk_bounds_contain_independently_perturbed_endpoints(self):
        rng=random.Random(27)
        for angle in (-179,-91,-45,0,45,89,179):
            for length in (8,30,300):
                for radius in (2,5,10):
                    lo,hi=inclination_bounds({'projectedLengthPx':length,'shaftDirectionDeg':angle},radius)
                    x,y=length*math.cos(math.radians(angle)),length*math.sin(math.radians(angle))
                    for _ in range(100):
                        offsets=[]
                        for endpoint in range(2):
                            a=rng.uniform(0,2*math.pi);r=radius*math.sqrt(rng.random())
                            offsets.append((r*math.cos(a),r*math.sin(a)))
                        actual=inclination(math.degrees(math.atan2(y+offsets[0][1]-offsets[1][1],x+offsets[0][0]-offsets[1][0])))
                        self.assertGreaterEqual(actual+1e-9,lo);self.assertLessEqual(actual-1e-9,hi)

    def test_five_levels_and_nested_bounds(self):
        result=observe(sample(),500)
        self.assertEqual(len(result['matchedGripHeights']),5)
        for row in result['matchedGripHeights']:
            if row['status']!='observation':continue
            tests=row['sensitivity']
            for a,b in zip(tests,tests[1:]):
                self.assertLessEqual(b['differenceBoundsDeg'][0],a['differenceBoundsDeg'][0])
                self.assertGreaterEqual(b['differenceBoundsDeg'][1],a['differenceBoundsDeg'][1])

    def test_height_match_can_disappear_under_error(self):
        def row(t,y):
            return {'timeMs':t,'handle':{'y':y},'shaftDirectionDeg':45,'projectedLengthPx':300}
        pairs=[row(100,.519),row(300,.519)]
        level=matched_height_levels(pairs,{'handle':{'y':.8}},{'handle':{'y':.2}},0,200,400,1000)[2]
        self.assertEqual(level['status'],'observation')
        self.assertEqual(level['sensitivity'][0]['direction'],'pair_uncertain')

    def test_reselection_can_reverse_nominal_difference(self):
        def row(t,y,angle):
            return {'timeMs':t,'handle':{'y':y},'shaftDirectionDeg':angle,'projectedLengthPx':300}
        # At 50% height the exact match is +10deg, but a nearby downswing
        # frame has -10deg; endpoint error can change which frame is selected.
        pairs=[row(100,.5,40),row(300,.5,50),row(320,.505,30)]
        levels=matched_height_levels(pairs,{'handle':{'y':.8}},{'handle':{'y':.2}},0,200,400,1000)
        middle=levels[2]
        self.assertEqual(middle['inclinationDifferenceDeg'],10)
        self.assertEqual(middle['sensitivity'][0]['direction'],'unresolved')
        self.assertEqual(middle['sensitivity'][0]['candidateFrameCounts'],[1,2])

if __name__=='__main__':unittest.main()
