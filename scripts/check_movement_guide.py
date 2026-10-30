import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.movement_guide import build_movement_guide

frames=[]
for t in range(0,1501,30):
    x=.45+.04*max(0,1-abs(t-900)/600)
    frames.append({'timeMs':t,'keypointSource':'observed','keypoints':{'nose':[.5,.2+.03*t/1500,.95],'left_hip':[x-.03,.6,.95],'right_hip':[x+.03,.6,.95]}})
body={'meta':{'width':1080,'height':1920,'durationMs':1500},'frames':frames}
def build(b): return build_movement_guide(b,100,1000)
g=build(body);assert g['status']=='ready' and set(g['tracks'])=={'head','pelvis'}
first=build_movement_guide(body,0,1000)
assert first['status']=='ready' and all(track['baseline']['timeMs']==0 for track in first['tracks'].values()), 'the first selected frame must not be shifted to 100ms'
fps30=copy.deepcopy(body)
for i,frame in enumerate(fps30['frames']): frame['timeMs']=i*1000/30
first30=build_movement_guide(fps30,0,1000)
assert first30['status']=='ready' and all(abs(track['baseline']['timeMs']-1000/30)<.01 for track in first30['tracks'].values()), 'use the nearest stable window when the first frame has too few neighbors'
hidden=copy.deepcopy(body)
for frame in hidden['frames']:
    if frame['timeMs'] < 200:
        for point in frame['keypoints'].values(): point[2]=.2
assert build_movement_guide(hidden,0,1000)['status']=='unavailable', 'do not silently substitute a distant frame when the selected start is obscured'
assert g['tracks']['pelvis']['phase']=='before' and abs(g['tracks']['pelvis']['focusTimeMs']-900)<=30
assert g['tracks']['head']['focusTimeMs'] <= 1000, 'head review should start at or before the impact candidate'
assert any(p['timeMs'] > 1000 for p in g['tracks']['head']['points']), 'post-impact review must remain available'
assert 'diagnosis' not in g and 'practice' not in g
for t in (300,600,900,1110):
    bad=copy.deepcopy(body)
    next(f for f in bad['frames'] if f['timeMs']==t)['keypoints']['nose'][2]=.2
    assert set(build(bad).get('tracks',{}))=={'pelvis'}, 'a missing head sample must not use an adjacent point'
    bad=copy.deepcopy(body)
    next(f for f in bad['frames'] if f['timeMs']==t)['keypoints']['left_hip'][2]=.2
    assert set(build(bad).get('tracks',{}))=={'head'}, 'a missing hip sample must not use an adjacent point'
bad=copy.deepcopy(body)
next(f for f in bad['frames'] if f['timeMs']==600)['keypointSource']='interpolated'
assert build(bad)['status']=='unavailable', 'an unobserved frame must not produce either track'
bad=copy.deepcopy(body)
for t in (0,1350):next(f for f in bad['frames'] if f['timeMs']==t)['keypointSource']='interpolated'
assert set(build(bad)['tracks'])=={'head','pelvis'}, 'samples outside the guide interval must not hide it'
bad=copy.deepcopy(body)
next(f for f in bad['frames'] if f['timeMs']==600)['keypoints']['nose'][1]+=.05
assert set(build(bad).get('tracks',{}))=={'pelvis'}, 'unstable smoothed samples must not create a complete track'
bad=copy.deepcopy(body)
for f in bad['frames']: f['keypoints']['nose'][2]=.2
assert set(build(bad)['tracks'])=={'pelvis'},'head rejection must not hide a valid pelvis'
bad=copy.deepcopy(body)
for f in bad['frames']:f['keypointSource']='interpolated'
assert build(bad)['status']=='unavailable'
bad=copy.deepcopy(body);bad['frames']=[f for f in bad['frames'] if not 700<f['timeMs']<1000]
assert build(bad)['status']=='unavailable','gaps cannot be bridged'
bad=copy.deepcopy(body);bad['frames'].append(bad['frames'][0]);assert build(bad)['status']=='unavailable'
for t in [None,False,float('nan'),-1,30001]:assert build_movement_guide(body,100,t)['status']=='unavailable'
for p in ([float('nan'),.2,.9],[.5,.2,1.1],[0,.2,.9]):
 bad=copy.deepcopy(body)
 for f in bad['frames']:f['keypoints']['nose']=p
 assert 'head' not in build(bad).get('tracks',{})
mirror=copy.deepcopy(body)
for f in mirror['frames']:
 for p in f['keypoints'].values():p[0]=1-p[0]
mg=build(mirror)
assert mg['tracks']['pelvis']['focusTimeMs']==g['tracks']['pelvis']['focusTimeMs']
assert abs(mg['tracks']['pelvis']['baseline']['x']-(1-g['tracks']['pelvis']['baseline']['x']))<1e-5
assert build_movement_guide(body,1100,1000)['status']=='unavailable'
print('movement guide source, confidence, gaps, timing, mirror and independent track checks passed')
