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
assert g['tracks']['pelvis']['phase']=='before' and abs(g['tracks']['pelvis']['focusTimeMs']-900)<=30
assert g['tracks']['head']['phase']=='after'
assert 'diagnosis' not in g and 'practice' not in g
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
