#!/usr/bin/env python3
"""Geometry, rejection and source-identity checks for the guided coach."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.practice_review import measure_posture, review_saved_body, preview_saved_body


def body_with_angles(a=40, b=20, width=1080, height=1920, mirror=False):
    frames=[]
    for t, angle in [(100+i*30,a) for i in range(4)]+[(500+i*30,b) for i in range(4)]:
        dx=300*math.sin(math.radians(angle))/width
        dy=300*math.cos(math.radians(angle))/height
        x=.5-dx if mirror else .5+dx
        points={}
        for side in ('left','right'):
            points[f'{side}_shoulder']=[x,.65-dy,.9]
            points[f'{side}_hip']=[.5,.65,.9]
        frames.append({'timeMs':t,'keypoints':points,'keypointSource':'observed'})
    return {'jobId':'job','meta':{'width':width,'height':height,'durationMs':2000},'frames':frames}


def measure(body):
    return measure_posture(body,'job',145,545)


def main():
    for width,height in ((1080,1920),(1920,1080)):
        for mirror in (True,False):
            result=measure(body_with_angles(width=width,height=height,mirror=mirror))
            assert result['status']=='review_candidate', result
            assert result['evidence']['changeDeg']==20
            assert len(result['practice']['steps'])==3
            assert 'confidence' not in result
    assert measure(body_with_angles(40,38))['status']=='no_clear_change'
    assert measure(body_with_angles(30,40))['status']=='no_clear_change'
    assert measure(body_with_angles(0,0))['status']=='withheld'
    base=body_with_angles()
    for replacement in (None,[.5,.2,.1],[float('nan'),.2,.9],[1.2,.2,.9],[.5,.2,2]):
        bad=copy.deepcopy(base)
        for f in bad['frames']:
            for side in ('left','right'):f['keypoints'][f'{side}_shoulder']=replacement
        assert measure(bad)['status']=='withheld'
    bad=copy.deepcopy(base)
    for f in bad['frames']:f['keypointSource']='interpolated'
    assert measure(bad)['status']=='withheld'
    bad=copy.deepcopy(base);bad['frames']=bad['frames'][::3]
    assert measure(bad)['status']=='withheld'
    bad=copy.deepcopy(base);bad['frames']+=bad['frames']
    assert measure(bad)['status']=='withheld'
    # Cannot combine address's left side and comparison's right side.
    bad=copy.deepcopy(base)
    for f in bad['frames']:
        f['keypoints'][('right' if f['timeMs']<300 else 'left')+'_shoulder'][2]=.1
    assert measure(bad)['status']=='withheld'
    # Reject jitter and bilateral disagreement rather than selecting a nicer result.
    bad=copy.deepcopy(base)
    for side in ('left','right'):bad['frames'][0]['keypoints'][side+'_shoulder'][0]=.52
    assert measure(bad)['status']=='withheld'
    bad=copy.deepcopy(base)
    stable=body_with_angles(40,40)
    for i,f in enumerate(bad['frames']):f['keypoints']['right_shoulder']=stable['frames'][i]['keypoints']['right_shoulder']
    assert measure(bad)['status']=='withheld'
    for a,b in ((145,100),(145,200),(145,2500),(float('nan'),545),(True,545)):
        try:measure_posture(base,'job',a,b)
        except ValueError:pass
        else:raise AssertionError('invalid times accepted')
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);uploads=root/'uploads';uploads.mkdir();data=root/'data';data.mkdir()
        source=uploads/'clip.mp4';source.write_bytes(b'video identity fixture')
        artifact=data/'pose.json'
        payload=SimpleNamespace(jobId='job',inputPath=str(source),bodyPath=str(artifact),addressMs=145,comparisonMs=545)
        base['source']={'inputPath':str(source)};artifact.write_text(json.dumps(base))
        settings=SimpleNamespace(upload_dir=uploads,data_dir=data)
        assert review_saved_body(settings,payload)['status']=='review_candidate'
        preview = preview_saved_body(settings,payload)
        assert {k:v for k,v in preview.items() if k!='movementGuide'} == {'ok': True, 'jobId':'job', 'status':'ready', 'addressMs':145, 'comparisonMs':545}
        assert not any(k in preview for k in ('evidence','practice','changeDeg'))
        payload.addressMs=0
        preview = preview_saved_body(settings,payload)
        assert preview['status']=='ready' and preview['addressMs']>=100
        payload.addressMs=145
        artifact.write_text(json.dumps({**base,'frames':[]}))
        assert preview_saved_body(settings,payload)['status']=='unavailable'
        artifact.write_text(json.dumps(base))
        for change in ({'jobId':'other'},{'source':{'inputPath':str(uploads/'different.mp4')}}):
            artifact.write_text(json.dumps({**base,**change}))
            try:review_saved_body(settings,payload)
            except ValueError:pass
            else:raise AssertionError('mismatched body accepted')
        artifact.write_text(json.dumps(base));os.utime(source,(source.stat().st_atime,artifact.stat().st_mtime+1))
        try:review_saved_body(settings,payload)
        except ValueError:pass
        else:raise AssertionError('stale body accepted')
    print('practice review aspect ratio, mirror, quality, timing and identity checks passed')

if __name__=='__main__':main()
