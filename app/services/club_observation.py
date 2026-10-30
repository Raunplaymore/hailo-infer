#!/usr/bin/env python3
"""Offline 2D club observations. No coaching labels or production writes."""
import math
import statistics


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def point(value):
    return isinstance(value, dict) and all(finite(value.get(k)) for k in ('x', 'y', 'confidence')) and value['confidence'] >= .3 and all(0 <= value[k] <= 1 for k in ('x', 'y'))


def angle_delta(a, b):
    return (b - a + 180) % 360 - 180


def nearest(rows, time, tolerance=50):
    best = min(rows, key=lambda row: abs(row['timeMs'] - time), default=None)
    return best if best and abs(best['timeMs'] - time) <= tolerance else None


def inclination(direction):
    """Unsigned image-plane shaft inclination to horizontal, in degrees."""
    return abs((direction + 90) % 180 - 90)


def inclination_bounds(row, radius_px):
    if not finite(radius_px) or radius_px < 0:
        raise ValueError('Endpoint radius must be nonnegative')
    length = row['projectedLengthPx']
    if length <= 2 * radius_px:
        return [0.0, 90.0]
    # Two endpoints in radius-r disks move their difference vector by at most 2r.
    error = math.degrees(math.asin(2 * radius_px / length))
    angle = inclination(row['shaftDirectionDeg'])
    return [max(0.0, angle-error), min(90.0, angle+error)]


def frame_edge_risk(row):
    return any(not isinstance(row.get(key), dict) or row[key].get('frameEdge') is not False
               for key in ('head', 'handle'))


def matched_height_levels(pairs, start_grip, top_grip, takeaway, top, impact_ms, height):
    if not start_grip or not top_grip or start_grip['handle']['y']-top_grip['handle']['y'] <= .05:
        return []
    sides = [[p for p in pairs if takeaway <= p['timeMs'] < top],
             [p for p in pairs if top < p['timeMs'] <= impact_ms]]
    results = []
    for fraction in (.2, .35, .5, .65, .8):
        target = start_grip['handle']['y'] + fraction*(top_grip['handle']['y']-start_grip['handle']['y'])
        selected = [min(side, key=lambda p: abs(p['handle']['y']-target), default=None) for side in sides]
        if any(p is None or abs(p['handle']['y']-target) > .02 for p in selected):
            results.append({'heightFraction':fraction,'status':'unavailable','sensitivity':[]})
            continue
        before, after = selected
        difference = inclination(after['shaftDirectionDeg'])-inclination(before['shaftDirectionDeg'])
        tests = []
        for radius in (2, 5, 10):
            bounds = []
            counts = []
            edge_risk = False
            for side, chosen in zip(sides, selected):
                closest = abs(chosen['handle']['y']-target)
                # Each candidate-to-target distance can change by 2r/H, so a
                # competitor up to 4r/H behind the closest point can become best.
                competitors = [p for p in side if abs(p['handle']['y']-target) <= closest+4*radius/height+1e-12]
                intervals = [inclination_bounds(p, radius) for p in competitors]
                edge_risk = edge_risk or any(frame_edge_risk(p) for p in competitors)
                bounds.append([min(v[0] for v in intervals), max(v[1] for v in intervals)])
                counts.append(len(competitors))
            low, high = bounds[1][0]-bounds[0][1], bounds[1][1]-bounds[0][0]
            available = all(abs(p['handle']['y']-target)+2*radius/height <= .02 for p in selected) and not edge_risk
            tests.append({'endpointRadiusPx':radius,'pairAvailabilityStable':available,'differenceBoundsDeg':[round(low,2),round(high,2)],
                          'direction':'pair_uncertain' if not available else 'steeper' if low>0 else 'flatter' if high<0 else 'unresolved',
                          'candidateFrameCounts':counts, **({'frameEdgeRisk':True} if edge_risk else {})})
        results.append({'heightFraction':fraction,'status':'observation','targetGripY':target,
                        'backswing':before,'downswing':after,
                        'directionDifference2dDeg':round(angle_delta(before['shaftDirectionDeg'],after['shaftDirectionDeg']),1),
                        'inclinationDifferenceDeg':round(difference,2),'sensitivity':tests,
                        'meaning':'same projected grip height, not same 3D posture or a good/bad criterion'})
    return results


def observe(analysis, impact_ms):
    meta = analysis.get('meta', {})
    width, height = meta.get('width'), meta.get('height')
    overlay = analysis.get('overlay', {})
    if overlay.get('clubTimeSource') != 'video_pts':
        raise ValueError('Source PTS required')
    if not all(finite(v) and v > 0 for v in (width, height, impact_ms)):
        raise ValueError('Valid image dimensions and impact candidate required')
    raw = overlay.get('clubFrames', [])
    times = [r.get('timeMs') for r in raw]
    if not times or not all(finite(t) and t >= 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Strictly increasing video timestamps required')
    if impact_ms > times[-1] + 50:
        raise ValueError('Impact candidate outside observed timeline')
    grips, pairs = [], []
    for row in raw:
        if point(row.get('handle')):
            grips.append(row)
        if not all(point(row.get(k)) for k in ('head', 'handle')):
            continue
        head, grip = row['head'], row['handle']
        dx, dy = (head['x'] - grip['x']) * width, (grip['y'] - head['y']) * height
        if math.hypot(dx, dy) < 1:
            continue
        pairs.append({**row, 'shaftDirectionDeg': math.degrees(math.atan2(dy, dx)), 'projectedLengthPx': math.hypot(dx,dy)})
    # shortcut: this is a grip-height anchor, not a validated anatomical top event.
    # Replace with reviewed event labels before using phases for coaching decisions.
    candidates = []
    takeaway = analysis.get('events', {}).get('takeawayMs')
    if not finite(takeaway) or not 0 <= takeaway < impact_ms:
        raise ValueError('Ordered takeaway candidate required')
    for row in grips:
        t = row['timeMs']
        if not takeaway + 100 <= t <= impact_ms - 100:
            continue
        window = [p for p in grips if abs(p['timeMs'] - t) <= 50]
        if len(window) >= 3 and max(p['timeMs'] for p in window) - min(p['timeMs'] for p in window) >= 40:
            candidates.append((statistics.median(p['handle']['y'] for p in window), row['handle']['y'], t))
    if not candidates:
        raise ValueError('Insufficient grip observations for height anchor')
    _, _, top = min(candidates)
    median_length = statistics.median(p['projectedLengthPx'] for p in pairs) if pairs else 0
    # Short projected segments amplify endpoint errors; this is an engineering gate.
    stable_pairs = [p for p in pairs if p['projectedLengthPx'] >= median_length * .25]
    # shortcut: absent metadata rows remain unknown until matched to the source frame timeline.
    raw_index = {row['timeMs']: i for i, row in enumerate(raw)}
    known_gaps, gap_edges, tail_gap_start = [], set(), None
    if stable_pairs:
        first, last = stable_pairs[0], stable_pairs[-1]
        if raw_index[first['timeMs']] or first.get('headBreakBefore') is True or first.get('handleBreakBefore') is True:
            known_gaps.append((raw[0]['timeMs'] if raw_index[first['timeMs']] else 0, first['timeMs']))
        for before, after in zip(stable_pairs, stable_pairs[1:]):
            stops = [before.get(f'{key}ValidUntilMs') for key in ('head', 'handle')]
            has_gap = (raw_index[after['timeMs']] != raw_index[before['timeMs']] + 1
                       or after.get('headBreakBefore') is True or after.get('handleBreakBefore') is True
                       or any(finite(stop) and stop <= after['timeMs'] for stop in stops))
            if has_gap:
                edge = (before['timeMs'], after['timeMs'])
                gap_edges.add(edge)
                known_gaps.append(edge)
        tail_stops = [last.get(f'{key}ValidUntilMs') for key in ('head', 'handle')]
        if raw_index[last['timeMs']] < len(raw) - 1:
            tail_stops.append(raw[raw_index[last['timeMs']] + 1]['timeMs'])
        tail_stops = [stop for stop in tail_stops if finite(stop)]
        if tail_stops:
            tail_gap_start = min(tail_stops)
    start_grip, top_grip = nearest(grips, takeaway), nearest(grips, top)
    levels = matched_height_levels(stable_pairs, start_grip, top_grip, takeaway, top, impact_ms, height)
    matched = next((row for row in levels if row['heightFraction'] == .5 and row['status'] == 'observation'), None)
    phases = [('backswing', takeaway, top), ('transition_window', max(takeaway, top-100), min(impact_ms, top+100)), ('downswing', top, impact_ms)]
    summaries = []
    for name, start, end in phases:
        selected = [r for r in stable_pairs if start <= r['timeMs'] <= end]
        gaps = [b['timeMs']-a['timeMs'] for a,b in zip(selected, selected[1:])]
        boundary_gap = max((selected[0]['timeMs']-start, end-selected[-1]['timeMs'])) if selected else end-start
        max_gap = max([boundary_gap, *gaps])
        linked = [(a,b) for a,b in zip(selected, selected[1:])
                  if b['timeMs']-a['timeMs'] <= 50 and (a['timeMs'],b['timeMs']) not in gap_edges]
        covered_ms = sum(b['timeMs']-a['timeMs'] for a,b in linked)
        angular_steps = [angle_delta(a['shaftDirectionDeg'], b['shaftDirectionDeg']) for a,b in linked]
        spikes = sum(abs(step) > 60 for step in angular_steps)
        missing = (any(gap_start < end and gap_end > start for gap_start, gap_end in known_gaps)
                   or (tail_gap_start is not None and tail_gap_start <= end))
        usable = len(selected) >= 4 and max_gap <= 50 and spikes == 0 and not missing and not any(frame_edge_risk(p) for p in selected)
        first, last = nearest(selected, start), nearest(selected, end)
        summaries.append({
            'phase': name, 'startMs': start, 'endMs': end, 'pairedFrames': len(selected),
            'observedTimeFraction': round(covered_ms/max(1,end-start),3), 'maxGapMs': round(max_gap,1),
            'largeDirectionSteps': spikes, 'status': 'observation' if usable else 'withheld',
            'shaftRotation2dDeg': round(sum(angular_steps),1) if usable else None,
            'gripShiftImageHeightPct': [round((last['handle']['x']-first['handle']['x'])*width/height*100,2),round((last['handle']['y']-first['handle']['y'])*100,2)] if usable and first and last else None,
        })
    anchor_times = [('takeaway_candidate',takeaway),('grip_height_anchor',top),('after_anchor_100ms',min(top+100,impact_ms)),('mid_downswing_candidate',(top+impact_ms)/2),('impact_candidate',impact_ms)]
    anchors = []
    for label,t in anchor_times:
        row = nearest(pairs,t,35)
        anchor = {'label':label,'requestedTimeMs':t,'observation':row}
        pose = nearest(overlay.get('poseFrames',[]),row['timeMs'],25) if row else None
        if pose and row:
            keys=pose.get('keypoints',{})
            elbow,wrist=keys.get('left_elbow'),keys.get('left_wrist')
            valid=lambda p:isinstance(p,list) and len(p)>=3 and all(finite(v) for v in p[:3]) and p[2]>=.5
            if valid(elbow) and valid(wrist):
                arm=((elbow[0]-wrist[0])*width,(elbow[1]-wrist[1])*height)
                club=((row['head']['x']-row['handle']['x'])*width,(row['head']['y']-row['handle']['y'])*height)
                length=math.hypot(*arm)*math.hypot(*club)
                if length>1:
                    anchor['leftForearmClubProjectedAngleDeg']=round(math.degrees(math.acos(max(-1,min(1,sum(a*b for a,b in zip(arm,club))/length)))),1)
        anchors.append(anchor)
    return {'jobId':analysis.get('jobId'),'status':'research_observation','automaticCoachingReady':False,'imageSize':{'width':width,'height':height},
            'boundarySource':{'takeaway':'legacy_candidate','top':'smoothed_grip_height_candidate','impact':'pose_tcn_candidate'},
            'coordinateConvention':'image pixels; shaft direction grip to head, counterclockwise from image right',
            'limits':['2D projection, not 3D shaft plane or anatomical wrist angle','candidate boundaries, not human ground truth','confidence/gap gates are engineering guards, not validated golf thresholds','missing metadata rows are unknown without the source frame timeline','no good/bad, casting or ball-flight diagnosis'],
            'phases':summaries,'anchors':anchors,'matchedGripHeight':matched,'matchedGripHeights':levels,
            'sensitivityScope':'deterministic endpoint-radius bounds including height-based frame reselection; phase boundaries, accepted detections, camera and timestamps fixed; not confidence intervals or measured accuracy','shortProjectedSegments':len(pairs)-len(stable_pairs)}

