"""Marker-independent dense geometry and motion-validated short-occlusion tracking."""
from dataclasses import dataclass, field
from itertools import combinations
import cv2
import numpy as np


@dataclass
class EstimatedFret:
    line: np.ndarray
    number: int
    method: str = 'spacing'


@dataclass
class Observation:
    neck: np.ndarray
    frets: list[np.ndarray]
    nut: np.ndarray | None
    confidence: float
    numbers: list[int | None] = field(default_factory=list)
    numbering: str = 'unknown'
    estimates: list[EstimatedFret] = field(default_factory=list)


def decode_maps(maps, threshold=.5):
    if maps.ndim != 3 or maps.shape[2] != 3 or not np.isfinite(maps).all():
        raise ValueError('Expected finite HxWx3 probability maps')
    binary = (maps[...,0] > threshold).astype(np.uint8)
    contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    contour = max(contours,key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < max(64, maps.shape[0]*maps.shape[1]*.001): return None
    neck = cv2.convexHull(contour).reshape(-1,2).astype(np.float32)
    rect = cv2.minAreaRect(neck)
    box = cv2.boxPoints(rect)
    edges = np.roll(box,-1,axis=0)-box
    axis = edges[np.argmax(np.linalg.norm(edges,axis=1))]
    axis /= np.linalg.norm(axis)
    if axis[np.argmax(np.abs(axis))]<0: axis=-axis
    width = min(rect[1]);length = max(rect[1])
    if width<4 or length<2*width: return None
    mask=np.zeros(binary.shape,np.uint8);cv2.fillConvexPoly(mask,neck.astype(np.int32),1)
    dilated=cv2.dilate(mask,np.ones((5,5),np.uint8))

    def lines(channel):
        evidence=((maps[...,channel]>threshold)&(dilated>0)).astype(np.uint8)*255
        # Constrain the Hough NORMAL to the board axis. An unconstrained
        # probabilistic Hough transform consumes dense wire pixels as spurious
        # longitudinal lines before it finds the actual transverse wires.
        angle=float(np.arctan2(axis[1],axis[0]) % np.pi)
        half=np.deg2rad(30)
        ranges=[]
        lo,hi=angle-half,angle+half
        if lo<0:ranges=[(0,hi),(np.pi+lo,np.pi)]
        elif hi>np.pi:ranges=[(lo,np.pi),(0,hi-np.pi)]
        else:ranges=[(lo,hi)]
        raw=[]
        for lo,hi in ranges:
            found=cv2.HoughLines(evidence,1,np.pi/720,max(5,int(width*.22)),min_theta=lo,max_theta=hi)
            if found is not None:raw.extend(found[:250,0])
        candidates=[]
        if not raw:
            return []
        # Score all Hough proposals together; avoids hundreds of Python loops
        # and repeated tiny allocations on every frame.
        proposals=np.asarray(raw)
        normals=np.stack((np.cos(proposals[:,1]),np.sin(proposals[:,1])),axis=1)
        directions=np.stack((-normals[:,1],normals[:,0]),axis=1)
        center=neck.mean(0).astype(float)
        centers=center+(proposals[:,0]-normals@center)[:,None]*normals
        offsets=np.linspace(-2*width,2*width,max(20,int(8*width)))
        samples=centers[:,None,:]+offsets[None,:,None]*directions[:,None,:]
        xy=np.rint(samples).astype(int)
        good=(xy[...,0]>=0)&(xy[...,0]<mask.shape[1])&(xy[...,1]>=0)&(xy[...,1]<mask.shape[0])
        x=np.clip(xy[...,0],0,mask.shape[1]-1)
        y=np.clip(xy[...,1],0,mask.shape[0]-1)
        inside=good&(mask[y,x]>0)
        counts=inside.sum(axis=1)
        support=maps[y,x,channel]
        coverage=((support>threshold)&inside).sum(axis=1)/np.maximum(counts,1)
        eligible=np.flatnonzero((counts>=max(4,width*.7))&(coverage>=.45))
        if len(eligible):
            first=inside[eligible].argmax(axis=1)
            last=inside.shape[1]-1-inside[eligible,::-1].argmax(axis=1)
            segments=np.stack((samples[eligible,first],samples[eligible,last]),axis=1).astype(np.float32)
            confidence=(support*inside).sum(axis=1)/np.maximum(counts,1)
            positions=segments.mean(axis=1)@axis
            candidates=[(float(confidence[k]),float(pos),segment)
                        for k,pos,segment in zip(eligible,positions,segments)]
        selected=[]
        for candidate in sorted(candidates,key=lambda item:-item[0]):
            if all(abs(candidate[1]-other[1])>max(2,width*.035) for other in selected):
                selected.append(candidate)
        return sorted(selected,key=lambda item:item[1])
    frets=lines(1);nuts=lines(2)
    # A nut must be near an end of the board; this reduces capo/internal-fret confusion.
    projections=neck@axis;lo,hi=projections.min(),projections.max()
    nuts=[x for x in nuts if min(abs(x[1]-lo),abs(x[1]-hi))<length*.15]
    nut=max(nuts,key=lambda item:item[0])[2] if nuts else None
    fret_segments=[x[2] for x in frets if nut is None or abs(x[1]-nut.mean(axis=0)@axis)>max(3,width*.12)]
    confidence=float(maps[...,0][mask>0].mean())
    if not fret_segments: return None
    observation=Observation(neck,fret_segments,nut,confidence)
    assign_numbers(observation)
    return observation


def lattice_fit(distances,max_fret=24):
    """Nut-anchored projective equal-temperament fit; missing frets are allowed.

    Return None when the best different assignment is too close. No nut-width
    scale prior and no assumption that detections form consecutive fret numbers.
    """
    s=np.asarray(distances,float)
    if len(s)<4 or not np.isfinite(s).all() or np.any(s<=0): return None
    selected=np.unique(np.linspace(0,len(s)-1,min(5,len(s))).astype(int))
    candidates={}
    n1,n2=np.array(list(combinations(range(1,max_fret+1),2))).T
    r1,r2=1-2.**(-n1/12),1-2.**(-n2/12)
    for i,j in combinations(selected,2):
        if s[j]-s[i]<1e-6: continue
        c=(s[j]/r2-s[i]/r1)/(s[i]-s[j]);a=s[i]/r1+s[i]*c
        good=(a>0)&(1+c*(1-2.**(-max_fret/12))>.05)
        a,c=a[good],c[good]
        if not len(a): continue
        denom=a[:,None]-c[:,None]*s
        r=np.divide(s,denom,out=np.full_like(denom,np.nan),where=denom>0)
        with np.errstate(invalid='ignore',divide='ignore'):
            floating=-12*np.log2(1-r)
        integer=np.rint(np.nan_to_num(floating,nan=-100,posinf=-100,neginf=-100)).astype(int)
        error=np.abs(floating-integer)
        inliers=(integer>=1)&(integer<=max_fret)&(error<.18)
        # Batch-reject weak candidates before the per-assignment checks.
        counts=inliers.sum(axis=1)
        eligible=np.flatnonzero(counts>=max(4,int(np.ceil(.65*len(s)))))
        for k in eligible:
            valid=inliers[k]
            if len(set(integer[k,valid]))!=int(valid.sum()): continue
            assignment=tuple(int(n) if ok else 0 for n,ok in zip(integer[k],valid))
            score=float(valid.sum()-.5*np.mean(error[k,valid]))
            if assignment not in candidates or score>candidates[assignment][0]:
                candidates[assignment]=(score,float(a[k]),float(c[k]))
    ranked=sorted(candidates.items(),key=lambda item:-item[1][0])
    if not ranked: return None
    if len(ranked)>1 and ranked[0][1][0]-ranked[1][1][0]<.025: return None
    assignment,(_,a,c)=ranked[0]
    return {'numbers':[n or None for n in assignment],'a':a,'c':c}


def assign_numbers(observation):
    observation.numbers=[None]*len(observation.frets)
    observation.numbering='unknown'
    observation.estimates=[]
    if observation.nut is None or len(observation.frets)<4: return
    centers=np.array([f.mean(axis=0) for f in observation.frets])
    origin=observation.nut.mean(axis=0)
    _,_,vectors=np.linalg.svd(centers-origin,full_matrices=False)
    axis=vectors[0]
    if np.median((centers-origin)@axis)<0: axis=-axis
    distances=(centers-origin)@axis
    order=np.argsort(distances)
    fit=lattice_fit(distances[order])
    if fit is None: return
    for i,n in zip(order,fit['numbers']): observation.numbers[i]=n
    observation.numbering='nut-anchored fit'
    estimate_missing_frets(observation, fit, origin, axis)



def estimate_missing_frets(observation, fit, origin, axis, max_missing=4):
    """Interpolate bounded gaps using an accepted nut-anchored spacing fit.

    Estimated lines are kept outside frets/numbers so they can never become
    measurements or reinforce the next frame's numbering fit.
    """
    observation.estimates = []
    if observation.nut is None or observation.numbering != 'nut-anchored fit':
        return
    measured = [(n, line) for n, line in zip(observation.numbers, observation.frets)
                if n is not None]
    if len(measured) < 4 or len({n for n, _ in measured}) != len(measured):
        return
    anchors = [(0, observation.nut)] + sorted(measured, key=lambda item: item[0])

    def position(n):
        ratio = 1-2.**(-n/12)
        denominator = 1+fit['c']*ratio
        return fit['a']*ratio/denominator if denominator > .05 else float('nan')

    for (low, left), (high, right) in zip(anchors, anchors[1:]):
        if not 1 <= high-low-1 <= max_missing:
            continue
        sl, sh = position(low), position(high)
        if not np.isfinite([sl, sh]).all() or sh-sl < 2*(high-low):
            continue
        # Endpoints have no string identity; align their order before interpolation.
        left, right = np.asarray(left, float), np.asarray(right, float)
        if np.linalg.norm(left-right[::-1]) < np.linalg.norm(left-right):
            right = right[::-1]
        if (right.mean(0)-left.mean(0))@axis <= 0:
            continue
        spacing = min(position(low+1)-sl, sh-position(high-1))
        residual = max(abs((left.mean(0)-origin)@axis-sl),
                       abs((right.mean(0)-origin)@axis-sh))
        if spacing <= 0 or residual > .2*spacing:
            continue
        for number in range(low+1, high):
            weight = (position(number)-sl)/(sh-sl)
            if not 0 < weight < 1:
                continue
            line = ((1-weight)*left+weight*right).astype(np.float32)
            if not np.isfinite(line).all() or np.linalg.norm(line[1]-line[0]) < 4:
                continue
            samples = line[0]+np.linspace(0, 1, 5)[:, None]*(line[1]-line[0])
            if any(cv2.pointPolygonTest(observation.neck.astype(np.float32),
                                       (float(p[0]), float(p[1])), True) < -2 for p in samples):
                continue
            # A real but unnumbered detection may already occupy this slot.
            center = line.mean(0)
            if any(np.linalg.norm(center-f.mean(0)) < max(2, .25*spacing)
                   for f in observation.frets):
                continue
            observation.estimates.append(EstimatedFret(line, number))


def retain_missing_frets(detected, warped, max_missing=4):
    """Bridge briefly missing measurements using already-validated camera motion."""
    numbered = [(n, line) for n, line in zip(detected.numbers, detected.frets)
                if n is not None]
    if len(numbered) < 2:
        return
    low = min(n for n, _ in numbered)
    if warped.nut is not None:
        low = 0
    present = {n for n, _ in numbered}
    boundaries = sorted(present | {low})
    permitted = {n for left, right in zip(boundaries, boundaries[1:])
                 if right-left-1 <= max_missing for n in range(left+1, right)}
    candidates = [EstimatedFret(line, n, 'tracked') for n, line in
                  zip(warped.numbers, warped.frets) if n is not None]
    candidates += [EstimatedFret(e.line, e.number, 'tracked') for e in warped.estimates]
    width = min(cv2.minAreaRect(detected.neck)[1])
    for estimate in candidates:
        if estimate.number not in permitted or estimate.number in present:
            continue
        if any(np.linalg.norm(estimate.line.mean(0)-f.mean(0)) < max(2, width*.15)
               for f in detected.frets):
            continue
        if any(cv2.pointPolygonTest(detected.neck.astype(np.float32),
                                   (float(p[0]), float(p[1])), True) < -2
               for p in estimate.line):
            continue
        detected.estimates.append(estimate)
        present.add(estimate.number)


def transform_observation(observation,matrix):
    def transform(points):
        return cv2.perspectiveTransform(np.asarray(points,np.float32).reshape(-1,1,2),matrix).reshape(-1,2)
    return Observation(transform(observation.neck),[transform(f) for f in observation.frets],
                       transform(observation.nut) if observation.nut is not None else None,
                       observation.confidence,list(observation.numbers),observation.numbering,
                       [EstimatedFret(transform(e.line), e.number, e.method) for e in observation.estimates])


class FretboardTracker:
    def __init__(self,max_gap=.5,smoothing=.06):
        if max_gap<=0 or smoothing<0: raise ValueError('Invalid tracker timing')
        self.max_gap,self.smoothing=max_gap,smoothing
        self.reset()

    def reset(self):
        self.previous_gray=None;self.observation=None;self.last_time=None;self.last_detection=None
        self.state='lost';self.flow_inliers=0;self.last_board_detection=None

    def _motion(self,gray):
        mask=np.zeros_like(self.previous_gray)
        cv2.fillConvexPoly(mask,self.observation.neck.astype(np.int32),255)
        points=cv2.goodFeaturesToTrack(self.previous_gray,150,.01,5,mask=mask)
        if points is None or len(points)<8:return None
        current,status,_=cv2.calcOpticalFlowPyrLK(self.previous_gray,gray,points,None)
        if current is None:return None
        back,status_back,_=cv2.calcOpticalFlowPyrLK(gray,self.previous_gray,current,None)
        if back is None:return None
        good=(status.ravel()>0)&(status_back.ravel()>0)&(np.linalg.norm(points-back,axis=2).ravel()<1.5)
        p,q=points[good],current[good]
        if len(p)<8:return None
        diag=np.hypot(*gray.shape)
        matrix,inliers=cv2.findHomography(p,q,cv2.RANSAC,max(2,diag*.002))
        if matrix is None or inliers is None:return None
        inside=inliers.ravel()>0;self.flow_inliers=int(inside.sum())
        if self.flow_inliers<8 or inside.mean()<.6:return None
        # Reject points confined to a hand or one tiny portion of the board.
        source=p[inside].reshape(-1,2)
        source_area=cv2.contourArea(cv2.convexHull(source))
        board_area=cv2.contourArea(self.observation.neck)
        if source_area<board_area*.15:return None
        warped=transform_observation(self.observation,matrix)
        new_area=cv2.contourArea(warped.neck)
        if not np.isfinite(warped.neck).all() or not .5<new_area/max(board_area,1)<2:return None
        if np.linalg.norm(warped.neck.mean(0)-self.observation.neck.mean(0))>diag*.25:return None
        return warped

    def update(self,frame,detected,timestamp):
        if not np.isfinite(timestamp):raise ValueError('Timestamp must be finite')
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if self.last_time is not None and (timestamp<=self.last_time or timestamp-self.last_time>self.max_gap or gray.shape!=self.previous_gray.shape):
            self.reset()
        warped=None;self.flow_inliers=0
        if self.observation is not None and self.previous_gray is not None:
            warped=self._motion(gray)
        if detected is not None:
            if warped is not None:
                # Smooth only aligned individual measurements, never mixed stale rails.
                dt=timestamp-self.last_time
                alpha=1-np.exp(-dt/max(self.smoothing,1e-6))
                width=min(cv2.minAreaRect(detected.neck)[1])
                used=set()
                for i,line in enumerate(detected.frets):
                    if not warped.frets:break
                    distances=[np.linalg.norm(line.mean(0)-f.mean(0)) if j not in used else np.inf for j,f in enumerate(warped.frets)]
                    j=int(np.argmin(distances))
                    if distances[j]<max(2,width*.15):
                        used.add(j);old=warped.frets[j]
                        if np.linalg.norm(line-old[::-1])<np.linalg.norm(line-old):old=old[::-1]
                        detected.frets[i]=(alpha*line+(1-alpha)*old).astype(np.float32)
                # Numbered history is retained only for a bounded gap after a nut fit.
                if detected.numbering=='unknown' and self.last_detection is not None and timestamp-self.last_detection<self.max_gap:
                    numbered_used=set()
                    for i,line in enumerate(detected.frets):
                        if not warped.frets:break
                        distances=[np.linalg.norm(line.mean(0)-f.mean(0)) if j not in numbered_used else np.inf for j,f in enumerate(warped.frets)]
                        j=int(np.argmin(distances))
                        if distances[j]<max(2,width*.15):
                            detected.numbers[i]=warped.numbers[j];numbered_used.add(j)
                    if any(n is not None for n in detected.numbers):
                        detected.numbering='short-term tracked'
                        retain_missing_frets(detected, warped)
            self.observation=detected
            if detected.numbering=='nut-anchored fit':self.last_detection=timestamp
            self.state='detected'
            self.last_board_detection=timestamp
        elif warped is not None and self.last_board_detection is not None and timestamp-self.last_board_detection<=self.max_gap:
            self.observation=warped;self.state='tracked'
            for estimate in self.observation.estimates:
                estimate.method='tracked'
            self.observation.confidence*=np.exp(-(timestamp-self.last_time)/self.max_gap)
        else:
            self.observation=None;self.state='lost';self.last_detection=None
        self.previous_gray=gray;self.last_time=timestamp
        return self.observation



