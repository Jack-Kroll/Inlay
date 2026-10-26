"""Conservative evidence that one specific finger placement is visually stable.

Consistency is a gate, not a calibrated probability. A stable geometric bias
can still be wrong, which is why public-label precision is measured separately.
"""
from __future__ import annotations
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
import math
import numpy as np

from . import fusion
from .fretboard import board_transform
from .hands import Hand
from .tuning import STANDARD_TUNING, candidates


@dataclass(frozen=True)
class VisualPolicy:
    window: float = .08
    min_frames: int = 3
    min_contact: float = .75
    min_margin: float = .08
    max_fret_motion: float = .35
    max_string_motion: float = .35
    agreement: float = .8
    max_residual: float = .045
    max_edge_residual: float = .01
    min_anchors: int = 6


class VisualContext:
    def __init__(self, samples, policy=VisualPolicy()):
        self.samples = samples
        self.times = [s['timestamp'] for s in samples]
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError('Visual sample timestamps must strictly increase')
        if (not np.isfinite(policy.window) or policy.window <= 0 or policy.min_frames < 3
                or not .5 < policy.agreement <= 1
                or not 0 < policy.min_contact <= 1 or not 0 < policy.min_margin <= 1
                or not all(np.isfinite(v) and v > 0 for v in
                           (policy.max_fret_motion, policy.max_string_motion,
                            policy.max_residual, policy.max_edge_residual))):
            raise ValueError('Invalid visual consistency window/frame count/agreement')
        self.policy = policy

    @classmethod
    def from_records(cls, records, strings=6, inset=.09, flipped=False, max_fret=22,
                     policy=VisualPolicy()):
        samples = []
        for record in records:
            transform, contacts, quality = None, [], None
            observation = record.observation
            if observation is not None:
                transform = board_transform(observation, strings=strings, inset=inset,
                                            flipped=flipped)
            if transform is not None:
                quality = dict(residual=transform.residual, edge_residual=transform.edge_residual,
                               anchors=transform.anchors, state=record.state)
                if record.hand_points is not None:
                    contacts = fusion.finger_contacts(
                        Hand(record.hand_points, record.hand_label, 1.), transform, max_fret=max_fret)
            samples.append(dict(timestamp=record.timestamp, contacts=contacts, quality=quality))
        return cls(samples, policy)

    def placement(self, timestamp, pitch, tuning=STANDARD_TUNING, max_fret=22):
        policy = self.policy
        low = bisect_left(self.times, timestamp-policy.window)
        high = bisect_right(self.times, timestamp+policy.window)
        samples = self.samples[low:high]
        if len(samples) < policy.min_frames:
            return None
        options = candidates(pitch, tuning, max_fret)
        # A visible finger can hover while an equal-pitch open string rings.
        # Preserve the existing open-string policy until press state is known.
        if any(fret == 0 for _, fret in options):
            return None
        votes = []
        for sample in samples:
            quality = sample['quality']
            if (quality is None or quality['state'] != 'detected'
                    or quality['anchors'] < policy.min_anchors
                    or not np.isfinite(quality['residual'])
                    or not np.isfinite(quality['edge_residual'])
                    or quality['residual'] > policy.max_residual
                    or quality['edge_residual'] > policy.max_edge_residual):
                continue
            contacts = sample['contacts']
            position = fusion.hand_position(contacts)
            scored = sorted([(fusion._score(s, f, contacts, position)[0], s, f)
                             for s, f in options], reverse=True)
            if not scored:
                continue
            geometry, string, fret = scored[0]
            margin = geometry - scored[1][0] if len(scored) > 1 else geometry
            if fret == 0 or margin < policy.min_margin:
                continue
            usable = [c for c in contacts if c.on_board]
            if not usable:
                continue
            contact = max(usable, key=lambda c: fusion._string_term(c, string)*fusion._fret_term(c, fret))
            support = fusion._string_term(contact, string)*fusion._fret_term(contact, fret)
            if support < policy.min_contact:
                continue
            votes.append(dict(key=(string, fret, contact.finger), timestamp=sample['timestamp'],
                              support=support, margin=margin, fret_float=contact.fret_float,
                              string_float=contact.string_float))
        if not votes:
            return None
        winner, count = Counter(v['key'] for v in votes).most_common(1)[0]
        if count < max(policy.min_frames, math.ceil(policy.agreement*len(samples))):
            return None
        same = [v for v in votes if v['key'] == winner]
        first, last = min(v['timestamp'] for v in same), max(v['timestamp'] for v in same)
        # Offline lookahead is allowed, but a one-sided settled hand is not
        # evidence that it had already arrived at the actual attack.
        if not first < timestamp < last:
            return None
        fret_motion = max(v['fret_float'] for v in same)-min(v['fret_float'] for v in same)
        string_motion = max(v['string_float'] for v in same)-min(v['string_float'] for v in same)
        if fret_motion > policy.max_fret_motion or string_motion > policy.max_string_motion:
            return None
        return dict(string=winner[0], fret=winner[1], finger=winner[2], frames=count,
                    total_frames=len(samples), first=first, last=last,
                    min_contact=float(min(v['support'] for v in same)),
                    min_margin=float(min(v['margin'] for v in same)),
                    fret_motion=float(fret_motion), string_motion=float(string_motion))
