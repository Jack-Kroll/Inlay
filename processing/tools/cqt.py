"""Cache a CQT per excerpt on exactly Basic Pitch's time and pitch grid.

The posteriorgram already carries some string information -- the verifier's
string head reaches .70 on held-out players from it alone -- but Basic Pitch
threw most of the timbre away: its heads describe pitch, not tone. Two strings
sounding the same pitch differ in brightness and inharmonicity, which lives in
the spectrum. This caches that spectrum on the same grid as the posteriorgram
so it can be stacked as extra patch channels.

Three bins per semitone from A0, matching the contour head's layout, so the
same patch geometry works for both.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from processing.music import verify
from processing.tools import guitarset



def one(job):
    name, audio, destination = job
    if destination.is_file():
        return name, 'cached'
    # The runtime computes this the same way; keep one implementation so a
    # cached CQT and a freshly computed one cannot disagree.
    spectrum = verify.spectrum(audio)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, cqt=spectrum.astype(np.float16))
    return name, f'{spectrum.shape[0]} frames'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('data/guitarset'))
    ap.add_argument('--output', type=Path, default=Path('data/guitarset/cqt'))
    ap.add_argument('--workers', type=int, default=5)
    args = ap.parse_args()
    found, _ = guitarset.excerpts(args.root)
    jobs = [(e.name, e.audio, args.output/f'{e.name}.npz') for e in sorted(found, key=lambda e: e.name)]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, note in pool.map(one, jobs, chunksize=1):
            done += 1
            if done % 30 == 0:
                print(f'{done}/{len(jobs)} {name} {note}', flush=True)
    print('done')


if __name__ == '__main__':
    main()
