"""Render every posteriorgram the verifier reads, from a different front end.

    uv run python -m processing.tools.frontend_caches --front-end runs/frontend/ft1/best.pt

The verifier's caches were all rendered by Basic Pitch's Core ML model:
GuitarSet's microphone and pickup for every player, the three augmented
renders of the training players, Guitar-TECHS's four chains, the re-amped
held-out player, and IDMT-SMT-GUITAR. A verifier for another front end needs
every one of them again, from that front end, and nothing else changes: the
audio of each domain is rebuilt from the same seeds
(``variants.render_wave``), and the CQT caches, which depend only on audio,
are reused as they are.

Workers decode and augment audio; the front end runs in this process, on
the GPU, one excerpt at a time. Output goes to ``posteriorgrams@<tag>`` beside
each existing cache (``frontend.cache_dir``), never over it.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import torch

from processing.music import frontend
from processing.tools import egfxset, guitarset, guitar_techs, idmt, variants
from processing.training import verifier_data

GUITARSET = Path('data/guitarset')
IDMT_ROOT = Path('data/idmt-smt-guitar')
AUGMENTED = ('aug-v1', 'aug-v2', 'aug-v3')
CHAINS = ('di', 'amp', 'ego', 'exo')


@dataclass(frozen=True)
class Render:
    destination: Path
    audio: Path
    domain: str = 'mic'
    name: str = ''
    augmented: bool = False
    amped: bool = False
    # A CQT to write as well, for a domain that has none yet.
    spectrum: Path | None = None


def jobs(front_end, sets):
    out = []
    root = frontend.cache_dir(GUITARSET, front_end)
    if 'guitarset' in sets:
        found, _ = guitarset.excerpts(GUITARSET)
        pickups, _ = guitarset.excerpts(GUITARSET, 'audio_mono-pickup', '_mix')
        for excerpt in found:
            out.append(Render(root/f'{excerpt.name}.npz', excerpt.audio))
            if excerpt.player in verifier_data.TRAIN_PLAYERS:
                out += [Render(root/domain/f'{excerpt.name}.npz', excerpt.audio, domain,
                               excerpt.name, augmented=True) for domain in AUGMENTED]
        out += [Render(root/'pickup'/f'{excerpt.name}.npz', excerpt.audio)
                for excerpt in pickups]
    if 'techs' in sets:
        for excerpt in guitar_techs.excerpts():
            domain = guitar_techs.domain(excerpt.key)
            if excerpt.key in CHAINS:
                out.append(Render(root/domain/f'{excerpt.name}.npz', excerpt.audio))
            elif excerpt.key == 'drive-v1' and excerpt.player == 'P3':
                # The re-amped held-out player: the distortion diagnostic.
                out.append(Render(root/domain/f'{excerpt.name}.npz', excerpt.audio,
                                  domain, excerpt.name, amped=True))
    if 'egfx' in sets:
        for excerpt in egfxset.excerpts():
            stem = f'{egfxset.DOMAIN}/{excerpt.name}.npz'
            out.append(Render(root/stem, excerpt.audio, spectrum=GUITARSET/'cqt'/stem))
    if 'idmt' in sets:
        target = frontend.cache_dir(IDMT_ROOT, front_end)
        recordings, _ = idmt.recordings(idmt.ROOT)
        out += [Render(target/f'{idmt.key(recording)}.npz', recording.audio)
                for recording in recordings]
    return out


def _wave(job):
    if job.augmented or job.amped:
        wave, _, _ = variants.render_wave(variants.Job(
            domain=job.domain, name=job.name, audio=job.audio, root=GUITARSET,
            augmented=job.augmented, amped=job.amped))
        return job, np.asarray(wave, np.float32)
    return job, frontend.load_audio(job.audio)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--front-end', type=Path, required=True)
    parser.add_argument('--sets', nargs='*', default=['guitarset', 'techs', 'egfx', 'idmt'])
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--device')
    args = parser.parse_args()

    device = torch.device(args.device or ('mps' if torch.backends.mps.is_available() else 'cpu'))
    model = frontend.load(args.front_end, device)
    todo = [job for job in jobs(args.front_end, args.sets)
            if not job.destination.is_file()
            or (job.spectrum is not None and not job.spectrum.is_file())]
    print(f'{len(todo)} posteriorgrams to render from {args.front_end}', flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number, (job, wave) in enumerate(pool.map(_wave, todo, chunksize=2), 1):
            heads = frontend.infer(model, wave, device)
            job.destination.parent.mkdir(parents=True, exist_ok=True)
            # Same layout as every Basic Pitch cache, shift included, so the
            # verifier's readers cannot tell the two apart.
            if job.spectrum is not None and not job.spectrum.is_file():
                from processing.music import verify
                job.spectrum.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(job.spectrum, cqt=verify.spectrum(wave).astype(np.float16))
            arrays = {key: heads[key].astype(np.float16) for key in heads}
            np.savez_compressed(job.destination, shift=np.int8(0), **arrays)  # type: ignore[arg-type]  # ** collides with savez's own allow_pickle keyword.
            if number % 200 == 0 or number == len(todo):
                print(f'  {number}/{len(todo)}  {time.time() - started:.0f}s', flush=True)


if __name__ == '__main__':
    main()
