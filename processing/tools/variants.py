"""Render a GuitarSet domain into the verifier's caches.

    uv run python -m processing.tools.variants --variants 3
    uv run python -m processing.tools.variants --source pickup --variants 0
    uv run python -m processing.tools.variants --source hex --remixes 3
    uv run python -m processing.tools.variants --shift 3 8 --variants 2

A *domain* is one recording chain over the same performances: GuitarSet's
microphone as recorded (already cached), its pickup mix of the same takes, a
fresh mixture of its debleeded hex pickup's six per-string channels, or any of
those put through ``processing.training.augment``'s synthesised amp and room.
Each lands in a directory of its own inside ``posteriorgrams/`` and ``cqt/``.

The verifier reads a Basic Pitch posteriorgram and a CQT, both cached per
excerpt, so augmenting its *audio* means re-rendering both. That sounds
expensive and is not: Basic Pitch's Core ML graph runs a thirty-second excerpt
in 0.08 s, two orders of magnitude under the 2.5 s the CQT costs. Rendering a
domain therefore costs about what the CQT cache already cost, which settles
the one design question here -- augmenting the CQT channels alone and leaving
the posteriorgram channels clean would have saved almost nothing and left
five of the model's eight input channels un-augmented.

``--shift`` additionally moves each excerpt up the neck by resampling it,
which changes the pitch and the timing of the labels as well as the audio. The
shift that was actually applied is stored beside the posteriorgram, so
``verifier_data`` reads it back rather than recomputing a random draw and
hoping the two agree.

Augmented domains are rendered for the training players only. The pickup
domain is rendered for every player, because a held-out player's pickup take
is the cross-domain number that says whether augmentation worked without
spending IDMT-SMT-GUITAR to find out.
"""
from __future__ import annotations
import argparse, contextlib, io, json, tempfile, zlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from processing.music import verify
from processing.tools import guitarset
from processing.training import augment, verifier_data

ROOT = Path('data/guitarset')
KEYS = ('note', 'onset', 'contour')
# GuitarSet ships each take through two transducers. The pickup mix is a
# sum of the hex pickup's six strings: a different instrument voice over the
# same notes, which is exactly the invariance the string head is missing.
SOURCES = {'mic': ('audio_mono-mic', '_mic'),
           'pickup': ('audio_mono-pickup', '_mix'),
           'hex': ('audio_hex-pickup_debleeded', '_hex_cln')}
_MODEL = []


def _model():
    """Basic Pitch, loaded once per worker process rather than once per job."""
    if not _MODEL:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model
        _MODEL.append(Model(ICASSP_2022_MODEL_PATH))
    return _MODEL[0]


@dataclass(frozen=True)
class Job:
    """One excerpt to render into one domain.

    Everything the worker needs travels in here rather than in a module
    global: macOS spawns worker processes, which re-import this module and
    would not see a global the parent had changed.
    """
    domain: str
    name: str
    audio: Path
    root: Path
    augmented: bool = False
    remix: bool = False
    strength: float = 1.
    shift: tuple[int, int] | None = None
    headroom: int = 0


def seed(domain, name):
    """A stable seed per (domain, excerpt).

    Python salts ``hash`` of a string per process, so a re-render would
    otherwise draw a different chain and silently disagree with the cache it
    was adding to.
    """
    return zlib.crc32(f'{domain}/{name}'.encode())


def _render(job):
    """Render one excerpt into one domain: posteriorgram and CQT."""
    stem = verifier_data.variant_name(job.domain, job.name)
    posteriorgram = job.root/'posteriorgrams'/f'{stem}.npz'
    spectrum = job.root/'cqt'/f'{stem}.npz'
    if posteriorgram.is_file() and spectrum.is_file():
        return job.name, 'cached'

    import librosa, soundfile
    from basic_pitch.inference import run_inference
    # Seeded by excerpt and domain, so a re-render reproduces this chain and
    # two domains of one excerpt never draw the same one.
    rng = np.random.default_rng(seed(job.domain, job.name))
    done = []
    if job.remix:
        # Six channels at 44.1 kHz, one string each, resampled together.
        stems = librosa.load(str(job.audio), sr=augment.SAMPLE_RATE, mono=False)[0]
        wave, note = augment.remix([np.asarray(stem, np.float32) for stem in stems], rng)
        done.append(note)
    else:
        wave = librosa.load(str(job.audio), sr=augment.SAMPLE_RATE, mono=True)[0]
        wave = np.asarray(wave, np.float32)
        done.append('as recorded')
    semitones = 0
    if job.shift:
        # Clamped to the neck: an excerpt that already reaches fret 20 can
        # only move two frets, whatever was asked for.
        low, high = job.shift
        semitones = int(min(rng.integers(low, high + 1), job.headroom))
        wave = augment.neck_shift(wave, semitones)
        done.append(f'up {semitones} semitones')
    if job.augmented:
        wave, note = augment.chain(wave, rng, strength=job.strength)
        done.append(note)
    description = '; '.join(done)

    with tempfile.TemporaryDirectory() as directory:
        # Basic Pitch resolves its input with librosa.load(str(path)), so the
        # augmented audio has to exist as a file for the length of the call.
        path = Path(directory)/'augmented.wav'
        soundfile.write(str(path), wave, augment.SAMPLE_RATE)
        # The Core ML path prints tensor diagnostics straight to stdout.
        with contextlib.redirect_stdout(io.StringIO()):
            output = run_inference(str(path), _model())
    posteriorgram.parent.mkdir(parents=True, exist_ok=True)
    # The shift travels with the posteriorgram: it changes what the labels are,
    # not just what the audio sounds like, and a second random draw elsewhere
    # would silently disagree with this one.
    np.savez_compressed(posteriorgram, shift=np.int8(semitones),
                        **{key: output[key].astype(np.float16) for key in KEYS})
    spectrum.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(spectrum,
                        cqt=verify.spectrum(wave).astype(np.float16))
    return job.name, description


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source', choices=sorted(SOURCES), default='mic',
                        help='Which transducer to render from')
    parser.add_argument('--variants', type=int, default=3,
                        help='Augmented copies to render; 0 renders the source itself')
    parser.add_argument('--remixes', type=int, default=0,
                        help='Hex only: fresh per-string mixtures to render, each '
                             'then put through the same amp and room as a variant')
    parser.add_argument('--players', nargs='*',
                        help='Which players to render; training players by default, '
                             'all six when --variants is 0')
    parser.add_argument('--shift', type=int, nargs=2, metavar=('LOW', 'HIGH'),
                        help='Also move each excerpt this many semitones up the '
                             'neck, drawn per excerpt and clamped to the neck')
    parser.add_argument('--strength', type=float, default=1.,
                        help='Scale the chain: decibels, drive and how often '
                             'each stage fires')
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()

    # An un-augmented domain is something to measure on as well as train on,
    # so it is rendered for the held-out players too.
    players = args.players or (None if args.variants == 0 and not args.remixes
                               else list(verifier_data.TRAIN_PLAYERS))
    directory, suffix = SOURCES[args.source]
    found, _ = guitarset.excerpts(args.root, directory, suffix)
    found = [excerpt for excerpt in sorted(found, key=lambda excerpt: excerpt.name)
             if players is None or excerpt.player in players]
    prefix = '' if args.source == 'mic' else f'{args.source}-'
    remix = args.source == 'hex'
    if args.remixes:
        if not remix:
            raise SystemExit('--remixes needs --source hex: only the hex pickup '
                             'records a string on its own channel')
        # Each remix also gets a recording chain: the hex pickup's own voice
        # is a single distinctive one, and the measurement that matters is
        # what exact string labels add on top of augmentation, not instead.
        domains = [f'hex-r{number}' for number in range(1, args.remixes + 1)]
        augmented = True
    elif args.variants == 0:
        domains = [args.source]
        augmented = False
    else:
        # A harder chain gets a name of its own, so two strengths can sit in
        # the caches at once and be trained against each other.
        label = 'aug' if args.strength == 1 else f'aug{args.strength:g}'
        if args.shift:
            label = f'shift{args.shift[0]}-{args.shift[1]}'
        domains = [f'{prefix}{label}-v{number}'
                   for number in range(1, args.variants + 1)]
        augmented = True
    headroom = {}
    if args.shift:
        # Read once per excerpt rather than once per render: the annotations
        # are 1.5 MB of JSON each.
        headroom = {excerpt.name: augment.shift_headroom(excerpt.notes())
                    for excerpt in found}
    shift = tuple(args.shift) if args.shift else None
    jobs = [Job(domain=domain, name=excerpt.name, audio=excerpt.audio,
                root=args.root, augmented=augmented, remix=remix,
                strength=args.strength, shift=shift,
                headroom=headroom.get(excerpt.name, 0))
            for domain in domains for excerpt in found]
    print(f'{len(found)} {args.source} excerpts x {len(domains)} '
          f'domain(s) {" ".join(domains)} = {len(jobs)} renders')

    rendered = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (name, description) in enumerate(pool.map(_render, jobs, chunksize=1), 1):
            rendered.append(description)
            if done % 60 == 0:
                print(f'{done}/{len(jobs)} {name}: {description}', flush=True)
    # What each render actually did, so a result can be read back against the
    # chains that produced it rather than the ranges they were drawn from.
    path = args.root/'posteriorgrams'/f'{domains[0]}.manifest.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({verifier_data.variant_name(job.domain, job.name):
                                description
                                for job, description in zip(jobs, rendered)}, indent=2))
    print(f'done; wrote {path}')


if __name__ == '__main__':
    main()
