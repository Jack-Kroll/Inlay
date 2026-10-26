"""The end-to-end test set: phone clips with reference tab, and the score.

    uv run python -m processing.tools.testset draft "clips/Blackbird.mov"
    uv run python -m processing.tools.testset score
    uv run python -m processing.tools.testset score --predictions runs/transcribe-final

Every other benchmark in this repository is audio against annotated audio. This
one runs the whole pipeline -- camera included -- on recordings like the ones
the app will see, and scores what comes out against tab a person checked.

A label is a text file per clip in ``data/testset`` (``<name>.tab.txt``):

    # Comments start with #.
    video: clips/Blackbird.mov    # relative to the repository, or absolute
    tuning: standard              # or drop-d, or six MIDI pitches / note names, high to low
    capo: 0
    @0.622 5:7                    # [@seconds] then string:fret per simultaneous note
    @1.380 5:7
    3:0 2:1                       # a chord: two notes at once
    B3                            # a note whose pitch is known and position is not

Strings are numbered as tab is read: 1 is the high E. Frets are counted from
the capo. When every line has a time, notes are matched by onset (50 ms, as
everywhere else here; ``--tolerance`` to widen it for hand-placed times).
When any line has none, the label is a sequence, and the transcription is
aligned to it in order -- which checks what was played and where, not when.

``draft`` writes a label from the pipeline's own output, marked
``status: draft``, so correcting it is faster than writing one. A draft is
the pipeline's answer, not the truth: ``score`` refuses drafts until the
status line is removed, after checking every note by ear and against the
video.

Scores are the ones the other benchmarks report: note F1 (pitch at the right
time, or the right place in the sequence), string accuracy over matched notes
with a known position, and tab F1 (pitch and string both right). Open-string
accuracy is reported apart, because an open string is the mistake a reader
notices first. See docs/testset.md.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any
import numpy as np

ROOT = Path('data/testset')
TOLERANCE = .05
VIDEO_SUFFIXES = ('.mov', '.mp4', '.m4v', '.MOV', '.MP4')
TUNINGS = {'standard': (64, 59, 55, 50, 45, 40), 'drop-d': (64, 59, 55, 50, 45, 38)}
NAMES = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
NOTE = re.compile(r'^([A-Ga-g])([#b]?)(-?\d)$')
POSITION = re.compile(r'^(\d+):(\d+)$')


def note_pitch(name):
    """'B3' -> 59, 'F#2' -> 42, 'Bb3' -> 58."""
    found = NOTE.match(name)
    if not found:
        raise ValueError(f'not a note name: {name!r}')
    letter, accidental, octave = found.groups()
    shift = {'#': 1, 'b': -1}.get(accidental, 0)
    return 12*(int(octave) + 1) + NAMES[letter.upper()] + shift


@dataclass(frozen=True)
class Note:
    pitch: int
    start: float | None = None
    string: int | None = None    # 1 = high E
    fret: int | None = None      # absolute, capo included


@dataclass
class Label:
    path: Path
    video: Path | None
    tuning: tuple[int, ...]
    capo: int = 0
    draft: bool = False
    events: list[list[Note]] = field(default_factory=list)

    @property
    def timed(self):
        return bool(self.events) and all(e[0].start is not None for e in self.events)

    @property
    def notes(self):
        return [note for event in self.events for note in event]


def parse_tuning(text):
    text = text.strip()
    if text.lower() in TUNINGS:
        return TUNINGS[text.lower()]
    parts = text.replace(',', ' ').split()
    pitches = tuple(int(p) if p.lstrip('-').isdigit() else note_pitch(p) for p in parts)
    if list(pitches) != sorted(pitches, reverse=True):
        raise ValueError('tuning must run from string 1 (highest) down')
    return pitches


def parse(path):
    """Read one label file."""
    path = Path(path)
    label = Label(path, None, TUNINGS['standard'])
    header = True
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        where = f'{path}:{number}'
        key, _, value = line.partition(':')
        if header and key.strip().lower() in ('video', 'tuning', 'capo', 'status') and value:
            key, value = key.strip().lower(), value.strip()
            if key == 'video':
                label.video = Path(value)
            elif key == 'tuning':
                label.tuning = parse_tuning(value)
            elif key == 'capo':
                label.capo = int(value)
            elif key == 'status':
                label.draft = value.lower() == 'draft'
            continue
        header = False
        tokens = line.split()
        start = None
        if tokens[0].startswith('@'):
            start = float(tokens.pop(0)[1:])
        if not tokens:
            raise ValueError(f'{where}: a time with no notes')
        event = []
        for token in tokens:
            position = POSITION.match(token)
            if position:
                string, fret = int(position.group(1)), int(position.group(2)) + label.capo
                if not 1 <= string <= len(label.tuning):
                    raise ValueError(f'{where}: no string {string} in a '
                                     f'{len(label.tuning)}-string tuning')
                if any(n.string == string for n in event):
                    raise ValueError(f'{where}: string {string} twice at once')
                event.append(Note(label.tuning[string - 1] + fret, start, string, fret))
            else:
                try:
                    event.append(Note(note_pitch(token), start))
                except ValueError:
                    raise ValueError(f'{where}: expected string:fret or a note name, '
                                     f'got {token!r}') from None
        label.events.append(event)
    if label.video is None:
        label.video = next((path.with_name(path.name.removesuffix('.tab.txt') + suffix)
                            for suffix in VIDEO_SUFFIXES
                            if path.with_name(path.name.removesuffix('.tab.txt') + suffix).is_file()),
                           None)
    return label


def write(path, notes, video, tuning, chord_window=.05, draft=True):
    """A label file from transcribed notes: ``(pitch, start, string, fret)``."""
    lines = ['# Written from the pipeline\'s own output. Check every note by ear and',
             '# against the video, fix or delete what is wrong, add what is missing,',
             '# then remove the status line. Strings: 1 is the high E.',
             f'video: {video}',
             'tuning: ' + next((name for name, pitches in TUNINGS.items()
                                if pitches == tuple(tuning)),
                               ' '.join(str(p) for p in tuning)),
             'capo: 0']
    if draft:
        lines.append('status: draft')
    ordered = sorted(notes, key=lambda n: (n.start, n.pitch))
    events, current = [], []
    for note in ordered:
        if current and note.start - current[0].start > chord_window:
            events.append(current)
            current = []
        current.append(note)
    if current:
        events.append(current)
    for event in events:
        tokens = []
        for note in event:
            if note.string is not None and note.fret is not None:
                tokens.append(f'{note.string}:{note.fret}')
            else:
                tokens.append(_name(note.pitch))
        lines.append(f'@{event[0].start:.3f} ' + ' '.join(tokens))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + '\n')


def _name(pitch):
    return ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][pitch % 12] \
        + str(pitch//12 - 1)


def read_prediction(path):
    """Notes from a ``packages/music`` performance document."""
    document = json.loads(Path(path).read_text())
    return [Note(int(n['pitch']), float(n['startSeconds']), n.get('string'), n.get('fret'))
            for n in document['notes']]


def match_timed(label, predicted, tolerance=TOLERANCE):
    """(label index, prediction index) pairs: same pitch, onsets within ``tolerance``,
    each note used once, as many pairs as possible."""
    import mir_eval
    reference, estimate = label, predicted
    if not reference or not estimate:
        return []
    hz = lambda notes: np.array([440.*2**((n.pitch - 69)/12) for n in notes])
    spans = lambda notes: np.array([[n.start, n.start + .1] for n in notes])
    return mir_eval.transcription.match_notes(
        spans(reference), hz(reference), spans(estimate), hz(estimate),
        onset_tolerance=tolerance, pitch_tolerance=50., offset_ratio=None)  # type: ignore[arg-type]  # mir_eval documents None to ignore offsets.


def match_sequence(events, predicted, chord_window=.05):
    """Align the transcription to an untimed label in order.

    Both sides become a sequence of chords; a dynamic program finds the order-
    preserving pairing of label chords with transcribed ones that matches the
    most pitches, skipping chords on either side for free. Within a paired
    chord, notes pair by pitch."""
    groups, current = [], []
    order = sorted(range(len(predicted)), key=lambda i: (predicted[i].start or 0., predicted[i].pitch))
    for index in order:
        if current and (predicted[index].start or 0.) - (predicted[current[0]].start or 0.) > chord_window:
            groups.append(current)
            current = []
        current.append(index)
    if current:
        groups.append(current)
    flat, offsets = [], []
    for event in events:
        offsets.append(len(flat))
        flat.extend(event)

    def common(a, b):
        left = [flat[offsets[a] + k].pitch for k in range(len(events[a]))]
        right = [predicted[i].pitch for i in groups[b]]
        return sum(min(left.count(p), right.count(p)) for p in set(left))

    rows, columns = len(events), len(groups)
    best = np.zeros((rows + 1, columns + 1), np.int64)
    for a in range(1, rows + 1):
        for b in range(1, columns + 1):
            best[a, b] = max(best[a - 1, b], best[a, b - 1],
                             best[a - 1, b - 1] + common(a - 1, b - 1))
    pairs, a, b = [], rows, columns
    while a and b:
        if best[a, b] == best[a - 1, b]:
            a -= 1
        elif best[a, b] == best[a, b - 1]:
            b -= 1
        else:
            free = list(groups[b - 1])
            for k in range(len(events[a - 1])):
                pitch = flat[offsets[a - 1] + k].pitch
                hit = next((i for i in free if predicted[i].pitch == pitch), None)
                if hit is not None:
                    free.remove(hit)
                    pairs.append((offsets[a - 1] + k, hit))
            a, b = a - 1, b - 1
    return sorted(pairs)


def score(label, predicted, tolerance=TOLERANCE, timed=None) -> dict[str, Any]:
    """Counts and the errors behind them, for one clip."""
    reference = label.notes
    timed = label.timed if timed is None else timed
    pairs = (match_timed(reference, predicted, tolerance) if timed
             else match_sequence(label.events, predicted))
    positioned = right_string = open_total = open_right = 0
    wrong, offsets = [], []
    for r, p in pairs:
        truth, guess = reference[r], predicted[p]
        if timed and truth.start is not None and guess.start is not None:
            offsets.append(guess.start - truth.start)
        if truth.string is None:
            continue
        positioned += 1
        hit = guess.string == truth.string
        right_string += hit
        if truth.fret == 0:
            open_total += 1
            open_right += hit
        if not hit:
            wrong.append(dict(time=truth.start, pitch=_name(truth.pitch),
                              want=f'{truth.string}:{truth.fret}',
                              got=(f'{guess.string}:{guess.fret}' if guess.string else 'unplaced')))
    matched_reference = {r for r, _ in pairs}
    matched_predicted = {p for _, p in pairs}
    missed = [dict(time=n.start, pitch=_name(n.pitch),
                   at=f'{n.string}:{n.fret}' if n.string else None)
              for i, n in enumerate(reference) if i not in matched_reference]
    extra = [dict(time=n.start, pitch=_name(n.pitch),
                  at=f'{n.string}:{n.fret}' if n.string else None)
             for i, n in enumerate(predicted) if i not in matched_predicted]
    return dict(timed=timed, referenced=len(reference), predicted=len(predicted),
                matched=len(pairs), positioned=positioned, right_string=right_string,
                labelled_positions=sum(n.string is not None for n in reference),
                open=open_total, open_right=open_right,
                onset_offset_ms=(round(float(np.median(offsets))*1000, 1) if offsets else None),
                missed=missed, extra=extra, wrong_string=wrong)


def summarise(counts):
    """Note F1, string accuracy and tab F1 from summed counts."""
    matched, predicted, referenced = counts['matched'], counts['predicted'], counts['referenced']
    precision = matched/predicted if predicted else 0.
    recall = matched/referenced if referenced else 0.
    f1 = 2*precision*recall/(precision + recall) if precision + recall else 0.
    string = counts['right_string']/counts['positioned'] if counts['positioned'] else None
    # Tab F1 counts a note right only when its string is right too. Label notes
    # with no position, and the transcribed notes matched to them, cannot be
    # judged that way and leave the tab score.
    tab = None
    if counts['labelled_positions']:
        judged = predicted - (matched - counts['positioned'])
        tab_precision = counts['right_string']/judged if judged else 0.
        tab_recall = counts['right_string']/counts['labelled_positions']
        tab = (2*tab_precision*tab_recall/(tab_precision + tab_recall)
               if tab_precision + tab_recall else 0.)
    return dict(note_f1=f1, precision=precision, recall=recall, string=string, tab_f1=tab,
                open=(counts['open_right']/counts['open'] if counts['open'] else None))


def labels(root=ROOT):
    return [parse(path) for path in sorted(Path(root).glob('*.tab.txt'))]


def slug(video):
    return re.sub(r'[^a-z0-9]+', '-', Path(video).stem.lower()).strip('-') or 'clip'


def transcribe(label, output, options):
    from processing.music.transcribe import transcribe_video
    result = transcribe_video(label.video, output/slug(label.video), options,
                              log=lambda *a, **k: None)
    return read_prediction(result.performance)


def predictions_from(directory, label):
    """A transcription written earlier: ``<dir>/<slug>/<stem>.performance.json``,
    or the ``.tab.jsonl`` beside it from runs made before the document existed."""
    assert label.video is not None
    base = Path(directory)/slug(label.video)/Path(label.video).stem
    document = base.with_name(base.name + '.performance.json')
    if document.is_file():
        return read_prediction(document)
    jsonl = base.with_name(base.name + '.tab.jsonl')
    if jsonl.is_file():
        rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
        return [Note(int(r['pitch']), float(r['startSeconds']), r.get('string'), r.get('fret'))
                for r in rows if r.get('kind') == 'note']
    raise FileNotFoundError(f'{document} not found; transcribe the clip first, or score '
                            'without --predictions to run the pipeline')


def _fmt(value):
    return '   -' if value is None else f'{value:.3f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    draft = commands.add_parser('draft', help='Write a draft label from the pipeline')
    draft.add_argument('video', type=Path)
    draft.add_argument('--root', type=Path, default=ROOT)
    draft.add_argument('--from-output', type=Path,
                       help='Use this transcription directory instead of running the '
                            'pipeline, e.g. runs/transcribe-final')
    draft.add_argument('--force', action='store_true', help='Overwrite an existing label')
    run = commands.add_parser('score', help='Score the pipeline against every label')
    run.add_argument('--root', type=Path, default=ROOT)
    run.add_argument('--predictions', type=Path,
                     help='Score transcriptions already on disk instead of running the pipeline')
    run.add_argument('--output', type=Path, default=Path('runs/testset/latest'),
                     help='Where the pipeline writes when it runs')
    run.add_argument('--tolerance', type=float, default=TOLERANCE,
                     help='Onset tolerance in seconds for timed labels')
    run.add_argument('--untimed', action='store_true',
                     help='Align every label as a sequence, ignoring its times')
    run.add_argument('--include-drafts', action='store_true',
                     help='Score drafts too. Only for checking the tool: a draft is the '
                          "pipeline's own answer")
    run.add_argument('--verifier', type=Path)
    run.add_argument('--verifier-threshold', type=float)
    run.add_argument('--device', default='mps')
    run.add_argument('--report', type=Path, help='Write every count and error as JSON')
    run.add_argument('--errors', action='store_true', help='List each clip\'s errors')
    args = parser.parse_args()

    if args.command == 'draft':
        name = args.video.stem
        path = args.root/f'{name}.tab.txt'
        if path.exists() and not args.force:
            raise SystemExit(f'{path} exists; pass --force to overwrite')
        stub = Label(path, args.video, TUNINGS['standard'])
        if args.from_output:
            notes = predictions_from(args.from_output, stub)
        else:
            from processing.music.options import Options
            notes = transcribe(stub, Path('runs/testset/drafts'), Options(write_video=False))
        write(path, notes, args.video, stub.tuning)
        print(f'wrote {path}: {len(notes)} notes. Correct it, then delete its status line.')
        return

    found = labels(args.root)
    if not found:
        raise SystemExit(f'No *.tab.txt labels under {args.root}; see docs/testset.md')
    drafts = [label for label in found if label.draft]
    if drafts and not args.include_drafts:
        print(f'skipping {len(drafts)} draft label(s): ' +
              ', '.join(label.path.name for label in drafts))
        found = [label for label in found if not label.draft]
    if not found:
        raise SystemExit('Nothing to score: every label is still a draft')
    options = None
    if args.predictions is None:
        from dataclasses import replace
        from processing.music.options import Options
        options = replace(Options(write_video=False), device=args.device,
                          verifier=args.verifier, verifier_threshold=args.verifier_threshold)
    total = dict(referenced=0, predicted=0, matched=0, positioned=0, right_string=0,
                 labelled_positions=0, open=0, open_right=0)
    report = {}
    print(f'\n{"clip":<28} {"mode":<6} {"notes":>5} {"F1":>6} {"P":>6} {"R":>6} '
          f'{"string":>7} {"tab F1":>7} {"open":>6} {"offset ms":>9}')
    for label in found:
        if label.video is None or not Path(label.video).is_file():
            print(f'{label.path.name}: video {label.video} not found; skipped')
            continue
        predicted = (predictions_from(args.predictions, label) if args.predictions
                     else transcribe(label, args.output, options))
        counts = score(label, predicted, args.tolerance, False if args.untimed else None)
        for key in total:
            total[key] += counts[key]
        summary = summarise(counts)
        report[label.path.name] = dict(counts, **summary)
        print(f'{label.path.name.removesuffix(".tab.txt")[:28]:<28} '
              f'{"timed" if counts["timed"] else "order":<6} {counts["referenced"]:>5} '
              f'{summary["note_f1"]:.3f} {summary["precision"]:.3f} {summary["recall"]:.3f} '
              f'{_fmt(summary["string"]):>7} {_fmt(summary["tab_f1"]):>7} '
              f'{_fmt(summary["open"]):>6} '
              f'{"" if counts["onset_offset_ms"] is None else counts["onset_offset_ms"]:>9}')
        if args.errors:
            for kind in ('missed', 'extra', 'wrong_string'):
                for error in counts[kind]:
                    print(f'    {kind:<12} {error}')
    summary = summarise(total)
    print(f'{"all clips":<28} {"":<6} {total["referenced"]:>5} {summary["note_f1"]:.3f} '
          f'{summary["precision"]:.3f} {summary["recall"]:.3f} {_fmt(summary["string"]):>7} '
          f'{_fmt(summary["tab_f1"]):>7} {_fmt(summary["open"]):>6}')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(total=dict(total, **summary), clips=report),
                                          indent=1))
        print(f'\nwrote {args.report}')


if __name__ == '__main__':
    main()
