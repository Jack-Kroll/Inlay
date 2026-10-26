"""Compare a frozen string-head checkpoint on cached-vision real clips.

Recompute audio once per clip, then hold activations, CQT and vision fixed.
This reports qualitative placement changes, not reference-tab accuracy.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np

from processing.music.pitch import extract_audio, raw_activations, verified_notes
from processing.music import verify
from processing.music.transcribe import transcribe_notes, fingered_rate
from processing.tools.replay_fusion import read_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path('runs/verifier/v8-gt/best.pt'))
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    reports = {}
    for slug in ('blackbird', 'thunderstruck', 'sweet-child'):
        old = json.loads((args.root / f'{slug}-replay.json').read_text())
        source = [json.loads(line) for line in Path(old['source_notes']).read_text().splitlines()]
        summary = source[0]
        records = read_records(args.root / f'{slug}-vision.json')
        with tempfile.TemporaryDirectory() as directory:
            audio = extract_audio(Path(summary['video']), Path(directory) / 'audio.wav')
            activations = raw_activations(audio)
            cqt = verify.spectrum(audio)
            variants, baseline_notes, baseline_tab = {}, None, None
            for name, checkpoint in [('baseline', args.baseline), ('probe', args.probe)]:
                notes = verified_notes(activations, checkpoint, args.device, cqt=cqt,
                                       threshold=summary['verifier_threshold'],
                                       minimum_pitch=min(summary['tuning']),
                                       maximum_pitch=max(summary['tuning']) + summary['max_fret'])
                note_array = np.array([[n.pitch, n.start, n.end, n.amplitude] for n in notes])
                if baseline_notes is None:
                    baseline_notes = note_array
                if not np.array_equal(note_array, baseline_notes):
                    raise ValueError('Frozen string probe changed note detections')
                assigned, groups = transcribe_notes(
                    records, notes, summary['tuning'], summary['max_fret'],
                    summary['string_inset'], summary['string_order'] == 'flipped',
                    audio_weight=summary['audio_string_weight'])
                tab = [asdict(n) for n in assigned.values()]
                if baseline_tab is None:
                    baseline_tab = tab
                changes = [dict(pitch=a['pitch'], start=a['start'],
                                before=[a['string'], a['fret']], after=[b['string'], b['fret']])
                           for a, b in zip(baseline_tab, tab)
                           if (a['string'], a['fret']) != (b['string'], b['fret'])]
                variants[name] = dict(notes=len(notes), support=fingered_rate(groups),
                                      open_notes=sum(n['fret'] == 0 for n in tab),
                                      changes=changes, placements=tab)
            reports[slug] = variants
            print(slug, {k: {a: b for a, b in v.items() if a != 'placements'}
                         for k, v in variants.items()}, flush=True)
    (args.root / 'string-probe-clips.json').write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
