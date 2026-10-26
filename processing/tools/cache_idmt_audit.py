"""Cache measurement-only IDMT verifier outputs for reproducible error audits.

IDMT assets and labels are never used by the training tools in this campaign.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from processing.music import verify
from processing.tools import idmt
from processing.tools.replay_fusion import fingerprint


def main():
    from basic_pitch.note_creation import model_frames_to_time
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path('runs/verifier/v8-gt/best.pt')
    model, channels, _ = verify.load(checkpoint, args.device)
    found, _ = idmt.recordings(idmt.ROOT)
    names = {}
    for number, recording in enumerate(found, 1):
        path = idmt.POSTERIORGRAMS/f'{idmt.key(recording)}.npz'
        if not path.is_file():
            continue
        reference = recording.notes()
        if not reference:
            continue
        with np.load(path) as stored:
            note = stored['note'].astype(np.float32)
            onset = stored['onset'].astype(np.float32)
            contour = stored['contour'].astype(np.float32)
        times = model_frames_to_time(len(contour))
        found_peaks = verify.peaks(onset)
        found_peaks[min(len(note), len(times)):] = False
        frame, pitch = [x.astype(np.int64) for x in np.nonzero(found_peaks)]
        spectrum = (verify.align_cqt(verify.spectrum(recording.audio), times[:len(note)], note.shape[1])
                    if channels > verify.CHANNELS else None)
        prepared = verify.prepare(note, onset, contour, spectrum)
        scores, deltas, strings = verify.score(model, prepared, frame, pitch, channels,
                                               args.device, batch=256)
        key = idmt.key(recording)
        output = args.output/f'{key}.npz'
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, note=note, onset=onset, times=times,
                            frame=frame, pitch=pitch, scores=scores, deltas=deltas, strings=strings,
                            reference=np.array([[n.start, n.end, n.pitch, n.string] for n in reference]))
        names[key] = dict(audio=str(recording.audio), group=recording.group,
                          sha256=fingerprint(recording.audio))
        if number % 100 == 0:
            print(f'{number}/{len(found)} IDMT recordings', flush=True)
    (args.output/'metadata.json').write_text(json.dumps(dict(
        measurement_only=True, model=str(checkpoint), model_sha256=fingerprint(checkpoint),
        recordings=names), indent=2))


if __name__ == '__main__':
    main()
