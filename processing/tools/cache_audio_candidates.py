"""Run the unchanged audio pipeline once per WAV for visual/audio experiments."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np

from processing.music import verify
from processing.music.pitch import raw_activations, extract_audio
from processing.tools.replay_fusion import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-root', type=Path, required=True)
    parser.add_argument('--pattern', default='*.wav')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('runs/verifier/v8-gt/best.pt'))
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, channels, _ = verify.load(args.model, args.device)
    paths = sorted(args.audio_root.glob(args.pattern), key=lambda p: p.name)
    if not paths:
        raise ValueError('No WAV files found')
    audio_metadata = {}
    metadata = dict(model=str(args.model), model_sha256=fingerprint(args.model),
                    min_pitch=40, max_pitch=86, audio=audio_metadata)
    for count, path in enumerate(paths):
        with tempfile.TemporaryDirectory() as temporary:
            audio = path if path.suffix.lower() == '.wav' else extract_audio(
                path, Path(temporary)/'audio.wav')
            activations = raw_activations(audio)
            cqt = (verify.align_cqt(verify.spectrum(audio), activations.times[:len(activations.note)],
                                    activations.note.shape[1]) if channels > verify.CHANNELS else None)
        prepared = verify.prepare(activations.note, activations.onset, activations.contour, cqt)
        found = verify.peaks(activations.onset)
        found[min(len(activations.note), len(activations.times)):] = False
        found[:, :40-verify.LOWEST_BIN] = False
        found[:, 86-verify.LOWEST_BIN+1:] = False
        frame, pitch = [v.astype(np.int64) for v in np.nonzero(found)]
        scores, deltas, strings = verify.score(model, prepared, frame, pitch,
                                                channels, args.device, batch=256)
        np.savez_compressed(args.output/f'{path.stem}.npz', note=activations.note,
                            onset=activations.onset, times=activations.times,
                            frame=frame, pitch=pitch, scores=scores, deltas=deltas, strings=strings)
        audio_metadata[path.name] = dict(path=str(path), sha256=fingerprint(path))
        print(f'{count+1}/{len(paths)} {path.name}: {len(frame)} candidates', flush=True)
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
