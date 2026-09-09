"""Offline guitar transcription: fretboard + fingertips + pitch -> tab overlay.

Basic Pitch reads a whole soundtrack, so this runs on a recording rather than a
live camera. The video is decoded twice: once to run the models, once to draw,
which keeps the string-order decision in front of every frame it affects.

Nothing here is a validated transcription. Pitch comes from a general-purpose
model, fret geometry from an under-trained detector, and string position from an
interpolation. Read the flags.
"""
from __future__ import annotations
import argparse
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import time
import cv2
import numpy as np

from processing.vision.preview import draw_text, draw_observation, open_capture
from processing.vision.tracker import FretboardTracker, decode_maps, transform_observation
from .fretboard import DEFAULT_STRING_INSET, board_transform
from .fusion import (assign_positions, finger_contacts, hand_position, plan_positions,
                     unexplained_contacts)
from .hands import CONNECTIONS, HandTracker, fretting_hand
from .pitch import notes_from_media
from .tuning import STANDARD_TUNING, note_name, validate_tuning

SUPPORT_COLORS = {'rescued': (255, 130, 240), 'fingered': (120, 255, 120), 'open': (255, 210, 120),
                  'position-only': (80, 190, 255), 'no-hand': (150, 150, 150),
                  'no-board': (150, 150, 150), 'none': (150, 150, 150)}
CHORD_WINDOW = .05


@dataclass
class FrameRecord:
    index: int
    timestamp: float
    state: str
    observation: object = None
    hand_points: np.ndarray | None = None
    hand_label: str = ''
    hands: list = field(default_factory=list)
    inference_ms: float = 0.
    latency_ms: float = 0.


def group_notes(notes, window=CHORD_WINDOW):
    """Cluster note onsets into chords so one placement decision covers each."""
    groups, current = [], []
    for note in sorted(notes, key=lambda n: n.start):
        if current and note.start - current[0].start > window:
            groups.append(current); current = []
        current.append(note)
    if current:
        groups.append(current)
    return groups


def nearest_record(records, timestamps, timestamp, tolerance=.12):
    """Frame closest to a note onset, preferring one that actually saw the board."""
    if not records:
        return None
    low = bisect_left(timestamps, timestamp-tolerance)
    high = bisect_right(timestamps, timestamp+tolerance)
    order = sorted(range(low, high),
                   key=lambda i: abs(timestamps[i]-timestamp))
    fallback = records[order[0]] if order else None
    for index in order:
        if records[index].observation is not None:
            return records[index]
    return fallback


def transcribe_notes(records, notes, tuning, max_fret, inset, flipped, strings=None):
    """Assign every note a (string, fret), one decision per chord."""
    timestamps = [r.timestamp for r in records]
    strings = len(tuning) if strings is None else strings
    groups = group_notes(notes)
    # Pass one: what the video saw at each onset.
    evidence = []
    for group in groups:
        record = nearest_record(records, timestamps, group[0].start)
        transform = None
        if record is not None and record.observation is not None:
            transform = board_transform(record.observation, strings=strings,
                                        inset=inset, flipped=flipped)
        hand = None
        if record is not None and record.hand_points is not None:
            hand = _hand_from(record)
        contacts = finger_contacts(hand, transform, max_fret=max_fret) if transform else []
        evidence.append((record, transform, hand, contacts))
    # Pass two: plan the neck position across the whole clip before placing any
    # note, so a chord with no visible hand is answered by the chords around it
    # rather than by a running average of earlier guesses.
    planned = plan_positions(groups, [hand_position(contacts)
                                      for _, _, _, contacts in evidence],
                             tuning, max_fret)
    assignments, per_group = {}, []
    sounding = []
    for group, (record, transform, hand, contacts), position in zip(
            groups, evidence, planned):
        sounding = [note for note in sounding if note.end > group[0].start]
        results = assign_positions(group, contacts, tuning, max_fret,
                                   board=transform is not None, hand=hand is not None,
                                   occupied=sounding, position=position)
        for result in results:
            # Onset and pitch are not unique IDs (nor are rounded timestamps).
            assignments[len(assignments)] = result
        sounding.extend(note for note in results if note.string is not None)
        per_group.append((record, contacts, results))
    return assignments, per_group


def _hand_from(record):
    from .hands import Hand
    return Hand(record.hand_points, record.hand_label, 1.)


def support_rate(per_group):
    total = sum(len(results) for _, _, results in per_group)
    if not total:
        return 0.
    return sum(1 for _, _, results in per_group for note in results
               if note.support in ('fingered', 'open')) / total


def fingered_rate(per_group):
    """Only direct fingertip support distinguishes the two string orders."""
    fingered = sum(1 for _, _, results in per_group for n in results if n.support == 'fingered')
    total = sum(1 for _, _, results in per_group for n in results if n.fret)
    return fingered / total if total else 0.


def analyse(video, model, device, notes, tuning, args):
    """Pass one: run the detector and hand tracker over every frame."""
    from processing.vision.model import DenseDetector
    detector = DenseDetector(model, device, args.imgsz)
    tracker = FretboardTracker(max_gap=args.max_gap)
    capture = open_capture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.
    records = []
    try:
        with HandTracker(args.hand_model, num_hands=args.hands) as tracker_hands:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok or (args.max_frames and index >= args.max_frames):
                    break
                timestamp = index/fps
                start = time.perf_counter()
                maps = detector.predict(frame)
                inference = (time.perf_counter()-start)*1000
                observation = tracker.update(frame, decode_maps(maps, args.threshold), timestamp)
                hands = tracker_hands.detect(frame, timestamp)
                chosen = fretting_hand(hands, observation)
                records.append(FrameRecord(
                    index, timestamp, tracker.state, observation,
                    None if chosen is None else chosen.points.copy(),
                    '' if chosen is None else chosen.label,
                    [h.points.copy() for h in hands], inference,
                    (time.perf_counter()-start)*1000))
                if args.progress and index % max(1, int(fps)) == 0:
                    print(f'  frame {index} t={timestamp:5.1f}s state={tracker.state}', flush=True)
                index += 1
    finally:
        capture.release()
    return records, fps


def mirror_points(points, width, mirror):
    points = np.asarray(points, float)
    if not mirror:
        return points
    return np.stack([width-1-points[..., 0], points[..., 1]], axis=-1)


def draw_strings(frame, transform, max_fret, strings=6, mirror=False):
    """Draw the interpolated strings.

    The transform is always the one fitted to the unmirrored observation, and
    only its output points are mirrored. Refitting on a mirrored board would
    flip the across axis and silently disagree with the tab.
    """
    for string in range(1, strings+1):
        points = transform.string_polyline(string, max_fret)
        if not np.isfinite(points).all():
            continue
        points = mirror_points(points, frame.shape[1], mirror)
        cv2.polylines(frame, [points.astype(np.int32)], False, (90, 90, 110), 1, cv2.LINE_AA)


def draw_hand(frame, points, contacts=None):
    for a, b in CONNECTIONS:
        cv2.line(frame, tuple(points[a].astype(int)), tuple(points[b].astype(int)), (200, 120, 200), 1, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, tuple(point.astype(int)), 2, (230, 160, 230), -1)
    for contact in contacts or []:
        colour = (120, 255, 120) if contact.on_board else (110, 110, 110)
        cv2.circle(frame, tuple(np.asarray(contact.point).astype(int)), 6, colour, 2)
        draw_text(frame, contact.label, contact.point+np.array([8, -8]), .42, colour)


def draw_tab_strip(frame, active, history, timestamp, window=4., strings=6,
                   tuning=None):
    if tuning is not None:
        strings = len(tuning)
    height, width = frame.shape[:2]
    top, bottom = height-115, height-25
    cv2.rectangle(frame, (0, top-18), (width, height), (18, 18, 22), -1)
    spacing = (bottom-top)/max(strings-1, 1)
    for string in range(strings):
        y = int(top+string*spacing)
        cv2.line(frame, (46, y), (width-10, y), (70, 70, 80), 1)
        label = note_name(tuning[string]) if tuning is not None else str(string+1)
        draw_text(frame, label, (10, y+4), .42, (140, 140, 150))
    draw_text(frame, f'last {window:.0f}s', (10, top-24), .4, (140, 140, 150))
    for note in history:
        if not (timestamp-window <= note.start <= timestamp) or note.string is None:
            continue
        x = int(46+(note.start-(timestamp-window))/window*(width-60))
        y = int(top+(note.string-1)*spacing)
        colour = SUPPORT_COLORS.get(note.support, (150, 150, 150))
        if note in active:
            cv2.circle(frame, (x, y), 9, (60, 60, 70), -1)
        draw_text(frame, str(note.fret), (x-4, y+4), .46, colour, 1)


def render(video, records, assignments, notes, args, fps, transform_kwargs, output):
    """Pass two: draw overlays onto the original frames."""
    capture = open_capture(str(video))
    writer = None
    ordered = sorted(assignments.values(), key=lambda n: n.start)
    try:
        for record in records:
            ok, frame = capture.read()
            if not ok:
                break
            display = cv2.flip(frame, 1) if args.mirror else frame.copy()
            matrix = np.array([[-1., 0., frame.shape[1]-1.], [0., 1., 0.], [0., 0., 1.]])
            observation = record.observation
            transform = None
            if observation is not None:
                transform = board_transform(observation, **transform_kwargs)
            contacts = []
            hand = _hand_from(record) if record.hand_points is not None else None
            if transform is not None:
                contacts = finger_contacts(hand, transform, max_fret=args.max_fret)
            shown = observation
            if observation is not None and args.mirror:
                shown = transform_observation(observation, matrix)
            if shown is not None:
                draw_observation(display, shown, record.state)
                if transform is not None and args.show_strings:
                    draw_strings(display, transform, args.max_fret,
                                 strings=transform.strings, mirror=args.mirror)
            for points in record.hands:
                shown_points = mirror_points(points, frame.shape[1], args.mirror)
                is_fretting = (record.hand_points is not None
                               and np.allclose(points, record.hand_points))
                draw_hand(display, shown_points,
                          _mirror_contacts(contacts, frame.shape[1], args.mirror) if is_fretting else None)
            active = [n for n in ordered if n.start <= record.timestamp < n.end]
            draw_tab_strip(display, active, ordered, record.timestamp, tuning=args.tuning)
            _draw_hud(display, record, transform, active, contacts, args)
            if writer is None:
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter.fourcc(*'mp4v'),
                                         fps, (display.shape[1], display.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot open video writer for {output}')
            writer.write(display)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return output


def _mirror_contacts(contacts, width, mirror):
    """Move contact markers for display only; string and fret stay as measured."""
    if not mirror:
        return contacts
    import copy
    mirrored = []
    for contact in contacts:
        clone = copy.copy(contact)
        clone.point = mirror_points(contact.point, width, True)
        mirrored.append(clone)
    return mirrored


def _draw_hud(frame, record, transform, active, contacts, args):
    numbering = record.observation.numbering if record.observation is not None else 'no board'
    board = 'none' if transform is None else f'{transform.anchors} anchors'
    draw_text(frame, f'{record.index:5d}  t={record.timestamp:6.2f}s  {record.state}  '
                     f'numbering: {numbering}  board: {board}', (12, 24), .5)
    draw_text(frame, f'inference {record.inference_ms:.0f} ms   strings: interpolated '
                     f'(inset {args.inset:.2f}{", flipped" if args.flipped else ""}) | Pink: rescued', (12, 44), .42,
              (170, 170, 180))
    y = 70
    for note in active[:6]:
        colour = SUPPORT_COLORS.get(note.support, (150, 150, 150))
        position = '--' if note.string is None else f'string {note.string} fret {note.fret}'
        flags = (' [' + ','.join(note.flags) + ']') if note.flags else ''
        draw_text(frame, f'{note.name:<4} {position:<18} {note.support:<14} '
                         f'{note.confidence:.2f}{flags}', (12, y), .46, colour)
        y += 20
    stray = unexplained_contacts(contacts, active)
    if stray:
        draw_text(frame, 'unexplained fingers: ' + ', '.join(c.label for c in stray),
                  (12, y), .42, (80, 190, 255))


def write_jsonl(path, records, assignments, args, tuning, summary):
    ordered = sorted(assignments.values(), key=lambda n: n.start)
    with path.open('w') as handle:
        handle.write(json.dumps({'kind': 'summary', **summary})+'\n')
        for note in ordered:
            handle.write(json.dumps({
                'kind': 'note', 'pitch': note.pitch, 'name': note.name,
                'startSeconds': round(note.start, 4), 'durationSeconds': round(note.end-note.start, 4),
                'amplitude': round(note.amplitude, 4), 'string': note.string, 'fret': note.fret,
                'support': note.support, 'confidence': round(note.confidence, 4),
                'finger': note.finger, 'alternatives': note.alternatives, 'flags': note.flags,
                **({'evidence': note.evidence} if note.evidence else {}),
            })+'\n')


def mux_audio(video, audio, destination):
    """Copy the original soundtrack onto the rendered video."""
    if not shutil.which('ffmpeg'):
        return None
    result = subprocess.run(
        ['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', str(video), '-i', str(audio),
         '-map', '0:v:0', '-map', '1:a:0',
         '-c:v', 'copy', '-c:a', 'aac', '-af', 'apad', '-shortest', str(destination)],
        capture_output=True, text=True)
    return destination if result.returncode == 0 and destination.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, type=Path, help='Dense fretboard checkpoint')
    parser.add_argument('--video', required=True, type=Path, help='Recorded clip with audio')
    parser.add_argument('--output', required=True, type=Path, help='Directory for overlay and JSONL')
    parser.add_argument('--audio', type=Path, help='Use this audio instead of the video soundtrack')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--imgsz', type=int)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--max-gap', type=float, default=.5)
    parser.add_argument('--max-fret', type=int, default=22)
    parser.add_argument('--inset', type=float, default=DEFAULT_STRING_INSET,
                        help='Fraction of board width between the edge and the outer strings')
    parser.add_argument('--string-order', choices=['auto', 'normal', 'flipped'], default='auto',
                        help='Which end of a fret wire carries string 1; auto picks the order '
                             'that explains more notes with an actual fingertip')
    parser.add_argument('--tuning', default='standard',
                        help='"standard" or comma-separated MIDI pitches, highest string first')
    parser.add_argument('--hands', type=int, default=2)
    parser.add_argument('--hand-model', type=Path)
    # Tuned on GuitarSet; see docs/pitch-tuning.md.
    parser.add_argument('--onset-threshold', type=float, default=.6)
    parser.add_argument('--frame-threshold', type=float, default=.45)
    parser.add_argument('--min-note-ms', type=float, default=60.)
    parser.add_argument('--mirror', action='store_true', help='Mirror the rendered video')
    parser.add_argument('--no-strings', dest='show_strings', action='store_false',
                        help='Hide the interpolated string lines')
    parser.add_argument('--visual-rescue', action='store_true',
                        help='Experimentally add weak audio notes confirmed by stable fingertips')
    parser.add_argument('--no-video', dest='write_video', action='store_false')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or not 0 <= args.inset < .5 or args.max_fret < 1:
        parser.error('Invalid threshold, inset or max-fret')
    if args.hands < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('Invalid hands or max-frames')
    if (not np.isfinite(args.max_gap) or args.max_gap <= 0
            or not 0 <= args.onset_threshold <= 1 or not 0 <= args.frame_threshold <= 1
            or not np.isfinite(args.min_note_ms) or args.min_note_ms <= 0
            or (args.imgsz is not None and (args.imgsz < 64 or args.imgsz % 32))):
        parser.error('Invalid max-gap, pitch thresholds, min-note-ms or imgsz')
    try:
        tuning = (STANDARD_TUNING if args.tuning == 'standard'
                  else validate_tuning([int(v) for v in args.tuning.split(',')]))
    except ValueError as error:
        parser.error(str(error))
    args.tuning = tuning
    args.output.mkdir(parents=True, exist_ok=True)
    from processing.training.train import select_device
    device = select_device(args.device)

    # Name the input up front: the output files are named after the video, so a
    # stale --video with a fresh --output is otherwise silently confusing.
    print(f'Transcribing {args.video} -> {args.output}/{args.video.stem}.*', flush=True)
    print('Detecting pitch...', flush=True)
    with tempfile.TemporaryDirectory() as workdir:
        notes, activations, audio_path = notes_from_media(
            args.video, args.audio, workdir, onset_threshold=args.onset_threshold,
            frame_threshold=args.frame_threshold, minimum_note_length=args.min_note_ms,
            minimum_frequency=440 * 2 ** ((min(tuning)-69)/12),
            # Basic Pitch rounds the cutoff to a MIDI bin and excludes that bin.
            # Use the next semitone so odd/even top pitches are both retained.
            maximum_frequency=440 * 2 ** ((min(127, max(tuning)+args.max_fret)+1-69)/12))
        print(f'  {len(notes)} note events', flush=True)
        print('Detecting fretboard and hands...', flush=True)
        records, fps = analyse(args.video, args.model, device, notes, tuning, args)
        print(f'  {len(records)} frames at {fps:.2f} fps', flush=True)
        if not records:
            raise RuntimeError(f'No video frames decoded from {args.video}')
        duration = records[-1].timestamp + 1/fps
        notes = [replace(note, end=min(note.end, duration)) for note in notes
                 if 0 <= note.start < duration and note.end > note.start]

        options = ([False, True] if args.string_order == 'auto'
                   else [args.string_order == 'flipped'])
        best, args.flipped = None, options[0]
        order_scores = {}
        for flipped in options:
            assignments, per_group = transcribe_notes(
                records, notes, tuning, args.max_fret, args.inset, flipped, len(tuning))
            score = fingered_rate(per_group)
            order_scores['flipped' if flipped else 'normal'] = score
            print(f'  string order {"flipped" if flipped else "normal"}: '
                  f'{score:.0%} of fretted notes have fingertip support', flush=True)
            if best is None or score > best[0]:
                best, args.flipped = (score, assignments, per_group), flipped
        assert best is not None  # Every string-order mode evaluates at least one option.
        _, assignments, per_group = best
        transform_kwargs = dict(strings=len(tuning), inset=args.inset, flipped=args.flipped)

        rescue_report = {'enabled': args.visual_rescue, 'accepted': 0}
        baseline_count = len(notes)
        if args.visual_rescue:
            from .rescue import rescue_notes
            rescued, details = rescue_notes(activations, records, list(assignments.values()),
                                             tuning, args.max_fret, args.inset, args.flipped,
                                             duration)
            rescue_report.update(details)
            rescue_report.update(onset_threshold=.45, frame_threshold=.35)
            for note in rescued:
                assignments[len(assignments)] = note
            print(f'  visually rescued {len(rescued)} weak audio notes', flush=True)

        counts = {}
        for note in assignments.values():
            counts[note.support] = counts.get(note.support, 0)+1
        unexplained = []
        for record, contacts, results in per_group:
            if record is None:
                continue
            active = [n for n in assignments.values()
                      if n.start <= record.timestamp < n.end]
            stray = unexplained_contacts(contacts, results + active)
            if stray:
                unexplained.append({
                    'timestamp': record.timestamp,
                    'contacts': [{'finger': c.finger, 'string': c.string, 'fret': c.fret,
                                  'point': c.point.tolist()} for c in stray],
                })
        summary = {
            'video': str(args.video.resolve()), 'model': str(args.model.resolve()),
            'frames': len(records), 'fps': fps, 'notes': len(assignments),
            'baseline_notes': baseline_count, 'visual_rescue': rescue_report,
            'tuning': list(tuning), 'max_fret': args.max_fret, 'string_inset': args.inset,
            'string_order': 'flipped' if args.flipped else 'normal',
            'string_order_selected': args.string_order,
            'string_order_scores': order_scores,
            'fingertip_support_rate': round(fingered_rate(per_group), 4),
            'support_counts': counts,
            'unexplained_fingers': unexplained,
            'board_frames': sum(r.observation is not None for r in records),
            'hand_frames': sum(r.hand_points is not None for r in records),
            'limitations': [
                'String positions are interpolated across the detected board; the model has no string head.',
                'A fingertip near a cell is not proof the string is pressed, and an undetected finger is not proof it is open.',
                'Pitch comes from a general-purpose polyphonic model, not a guitar-specific one.',
                'Fret numbering needs the nut; without it notes fall back to position-only or no-board support.',
                'End-to-end audio/video transcription accuracy is unmeasured; GuitarSet benchmarks cover audio pitch and blind string assignment separately.',
            ],
        }
        jsonl = args.output/f'{args.video.stem}.tab.jsonl'
        write_jsonl(jsonl, records, assignments, args, tuning, summary)
        print(json.dumps({k: v for k, v in summary.items()
                          if k not in ('limitations', 'unexplained_fingers')}, indent=2))
        print(f'Notes: {jsonl}', flush=True)

        if args.write_video:
            print('Rendering overlay...', flush=True)
            silent = args.output/f'{args.video.stem}.overlay.silent.mp4'
            render(args.video, records, assignments, notes, args, fps, transform_kwargs, silent)
            final = args.output/f'{args.video.stem}.overlay.mp4'
            if mux_audio(silent, audio_path, final):
                silent.unlink(missing_ok=True)
            else:
                silent.replace(final)
                print('Audio mux failed; the overlay is silent.', flush=True)
            print(f'Overlay: {final}', flush=True)


if __name__ == '__main__':
    main()
