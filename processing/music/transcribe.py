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
import re
import subprocess
import shutil
import tempfile
import time
import cv2
import numpy as np

from processing.vision.preview import draw_text, draw_observation, open_capture
from processing.vision.tracker import FretboardTracker, decode_maps, transform_observation
from processing.vision import DEFAULT_MODEL
from .fretboard import DEFAULT_STRING_INSET, board_transform
from . import fusion, verify
from .fusion import (AUDIO_STRING_WEIGHT, assign_positions, finger_contacts,
                     hand_position, plan_positions, unexplained_contacts)
from .hands import CONNECTIONS, HandTracker, fretting_hand
from .options import Options
from .pitch import notes_from_media
from .tuning import STANDARD_TUNING, note_name, validate_tuning

# Overlay colour (BGR) for each kind of evidence behind a note's placement.
SUPPORT_COLORS = {'fingered': (120, 255, 120), 'open': (255, 210, 120),
                  'position-only': (80, 190, 255), 'no-hand': (150, 150, 150),
                  'no-board': (150, 150, 150), 'none': (150, 150, 150)}
# Onsets closer together than this (seconds) are treated as one chord.
CHORD_WINDOW = .05


def default_output_directory(video: Path) -> Path:
    """Put each recording in a stable, shell-friendly transcription folder."""
    slug = re.sub(r'[^a-z0-9]+', '-', video.stem.lower()).strip('-')
    return Path('runs/transcribe') / (slug or 'transcription')


@dataclass
class FrameRecord:
    """What the vision pass saw in one video frame, kept for the audio stage."""
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
    """Closest board frame, recovering a missing hand from at most 40 ms away."""
    if not records:
        return None
    low = bisect_left(timestamps, timestamp-tolerance)
    high = bisect_right(timestamps, timestamp+tolerance)
    order = sorted(range(low, high),
                   key=lambda i: abs(timestamps[i]-timestamp))
    fallback = records[order[0]] if order else None
    boarded = [records[index] for index in order
               if records[index].observation is not None]
    if not boarded:
        return fallback
    nearest = boarded[0]
    if nearest.hand_points is None:
        # A one-frame hand dropout should not discard nearby clear evidence.
        # Keep this stricter than the board search: fingers move between notes.
        for record in boarded:
            if (record.hand_points is not None
                    and abs(record.timestamp - timestamp) <= min(tolerance, .04)):
                return record
    return nearest


def transcribe_notes(records, notes, tuning, max_fret, inset, flipped, strings=None,
                     audio_weight=None, fret_weight=0., visual_priority=False, visual_policy=None):
    """Assign every note a (string, fret), one decision per chord.

    ``audio_weight`` overrides how much a note's audio string posterior counts;
    None uses ``fusion.AUDIO_STRING_WEIGHT``.
    """
    if audio_weight is None:
        audio_weight = fusion.AUDIO_STRING_WEIGHT
    timestamps = [r.timestamp for r in records]
    strings = len(tuning) if strings is None else strings
    groups = group_notes(notes)
    visual_context = None
    if visual_priority:
        from .visual_evidence import VisualContext, VisualPolicy
        visual_context = VisualContext.from_records(
            records, strings, inset, flipped, max_fret,
            policy=visual_policy if visual_policy is not None else VisualPolicy())
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
    # Pass three: pick a (string, fret) for every note in each chord, given the
    # fingertips seen and the planned neck position.
    assignments, per_group = {}, []
    sounding = []  # Notes still ringing, which a new pluck on their string cuts off.
    for group, (record, transform, hand, contacts), position in zip(
            groups, evidence, planned):
        sounding = [note for note in sounding if note.end > group[0].start]
        results = assign_positions(group, contacts, tuning, max_fret,
                                   board=transform is not None, hand=hand is not None,
                                   occupied=sounding, position=position,
                                   audio_weight=audio_weight, fret_weight=fret_weight,
                                   visual_evidence=([visual_context.placement(n.start, n.pitch, tuning, max_fret)
                                                     for n in group] if visual_context is not None else None))
        for result in results:
            # Onset and pitch are not unique IDs (nor are rounded timestamps).
            assignments[len(assignments)] = result
        sounding.extend(note for note in results if note.string is not None)
        per_group.append((record, contacts, results))
    return assignments, per_group


def _hand_from(record):
    """Rebuild a Hand from the landmark points stored on a frame record."""
    from .hands import Hand
    return Hand(record.hand_points, record.hand_label, 1.)


def fingered_rate(per_group):
    """Fraction of fretted notes that landed under an actual fingertip.

    Geometry can't tell which end of a fret wire is string 1. Transcribing both
    ways and keeping the one where more notes line up with a finger decides it;
    only direct fingertip support distinguishes the two orders.
    """
    fingered = sum(1 for _, _, results in per_group for n in results if n.support == 'fingered')
    total = sum(1 for _, _, results in per_group for n in results if n.fret)
    return fingered / total if total else 0.


def analyse(video, model, device, args):
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
                # Of the detected hands, keep the one actually on the neck.
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
    """Flip x coordinates horizontally when ``mirror`` is set."""
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
    """Hand skeleton, plus a labelled ring on each fingertip contact (green = on board)."""
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
    """Scrolling tab along the bottom: the last ``window`` seconds of notes."""
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
    # Each note is its fret number, placed on its string line at its onset time.
    for note in history:
        if not (timestamp-window <= note.start <= timestamp) or note.string is None:
            continue
        x = int(46+(note.start-(timestamp-window))/window*(width-60))
        y = int(top+(note.string-1)*spacing)
        colour = SUPPORT_COLORS.get(note.support, (150, 150, 150))
        if note in active:
            cv2.circle(frame, (x, y), 9, (60, 60, 70), -1)
        draw_text(frame, str(note.fret), (x-4, y+4), .46, colour, 1)


def render(video, records, assignments, args, fps, transform_kwargs, output):
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
            # Geometry is always computed in raw (unmirrored) coordinates;
            # mirroring is applied to the drawn points only.
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
            # Notes sounding at this frame.
            active = [n for n in ordered if n.start <= record.timestamp < n.end]
            draw_tab_strip(display, active, ordered, record.timestamp, tuning=args.tuning)
            _draw_hud(display, record, transform, active, contacts, transform_kwargs)
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


def _draw_hud(frame, record, transform, active, contacts, layout):
    """Status text in the top-left: tracker state, active notes, stray fingers."""
    numbering = record.observation.numbering if record.observation is not None else 'no board'
    board = 'none' if transform is None else f'{transform.anchors} anchors'
    draw_text(frame, f'{record.index:5d}  t={record.timestamp:6.2f}s  {record.state}  '
                     f'numbering: {numbering}  board: {board}', (12, 24), .5)
    draw_text(frame, f'inference {record.inference_ms:.0f} ms   strings: interpolated '
                     f'(inset {layout["inset"]:.2f}{", flipped" if layout["flipped"] else ""})', (12, 44), .42,
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


def write_jsonl(path, assignments, summary):
    """One summary line, then one line per note in time order."""
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
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL,
                        help=f'Dense fretboard checkpoint (default: {DEFAULT_MODEL})')
    parser.add_argument('--video', required=True, type=Path, help='Recorded clip with audio')
    parser.add_argument('--output', type=Path,
                        help='Output directory (default: runs/transcribe/<video-name>)')
    parser.add_argument('--audio', type=Path, help='Use this audio instead of the video soundtrack')
    parser.add_argument('--device', default='mps', help='Compute device (default: mps)')
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
    parser.add_argument('--verifier', type=Path,
                        help='Verifier checkpoint; defaults to the shipped one if present')
    parser.add_argument('--no-verifier', dest='use_verifier', action='store_false',
                        help='Decode with Basic Pitch thresholds instead of the verifier')
    # Both of these were chosen on GuitarSet, which is six acoustic guitars
    # through one microphone. They are exposed because they are the two things
    # to reach for when transcribing something that does not sound like that.
    parser.add_argument('--verifier-threshold', type=float,
                        help=f'Score a note must beat (default {verify.SCORE_THRESHOLD}); '
                             'lower it to report more notes')
    parser.add_argument('--audio-string-weight', type=float,
                        help=f'How much audio timbre decides the string '
                             f'(default {AUDIO_STRING_WEIGHT}); 0 leaves it to the camera '
                             'and musical context')
    parser.add_argument('--visual-priority', action='store_true',
                        help='Experimental: favor stable, unambiguous fretted placements '
                             'over audio string predictions')
    parser.add_argument('--recover-strong-attacks', action='store_true',
                        help='Experimental: recover rejected candidates with strong '
                             'audio attack and pitch support')
    parser.add_argument('--open-audio-neutral', type=float,
                        help='0 to 1: how far audio is kept from arguing a note off an '
                             f'open string (default {fusion.OPEN_AUDIO_NEUTRAL})')
    # The thresholds below are the fallback when no verifier is used; they were
    # tuned on GuitarSet, see docs/pitch-tuning.md.
    parser.add_argument('--onset-threshold', type=float, default=.6)
    parser.add_argument('--frame-threshold', type=float, default=.45)
    parser.add_argument('--min-note-ms', type=float, default=60.)
    parser.add_argument('--mirror', action='store_true', help='Mirror the rendered video')
    parser.add_argument('--no-strings', dest='show_strings', action='store_false',
                        help='Hide the interpolated string lines')
    parser.add_argument('--no-video', dest='write_video', action='store_false')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()
    try:
        tuning = (STANDARD_TUNING if args.tuning == 'standard'
                  else validate_tuning([int(v) for v in args.tuning.split(',')]))
    except ValueError as error:
        parser.error(str(error))
    options = Options(
        model=args.model, device=args.device, imgsz=args.imgsz, threshold=args.threshold,
        max_gap=args.max_gap, max_fret=args.max_fret, inset=args.inset,
        string_order=args.string_order, tuning=tuple(tuning), hands=args.hands,
        hand_model=args.hand_model, verifier=args.verifier, use_verifier=args.use_verifier,
        verifier_threshold=args.verifier_threshold,
        audio_string_weight=args.audio_string_weight, visual_priority=args.visual_priority,
        recover_strong_attacks=args.recover_strong_attacks,
        open_audio_neutral=args.open_audio_neutral, onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold, min_note_ms=args.min_note_ms,
        mirror=args.mirror, show_strings=args.show_strings, write_video=args.write_video,
        max_frames=args.max_frames, progress=args.progress)
    problems = options.problems()
    if problems:
        parser.error(problems[0])
    output = args.output if args.output is not None else default_output_directory(args.video)
    transcribe_video(args.video, output, options, audio=args.audio)


@dataclass
class Transcription:
    """What one run produced: the files it wrote and what is in them."""
    jsonl: Path
    performance: Path
    overlay: Path | None
    summary: dict
    assignments: dict


def performance(assignments, tuning):
    """The run as a ``packages/music`` performance document, version 1.

    That schema is the boundary the app reads; the JSONL beside it keeps the
    diagnostics. A note with no string carries neither string nor fret, as the
    schema requires the two together."""
    notes = []
    for note in sorted(assignments.values(), key=lambda n: (n.start, n.pitch)):
        duration = round(note.end - note.start, 4)
        if duration <= 0:
            continue
        entry = {'pitch': int(note.pitch), 'startSeconds': round(max(note.start, 0.), 4),
                 'durationSeconds': duration,
                 'confidence': round(min(max(float(note.confidence), 0.), 1.), 4)}
        if note.string is not None and note.fret is not None:
            entry.update(string=int(note.string), fret=int(note.fret))
        notes.append(entry)
    return {'version': 1, 'tuning': [int(p) for p in tuning], 'notes': notes}


def transcribe_video(video, output, options=None, audio=None, log=None):
    """Transcribe one recording into ``output``; the whole pipeline.

    Writes ``<stem>.tab.jsonl`` (summary and per-note diagnostics),
    ``<stem>.performance.json`` (the app's contract) and, unless
    ``options.write_video`` is off, ``<stem>.overlay.mp4``."""
    options = options or Options()
    log = log or print
    problems = options.problems()
    if problems:
        raise ValueError('; '.join(problems))
    video, output = Path(video), Path(output)
    tuning = tuple(options.tuning)
    output.mkdir(parents=True, exist_ok=True)
    from processing.training.train import select_device
    device = select_device(options.device)
    verifier = options.resolved_verifier()
    if options.use_verifier and verifier is None:
        log('No verifier checkpoint found; using Basic Pitch thresholds. '
            'Train one with processing.training.train_verifier.', flush=True)
    # The fusion stage reads this as a module setting. Set it for this run and
    # put it back, so one job's option cannot leak into the next in a worker.
    neutral = fusion.OPEN_AUDIO_NEUTRAL
    if options.open_audio_neutral is not None:
        fusion.OPEN_AUDIO_NEUTRAL = options.open_audio_neutral
    try:
        return _transcribe(video, output, options, audio, log, tuning, device, verifier)
    finally:
        fusion.OPEN_AUDIO_NEUTRAL = neutral


def _transcribe(video, output, options, audio, log, tuning, device, verifier):
    # Name the input up front: the output files are named after the video, so a
    # stale --video with a fresh --output is otherwise silently confusing.
    log(f'Transcribing {video} -> {output}/{video.stem}.*', flush=True)
    log('Detecting pitch...', flush=True)
    with tempfile.TemporaryDirectory() as workdir:
        # --- Step 1: audio -> notes. Only listen for pitches this guitar can
        # play, from the lowest open string up to the highest string's top fret.
        notes, activations, audio_path = notes_from_media(
            video, audio, workdir, onset_threshold=options.onset_threshold,
            frame_threshold=options.frame_threshold, minimum_note_length=options.min_note_ms,
            minimum_frequency=440 * 2 ** ((min(tuning)-69)/12),
            # Basic Pitch rounds the cutoff to a MIDI bin and excludes that bin.
            # Use the next semitone so odd/even top pitches are both retained.
            maximum_frequency=440 * 2 ** ((min(127, max(tuning)+options.max_fret)+1-69)/12),
            verifier=verifier, device=device,
            verifier_threshold=options.verifier_threshold,
            recover_attacks=options.recover_strong_attacks)
        log(f'  {len(notes)} note events'
            f'{" (verifier)" if verifier else " (basic-pitch thresholds)"}', flush=True)

        # --- Step 2: video -> fretboard geometry and fretting hand per frame.
        log('Detecting fretboard and hands...', flush=True)
        records, fps = analyse(video, options.model, device, options)
        log(f'  {len(records)} frames at {fps:.2f} fps', flush=True)
        if not records:
            raise RuntimeError(f'No video frames decoded from {video}')
        # Drop or trim notes that fall outside the decoded video.
        duration = records[-1].timestamp + 1/fps
        notes = [replace(note, end=min(note.end, duration)) for note in notes
                 if 0 <= note.start < duration and note.end > note.start]

        # --- Step 3: notes + vision -> tab. With string order auto, transcribe
        # both ways and keep the order more notes agree with a fingertip.
        choices = ([False, True] if options.string_order == 'auto'
                   else [options.string_order == 'flipped'])
        best, chosen = None, choices[0]
        order_scores = {}
        for flipped in choices:
            assignments, per_group = transcribe_notes(
                records, notes, tuning, options.max_fret, options.inset, flipped, len(tuning),
                audio_weight=options.audio_string_weight,
                visual_priority=options.visual_priority)
            score = fingered_rate(per_group)
            order_scores['flipped' if flipped else 'normal'] = score
            log(f'  string order {"flipped" if flipped else "normal"}: '
                f'{score:.0%} of fretted notes have fingertip support', flush=True)
            if best is None or score > best[0]:
                best, chosen = (score, assignments, per_group), flipped
        assert best is not None  # Every string-order mode evaluates at least one option.
        _, assignments, per_group = best
        transform_kwargs = dict(strings=len(tuning), inset=options.inset, flipped=chosen)

        # --- Step 4: diagnostics, then write the JSONL and overlay video.
        counts = {}
        for note in assignments.values():
            counts[note.support] = counts.get(note.support, 0)+1
        # Fingers on the board that no note explains (hovering, damping, or bad geometry).
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
            'video': str(video.resolve()), 'model': str(Path(options.model).resolve()),
            'frames': len(records), 'fps': fps, 'notes': len(assignments),
            'verifier': str(verifier) if verifier else None,
            'verifier_threshold': (options.verifier_threshold if options.verifier_threshold
                                   is not None else verify.SCORE_THRESHOLD) if verifier else None,
            'audio_string_weight': (options.audio_string_weight
                                    if options.audio_string_weight is not None
                                    else AUDIO_STRING_WEIGHT),
            'open_audio_neutral': fusion.OPEN_AUDIO_NEUTRAL,
            'visual_priority': options.visual_priority,
            'recover_strong_attacks': options.recover_strong_attacks,
            'verifier_suppression': verify.SUPPRESSION_MODE if verifier else None,
            'tuning': list(tuning), 'max_fret': options.max_fret, 'string_inset': options.inset,
            'string_order': 'flipped' if chosen else 'normal',
            'string_order_selected': options.string_order,
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
        jsonl = output/f'{video.stem}.tab.jsonl'
        write_jsonl(jsonl, assignments, summary)
        document = output/f'{video.stem}.performance.json'
        document.write_text(json.dumps(performance(assignments, tuning), indent=1))
        log(json.dumps({k: v for k, v in summary.items()
                        if k not in ('limitations', 'unexplained_fingers')}, indent=2))
        log(f'Notes: {jsonl}', flush=True)

        overlay = None
        if options.write_video:
            log('Rendering overlay...', flush=True)
            silent = output/f'{video.stem}.overlay.silent.mp4'
            render(video, records, assignments, options, fps, transform_kwargs, silent)
            # OpenCV writes video without sound; ffmpeg copies the audio back on.
            overlay = output/f'{video.stem}.overlay.mp4'
            if mux_audio(silent, audio_path, overlay):
                silent.unlink(missing_ok=True)
            else:
                silent.replace(overlay)
                log('Audio mux failed; the overlay is silent.', flush=True)
            log(f'Overlay: {overlay}', flush=True)
    return Transcription(jsonl, document, overlay, summary, assignments)


if __name__ == '__main__':
    main()
