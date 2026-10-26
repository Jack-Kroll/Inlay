"""Combine sounding pitch with fretting-hand geometry into tab positions.

Audio says *which pitch*; the hand says *where on the neck*. Neither alone is
enough: a pitch has several (string, fret) spellings, and a fingertip position
cannot tell a pressed string from a hovering one. The weights below are
heuristics chosen to be legible, not calibrated against annotated playing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .fretboard import DEFAULT_STRING_INSET, fret_ratio, ratio_to_fret
from .hands import FRETTING_FINGERS
from .tuning import STANDARD_TUNING, candidates, note_name

# How far off a cell a fingertip may sit before it stops counting, in strings
# and in frets. Beyond these the term reaches zero.
STRING_TOLERANCE = 1.
FRET_TOLERANCE = 1.
# Width, in frets, of the soft preference for candidates near the hand.
POSITION_WIDTH = 4.
CONTACT_WEIGHT, POSITION_WEIGHT = .75, .25
SUPPORT_THRESHOLD = .35
# How much a per-note string posterior from audio is allowed to move the
# choice, against the geometric evidence it is mixed with. Audio and geometry
# disagree about different things -- audio hears timbre and cannot see the
# neck; the camera sees the neck and cannot hear which string rang -- so the
# two are genuinely independent evidence about the same question. Zero here
# reproduces the geometry-only behaviour exactly.
#
# .25 is the middle of the plateau GuitarSet's validation player shows from .20
# to .35. With a string head that has heard electric guitars (Guitar-TECHS, see
# docs/note-verifier.md) it pays on every labelled recording chain there is:
# string accuracy against the same decode with the term off, on GuitarSet's
# held-out player .658 -> .865 and on Guitar-TECHS's held-out electric .40-.49
# -> .56-.63 whether the microphone is on the amp or across the room.
AUDIO_STRING_WEIGHT = .25
# How far the audio term is kept from arguing a note off an open string, 0 to
# 1; see _score. At 1, audio still chooses between fretted spellings but cannot
# say a note is not open, which is left to the camera and musical context.
#
# The string head cannot hear an open string through a room. On Guitar-TECHS's
# room microphones the audio term takes open-string accuracy *down* (.441 ->
# .305 head-mounted) while taking fretted notes up by more, and on a phone
# recording of Thunderstruck's open-B riff it put 42 of 48 B3 notes on string 3
# fret 4 -- the right pitch, a fingering nobody plays. At 1 the same clip keeps
# 34 open and reaches .721 fingertip support, the camera's independent opinion
# and the highest of any configuration tried; with the term off it was .677.
# Labelled audio pays for that with 0.6 points of string accuracy averaged over
# the electric chains, and 2.6 on GuitarSet's close-miked acoustic, because
# some fretted notes now go to an open string audio could have ruled out. That
# trade was taken knowingly: a misplaced open string is the error a reader
# notices first, and the phone in a room is what this is for.
OPEN_AUDIO_NEUTRAL = 1.
# An unoccupied string is only evidence of absence: a fingertip can be missed or
# occluded. Credit it below a fingertip sitting squarely on a cell, so positive
# evidence always outranks the lack of it.
OPEN_CREDIT = .7
# With no fingertips located at all, "nothing is sitting on that string" is not
# evidence of an open string, it is a blank frame. Any credit above zero makes
# every open spelling outrank every fretted one, whose contact term is also
# zero; measured against GuitarSet, dropping it to zero and letting neck
# position decide raised blind string accuracy from .516 to .558.
BLIND_OPEN_CREDIT = 0.


# MediaPipe's fingertip landmark is the end of the finger, seen from above the
# board; the string is stopped by the pad behind it. On Kaggle's labelled
# stills the fingertip fell past the pressed cell -- towards the body -- for 29%
# of fingers on the development recordings and 73% on the held-out ones, and
# short of it for 0.7% and 9%. A constant distance, which becomes a whole fret
# where frets are narrow. This moves the located point back towards the nut by
# that distance, as a fraction of scale length: 1% is 6.5 mm on a Stratocaster.
#
# Chosen on the development recordings alone, where string accuracy given the
# pitch peaked at .010 (.7949 -> .7967, flat from .008 to .016). Scored once on
# the held-out recordings: .6726 -> .6840, with the fingertip's own fret right
# twice as often (.099 -> .203). Knowing the pitch already rules out most
# wrong strings, so the larger effect is on fingertip support and the planned
# hand position, which read the fret directly. Kaggle is one player on one
# guitar; see docs/alignment.md.
FINGERTIP_SETBACK = .010
# Neck positions are planned on a whole-fret grid; the hand needs no finer
# resolution than that. MOVE_PENALTY is the cost of shifting one fret, against
# an emission worth at most 1 per note, so a shift must pay for itself. Both
# were taken from the marginal peak on a tuning subset, not the grid argmax:
# the surface is flat near the top and the argmax is noise. See
# docs/tab-logic.md.
MOVE_PENALTY = .3
REACH_WIDTH = 1.75
# What an open string is worth on the fret axis. It needs no hand, so it is not
# judged on distance from the fingers -- but scoring it a flat 1.0 would make it
# beat every fretted spelling outright, so it sits below a well-placed finger.
# This is a prior, not a measurement: nothing available to this stage can tell
# an open B from the same pitch fretted elsewhere. It is set where the stage
# stops systematically refusing to say "open" -- at .7 it called open a fifth as
# often as it should have. See docs/tab-logic.md.
OPEN_REACH = .85


def _fret_reach(fret, position):
    """How comfortably a fret is played with the hand at ``position``.

    Fret 0 needs no hand at all, so it is reachable from anywhere rather than
    being penalised for sitting far from the fingers.
    """
    if fret == 0:
        return np.full_like(position, OPEN_REACH)
    return np.exp(-.5 * ((fret - position)/REACH_WIDTH) ** 2)



def plan_positions(groups, observed, tuning=STANDARD_TUNING, max_fret=22):
    """Choose a neck position per onset group, for the whole clip at once.

    Carrying a position forward from the notes just played can only average
    frets already guessed, so an error feeds on itself. Planning the trajectory
    in one pass lets a group be placed by what comes after it as well as what
    came before, and lets groups where the hand was actually seen anchor the
    ones where it was not.

    ``observed`` is the hand position per group, or None where no fingertips
    were located. A seen hand is taken as fact and pins that group; the search
    only fills the gaps between such anchors.

    This is the Viterbi algorithm: states are neck positions (fret 0..max_fret),
    each chord scores every position (emission), and moving the hand between
    chords costs MOVE_PENALTY per fret (transition). Dynamic programming finds
    the best-scoring path through the whole clip.
    """
    if not groups:
        return []
    if len(observed) != len(groups):
        raise ValueError('Expected one observed hand position per onset group')
    # Invalid geometric measurements must not make every path impossible.
    # Ignore them like missing observations; retain valid anchors elsewhere.
    observed = [p if p is not None and np.isfinite(p) and 0 <= p <= max_fret
                else None for p in observed]
    grid = np.arange(max_fret + 1, dtype=float)
    # Emission: how well a group's pitches sit under a hand at each position.
    # For each note take its most reachable spelling, then sum over the chord.
    emissions = []
    for group in groups:
        row = np.zeros(len(grid))
        for note in group:
            options = candidates(note.pitch, tuning, max_fret)
            if not options:
                continue
            row += np.maximum.reduce([_fret_reach(fret, grid) for _, fret in options])
        emissions.append(row)
    # best[i][p] = the position at group i-1 that led to position p at group i.
    best = np.zeros((len(groups), len(grid)), dtype=np.intp)
    score = emissions[0].copy()
    # A seen hand pins its group: every other position becomes impossible.
    if observed[0] is not None:
        score = np.where(grid == round(observed[0]), score, -np.inf)
    # Forward pass: best total score for ending at each position after each group.
    for index in range(1, len(groups)):
        # Cost of arriving at each position from the best predecessor.
        moved = score[:, None] - MOVE_PENALTY*np.abs(grid[:, None] - grid[None, :])
        source = np.argmax(moved, axis=0)
        score = moved[source, np.arange(len(grid))] + emissions[index]
        anchor = observed[index]
        if anchor is not None:
            score = np.where(grid == round(anchor), score, -np.inf)
        best[index] = source
    if not np.isfinite(score).any():
        return [None]*len(groups)
    # Backward pass: start from the best final position and follow the pointers.
    positions = [0]*len(groups)
    positions[-1] = int(np.argmax(score))
    for index in range(len(groups) - 1, 0, -1):
        positions[index - 1] = int(best[index][positions[index]])
    return [float(grid[index]) for index in positions]


@dataclass
class FingerContact:
    """One fingertip located on the board."""
    finger: str          # 'index', 'middle', 'ring' or 'pinky'.
    point: np.ndarray    # Image pixels.
    fret: int            # The fret this finger would sound.
    string: int          # Nearest string (1 = highest).
    fret_float: float    # Continuous board position, e.g. 4.3 = just past wire 4.
    string_float: float
    on_board: bool       # False if the fingertip is off the string band or neck.
    # Where the fingertip itself was, before FINGERTIP_SETBACK; None on contacts
    # built before the setback existed, whose fret_float is the fingertip.
    tip_fret: float | None = None

    @property
    def label(self):
        return f'{self.finger[0].upper()}{self.string}/{self.fret}'


@dataclass
class TabNote:
    """A detected pitch plus where on the guitar we think it was played.

    ``support`` records the evidence behind the placement: 'fingered' (a
    fingertip is on that cell), 'open', 'position-only' (just the planned neck
    position), 'no-hand', or 'no-board'.
    """
    pitch: int
    start: float
    end: float
    amplitude: float
    string: int | None = None
    fret: int | None = None
    support: str = 'none'
    confidence: float = 0.
    finger: str | None = None
    alternatives: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    @property
    def name(self):
        return note_name(self.pitch)


@dataclass
class Sounded:
    """The minimum the tab stage needs from a note: a pitch with a lifetime.

    ``assign_positions`` reads ``pitch``, ``start``, ``end``, ``amplitude`` and
    an optional ``string_posterior``, and nothing else. In the pipeline that is
    a ``processing.music.pitch.NoteEvent``; this is the stand-in for callers
    that have a pitch and no audio stage behind it -- the GuitarSet and IDMT
    evaluations feeding annotated notes in, and the tests.
    """
    pitch: int
    start: float
    end: float
    amplitude: float = 1.
    string_posterior: np.ndarray | None = None


def place_contact(finger, point, tip_fret, string_float, strings=6, max_fret=22, margin=.6,
                  setback=None):
    """A ``FingerContact`` for a fingertip seen at continuous fret ``tip_fret``,
    moved ``setback`` of the scale length back towards the nut first."""
    setback = FINGERTIP_SETBACK if setback is None else setback
    fret_float = float(ratio_to_fret(fret_ratio(tip_fret) - setback))
    # A finger between wires 4 and 5 (fret_float 4.3) sounds fret 5, so round up.
    fret = int(np.ceil(fret_float - 1e-9))
    string = int(round(string_float))
    on_board = (0 < fret <= max_fret and 1 - margin <= string_float <= strings + margin)
    return FingerContact(finger, np.asarray(point, float), max(fret, 0),
                         min(max(string, 1), strings), fret_float, float(string_float),
                         on_board, float(tip_fret))


def contact_from_cache(row, strings=6, max_fret=22, setback=None):
    """Rebuild a cached contact under the current setback. Caches written
    before ``tip_fret`` existed hold the fingertip itself in ``fret_float``."""
    tip = row.get('tip_fret')
    tip = row['fret_float'] if tip is None else tip
    return place_contact(row['finger'], row['point'], tip, row['string_float'], strings,
                         max_fret, setback=setback)


def finger_contacts(hand, transform, fingers=FRETTING_FINGERS, max_fret=22, margin=.6):
    """Locate fingertips on the board. Off-board fingertips are kept but marked."""
    if hand is None or transform is None:
        return []
    names = list(fingers)
    points = np.array([hand.fingertip(name) for name in names], float)
    frets, strings = transform.locate(points)
    contacts = []
    for name, point, fret_float, string_float in zip(names, points, frets, strings):
        if not np.isfinite(fret_float) or not np.isfinite(string_float):
            continue
        contacts.append(place_contact(name, point, float(fret_float), float(string_float),
                                      transform.strings, max_fret, margin))
    return contacts


def _string_term(contact, string):
    """1 when the fingertip is exactly on ``string``, falling linearly to 0 one string away."""
    return max(0., 1 - abs(contact.string_float - string) / STRING_TOLERANCE)


def _fret_term(contact, fret):
    """1 inside the fret's cell, falling linearly to 0 one fret outside it."""
    # Anywhere inside the cell is a full match; the cell spans fret-1 to fret.
    inside = fret - 1 <= contact.fret_float <= fret
    if inside:
        return 1.
    gap = (fret - 1) - contact.fret_float if contact.fret_float < fret - 1 else contact.fret_float - fret
    return max(0., 1 - gap / FRET_TOLERANCE)


def hand_position(contacts):
    """Where the hand is on the neck: median fret of on-board fingertips, or None."""
    usable = [c.fret_float for c in contacts if c.on_board]
    return float(np.median(usable)) if usable else None


def _score(string, fret, contacts, position, posterior=None,
           audio_weight=AUDIO_STRING_WEIGHT, rival=None, fret_weight=0.):
    """Return (score, finger, contact term) for one candidate spelling.

    geometry = CONTACT_WEIGHT * (is a fingertip on this cell?)
             + POSITION_WEIGHT * (is this fret near where the hand is?)
    score    = (1 - audio_weight)*geometry + audio_weight*posterior[string]

    ``posterior`` is what audio thinks of each string for this note, highest
    string first, or None when audio has no opinion. It is mixed in rather than
    added so the scale stays comparable across notes that have it and notes
    that do not.
    """
    if fret == 0:
        # An open string is credited when nothing is sitting on that string.
        occupied = max((_string_term(c, string) * (1. if c.on_board else .3)
                        for c in contacts), default=0.)
        credit = OPEN_CREDIT if contacts else BLIND_OPEN_CREDIT
        contact, finger = credit * (1 - occupied), None
    else:
        # A fretted note is credited by the best-matching on-board fingertip.
        best, finger = 0., None
        for candidate in contacts:
            value = _string_term(candidate, string) * _fret_term(candidate, fret)
            if candidate.on_board and value > best:
                best, finger = value, candidate.finger
        contact = best
    # A fingertip's fret coordinate can be useful even when its interpolated
    # string coordinate is uncertain. Keep the original contact value for
    # support labels: fret-only agreement is not proof of a finger on a string.
    geometric_contact = contact
    if fret > 0 and fret_weight:
        geometric_contact = max(
            (_fret_term(c, fret) * (fret_weight + (1 - fret_weight)*_string_term(c, string))
             for c in contacts if c.on_board), default=0.)
    if position is None:
        geometry = geometric_contact
    else:
        # Fret 0 needs no hand, so it is not judged on how near the fingers are.
        nearness = (OPEN_REACH if fret == 0 else
                    float(np.exp(-.5 * ((fret - position) / POSITION_WIDTH) ** 2)))
        geometry = CONTACT_WEIGHT * geometric_contact + POSITION_WEIGHT * nearness
    if posterior is None or not audio_weight:
        return geometry, finger, contact
    heard = float(posterior[string - 1]) if 0 < string <= len(posterior) else 0.
    if fret == 0 and OPEN_AUDIO_NEUTRAL and rival is not None:
        # At 1, audio may say which string a fretted note is on but not whether
        # it is open: the open spelling is credited with what audio gave its
        # best fretted rival, so the two differ by geometry alone, exactly as
        # with the term off. Crediting it with the posterior's maximum instead
        # would favour open whenever audio was unsure.
        heard += OPEN_AUDIO_NEUTRAL*(rival - heard)
    return (1 - audio_weight)*geometry + audio_weight*heard, finger, contact


def _match_strings(scored, strings, amplitudes=None):
    """Choose one string per note so the whole onset scores highest together.

    Walking the spellings best-first gives each note the best string still
    free, which is optimal for that note and not for the chord: one confident
    mistake takes a string away from the note that owned it, that note takes
    the next, and a single wrong nudge can respell a whole voicing. Deciding
    the onset in one go makes the mistake pay for the disruption it causes, so
    it has to outscore the rest of the chord rather than just the note it
    lands on.

    This is exact rather than another heuristic. The only thing the notes
    before this one leave behind is *which strings they used*, and a tuning
    has at most twelve, so a dynamic program over the subsets of strings
    already taken finds the best total in well under a millisecond.

    Returns ``{note index: (string, fret)}``, leaving out any note it could
    not place -- an onset with more notes than strings -- for the caller to
    resolve one at a time.
    """
    # The best spelling of each note on each string. A pitch sounds at one
    # fret per string, so this only drops ties, never real alternatives.
    best = {}
    for score, _, index, string, fret, _ in scored:
        key = (index, string)
        if key not in best or score > best[key][0]:
            best[key] = (score, fret)
    order = sorted({index for index, _ in best})
    # state[mask] = (total score, notes placed, amplitude, predecessor) over the
    # notes considered so far, where ``mask`` is the set of strings they used.
    # More notes placed breaks a tie, so a note is never dropped for nothing.
    state = {0: (0., 0, 0., (0, None))}
    history = []
    for index in order:
        following = {}
        for mask, (total, placed, amplitude, _) in state.items():
            step = (total, placed, amplitude, (mask, None))   # Leave unplaced.
            if mask not in following or step[:3] > following[mask][:3]:
                following[mask] = step
            for string in range(1, strings + 1):
                bit = 1 << (string - 1)
                found = best.get((index, string))
                if found is None or mask & bit:
                    continue
                step = (total + found[0], placed + 1,
                        amplitude + (amplitudes[index] if amplitudes is not None else 0.),
                        (mask, (string, found[1])))
                taken = mask | bit
                if taken not in following or step[:3] > following[taken][:3]:
                    following[taken] = step
        history.append(following)
        state = following
    chosen = {}
    if not history:
        return chosen
    mask = max(state, key=lambda key: state[key][:3])
    for index, step in zip(reversed(order), reversed(history)):
        _, _, _, (previous, taken) = step[mask]
        if taken is not None:
            chosen[index] = taken
        mask = previous
    return chosen


def assign_positions(notes, contacts, tuning=STANDARD_TUNING, max_fret=22,
                     board=True, hand=True, occupied=(), position=None,
                     audio_weight=AUDIO_STRING_WEIGHT, fret_weight=0., visual_evidence=None):
    """Choose one (string, fret) per simultaneous note.

    No two notes may share a string, since one string sounds one pitch. The
    onset is solved as a whole rather than note by note (see
    ``_match_strings``); a note pushed off its preferred string by the rest of
    the chord is flagged rather than silently respelled.

    ``occupied`` holds notes still ringing from earlier onsets. Those do not
    block the string: plucking a string again is what stops the note already on
    it, so the earlier note is truncated to the new onset instead. Only notes
    sounded together compete for a string. In GuitarSet only 3 of 60,670
    consecutive same-string pairs overlap at all, so a ringing note that appears
    to collide with a later one is a re-pluck, not a conflict.
    """
    if not 0 <= fret_weight <= 1:
        raise ValueError('fret_weight must be between 0 and 1')
    if visual_evidence is not None and len(visual_evidence) != len(notes):
        raise ValueError('Expected one visual evidence entry per note')
    results = [TabNote(n.pitch, n.start, n.end, n.amplitude) for n in notes]
    # What audio thought of each string, per note, read off the incoming notes
    # rather than the results built from them: only the caller's notes carry it.
    posteriors = [getattr(note, 'string_posterior', None) for note in notes]
    # A seen hand always wins; ``position`` is the caller's fallback for when
    # the hand is missing, so the neck does not appear to teleport per chord.
    seen = hand_position(contacts)
    position = seen if seen is not None else position
    # Score every possible (note, string, fret) combination in the chord.
    scored = []
    for index, note in enumerate(results):
        options = candidates(note.pitch, tuning, max_fret)
        note.alternatives = list(options)
        if not options:
            note.flags.append('out-of-range')
            continue
        posterior = posteriors[index]
        effective_weight = audio_weight
        proof = visual_evidence[index] if visual_evidence is not None else None
        if proof is not None and posterior is not None and audio_weight and board and hand:
            # Neighbors may confirm the current frame, never override a
            # disagreeing current geometry or manufacture a pressed string.
            geometric = [(_score(s, f, contacts, position)[0], s, f) for s, f in options]
            _, best_string, best_fret = max(geometric)
            if (best_string, best_fret) == (proof['string'], proof['fret']):
                effective_weight = 0.
                note.evidence['visual_priority'] = dict(proof, audio_weight_used=0.)
        # What audio gave the best fretted spelling of this note, which an open
        # spelling is credited with; see OPEN_AUDIO_NEUTRAL.
        fretted = [float(posterior[string - 1]) for string, fret in options
                   if fret > 0 and posterior is not None and 0 < string <= len(posterior)]
        rival = max(fretted) if fretted else None
        for string, fret in options:
            score, finger, contact = _score(string, fret, contacts, position, posterior,
                                            effective_weight, rival, fret_weight=fret_weight)
            scored.append((score, contact, index, string, fret, finger))
    # Equal scores favour stronger audio activations, then lower-fret spellings.
    scored.sort(key=lambda row: (-row[0], -results[row[2]].amplitude, row[4], row[3]))
    # Each note's favourite spelling, to detect when it gets bumped off it.
    best_choice = {}
    for row in scored:
        best_choice.setdefault(row[2], (row[3], row[4]))
    # Solve the onset as a whole, then apply what it decided. Its spellings go
    # first: a note it could not place must not take a string it promised to
    # one it did, and falls back to the best free spelling left.
    chosen = _match_strings(scored, len(tuning), [n.amplitude for n in results])
    ordered = ([row for row in scored if chosen.get(row[2]) == (row[3], row[4])]
               + [row for row in scored if row[2] not in chosen])
    taken_note = set()
    simultaneous = {id(note) for note in results}
    sounding = list(occupied)
    for score, contact, index, string, fret, finger in ordered:
        note = results[index]
        if index in taken_note:
            continue
        clash = [other for other in sounding if other.string == string
                 and note.start < other.end and other.start < note.end]
        if any(id(other) in simultaneous for other in clash):
            continue
        for other in clash:
            other.end = max(other.start, note.start)
            if 'stopped-by-repluck' not in other.flags:
                other.flags.append('stopped-by-repluck')
        note.string, note.fret, note.confidence, note.finger = string, fret, float(score), finger
        if posteriors[index] is not None and audio_weight:
            note.evidence['audio_string'] = [round(float(value), 3)
                                             for value in posteriors[index]]
        # Label what evidence backed the choice (see TabNote.support).
        # Only a genuine displacement counts: this note's own best-scoring
        # spelling was already claimed by a different note.
        if best_choice[index] != (string, fret):
            note.flags.append('string-taken')
        if not board:
            note.support = 'no-board'
        elif not hand:
            note.support = 'no-hand'
        elif contact >= SUPPORT_THRESHOLD:
            note.support = 'open' if fret == 0 else 'fingered'
        elif position is not None:
            note.support = 'position-only'
            note.flags.append('unsupported-placement')
        else:
            note.support = 'no-hand'
        taken_note.add(index)
        sounding.append(note)
    for note in results:
        if note.string is None and 'out-of-range' not in note.flags:
            note.flags.append('no-free-string')
    return results


def unexplained_contacts(contacts, assigned, threshold=SUPPORT_THRESHOLD):
    """Fingertips on the board that no sounding note accounts for.

    Usually a finger damping, hovering, or mid-shift; sometimes it means the
    string geometry is wrong. Reported rather than hidden.
    """
    used = [(note.string, note.fret) for note in assigned
            if note.string is not None and note.fret]
    return [c for c in contacts if c.on_board
            and not any(_string_term(c, s) * _fret_term(c, f) >= threshold
                        for s, f in used)]
