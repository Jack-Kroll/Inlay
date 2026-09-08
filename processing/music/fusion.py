"""Combine sounding pitch with fretting-hand geometry into tab positions.

Audio says *which pitch*; the hand says *where on the neck*. Neither alone is
enough: a pitch has several (string, fret) spellings, and a fingertip position
cannot tell a pressed string from a hovering one. The weights below are
heuristics chosen to be legible, not calibrated against annotated playing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .fretboard import DEFAULT_STRING_INSET
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
    """
    if not groups:
        return []
    grid = np.arange(max_fret + 1, dtype=float)
    # Emission: how well a group's pitches sit under a hand at each position.
    emissions = []
    for group in groups:
        row = np.zeros(len(grid))
        for note in group:
            options = candidates(note.pitch, tuning, max_fret)
            if not options:
                continue
            row += np.maximum.reduce([_fret_reach(fret, grid) for _, fret in options])
        emissions.append(row)
    best = np.zeros((len(groups), len(grid)), dtype=np.intp)
    score = emissions[0].copy()
    if observed[0] is not None:
        score = np.where(grid == round(observed[0]), score, -np.inf)
    for index in range(1, len(groups)):
        # Cost of arriving at each position from the best predecessor.
        moved = score[:, None] - MOVE_PENALTY*np.abs(grid[:, None] - grid[None, :])
        source = np.argmax(moved, axis=0)
        score = moved[source, np.arange(len(grid))] + emissions[index]
        if observed[index] is not None:
            score = np.where(grid == round(observed[index]), score, -np.inf)
        best[index] = source
    if not np.isfinite(score).any():
        return [None]*len(groups)
    positions = [0]*len(groups)
    positions[-1] = int(np.argmax(score))
    for index in range(len(groups) - 1, 0, -1):
        positions[index - 1] = int(best[index][positions[index]])
    return [float(grid[index]) for index in positions]


@dataclass
class FingerContact:
    finger: str
    point: np.ndarray
    fret: int
    string: int
    fret_float: float
    string_float: float
    on_board: bool

    @property
    def label(self):
        return f'{self.finger[0].upper()}{self.string}/{self.fret}'


@dataclass
class TabNote:
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

    @property
    def name(self):
        return note_name(self.pitch)


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
        fret = int(np.ceil(fret_float - 1e-9))
        string = int(round(string_float))
        on_board = (0 < fret <= max_fret
                    and 1 - margin <= string_float <= transform.strings + margin)
        contacts.append(FingerContact(name, point, max(fret, 0),
                                      min(max(string, 1), transform.strings),
                                      float(fret_float), float(string_float), on_board))
    return contacts


def _string_term(contact, string):
    return max(0., 1 - abs(contact.string_float - string) / STRING_TOLERANCE)


def _fret_term(contact, fret):
    # Anywhere inside the cell is a full match; the cell spans fret-1 to fret.
    inside = fret - 1 <= contact.fret_float <= fret
    if inside:
        return 1.
    gap = (fret - 1) - contact.fret_float if contact.fret_float < fret - 1 else contact.fret_float - fret
    return max(0., 1 - gap / FRET_TOLERANCE)


def hand_position(contacts):
    usable = [c.fret_float for c in contacts if c.on_board]
    return float(np.median(usable)) if usable else None


def _score(string, fret, contacts, position):
    """Return (score, finger, contact term) for one candidate spelling."""
    if fret == 0:
        # An open string is credited when nothing is sitting on that string.
        occupied = max((_string_term(c, string) * (1. if c.on_board else .3)
                        for c in contacts), default=0.)
        credit = OPEN_CREDIT if contacts else BLIND_OPEN_CREDIT
        contact, finger = credit * (1 - occupied), None
    else:
        best, finger = 0., None
        for candidate in contacts:
            value = _string_term(candidate, string) * _fret_term(candidate, fret)
            if candidate.on_board and value > best:
                best, finger = value, candidate.finger
        contact = best
    if position is None:
        return contact, finger, contact
    # Fret 0 needs no hand, so it is not judged on how near the fingers are.
    nearness = (OPEN_REACH if fret == 0 else
                float(np.exp(-.5 * ((fret - position) / POSITION_WIDTH) ** 2)))
    return CONTACT_WEIGHT * contact + POSITION_WEIGHT * nearness, finger, contact


def assign_positions(notes, contacts, tuning=STANDARD_TUNING, max_fret=22,
                     board=True, hand=True, occupied=(), position=None):
    """Choose one (string, fret) per simultaneous note.

    No two notes may share a string, since one string sounds one pitch. Notes
    are resolved best-scoring first; a note pushed off its preferred string by a
    stronger audio event is flagged rather than silently respelled.

    ``occupied`` holds notes still ringing from earlier onsets. Those do not
    block the string: plucking a string again is what stops the note already on
    it, so the earlier note is truncated to the new onset instead. Only notes
    sounded together compete for a string. In GuitarSet only 3 of 60,670
    consecutive same-string pairs overlap at all, so a ringing note that appears
    to collide with a later one is a re-pluck, not a conflict.
    """
    results = [TabNote(n.pitch, n.start, n.end, n.amplitude) for n in notes]
    # A seen hand always wins; ``position`` is the caller's fallback for when
    # the hand is missing, so the neck does not appear to teleport per chord.
    seen = hand_position(contacts)
    position = seen if seen is not None else position
    scored = []
    for index, note in enumerate(results):
        options = candidates(note.pitch, tuning, max_fret)
        note.alternatives = list(options)
        if not options:
            note.flags.append('out-of-range')
            continue
        for string, fret in options:
            score, finger, contact = _score(string, fret, contacts, position)
            scored.append((score, contact, index, string, fret, finger))
    # Equal scores favour stronger audio activations, then lower-fret spellings.
    scored.sort(key=lambda row: (-row[0], -results[row[2]].amplitude, row[4], row[3]))
    best_choice = {}
    for row in scored:
        best_choice.setdefault(row[2], (row[3], row[4]))
    taken_note = set()
    simultaneous = {id(note) for note in results}
    sounding = list(occupied)
    for score, contact, index, string, fret, finger in scored:
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


def orientation_support(frames):
    """Fraction of notes with direct finger support, used to pick string order."""
    total = sum(len(notes) for notes in frames)
    if not total:
        return 0.
    good = sum(1 for notes in frames for n in notes if n.support in ('fingered', 'open'))
    return good / total
