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
# Open strings need no hand, so the neck-position preference says nothing about
# them. Give them the neutral value instead of a free maximum.
OPEN_NEARNESS = .5


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


def _hand_position(contacts):
    usable = [c.fret_float for c in contacts if c.on_board]
    return float(np.median(usable)) if usable else None


def _score(string, fret, contacts, position):
    """Return (score, finger, contact term) for one candidate spelling."""
    if fret == 0:
        # An open string is credited when nothing is sitting on that string.
        occupied = max((_string_term(c, string) * (1. if c.on_board else .3)
                        for c in contacts), default=0.)
        contact, finger = OPEN_CREDIT * (1 - occupied), None
    else:
        best, finger = 0., None
        for candidate in contacts:
            value = _string_term(candidate, string) * _fret_term(candidate, fret)
            if candidate.on_board and value > best:
                best, finger = value, candidate.finger
        contact = best
    if position is None:
        return contact, finger, contact
    nearness = (OPEN_NEARNESS if fret == 0 else
                float(np.exp(-.5 * ((fret - position) / POSITION_WIDTH) ** 2)))
    return CONTACT_WEIGHT * contact + POSITION_WEIGHT * nearness, finger, contact


def assign_positions(notes, contacts, tuning=STANDARD_TUNING, max_fret=22,
                     board=True, hand=True):
    """Choose one (string, fret) per simultaneous note.

    No two notes may share a string, since one string sounds one pitch. Notes
    are resolved best-scoring first; a note pushed off its preferred string by a
    louder neighbour is flagged rather than silently respelled.
    """
    results = [TabNote(n.pitch, n.start, n.end, n.amplitude) for n in notes]
    position = _hand_position(contacts)
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
    # Ties favour the lower fret, matching the presentation order of candidates.
    scored.sort(key=lambda row: (-row[0], row[4], row[3]))
    best_choice = {}
    for row in scored:
        best_choice.setdefault(row[2], (row[3], row[4]))
    taken_note, taken_string = set(), set()
    for score, contact, index, string, fret, finger in scored:
        if index in taken_note or string in taken_string:
            continue
        note = results[index]
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
        taken_string.add(string)
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
