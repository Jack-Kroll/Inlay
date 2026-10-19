"""Diff two tab JSONL files: what changed between two transcriptions.

    uv run python -m processing.tools.compare_tab \
      runs/transcribe/blackbird-baseline/Blackbird.tab.jsonl \
      runs/transcribe/blackbird/Blackbird.tab.jsonl

Neither file is ground truth, so this does not say which is right. It says
where they disagree, which is what to look at in the overlay video: a note one
run reports and the other does not, or a note both report and place on
different strings.

Notes are paired by pitch and onset, nearest first, within ``--tolerance``.
A pitch that moved further than that is reported as one dropped note and one
added note, because that is what it looks like to a reader.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

TOLERANCE = .05


def load(path):
    """(summary, notes) from a tab JSONL. The first record is the summary."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f'{path} is empty')
    summary = rows[0] if rows[0].get('kind') != 'note' else {}
    notes = [row for row in rows if row.get('kind') == 'note']
    return summary, notes


def pair(left, right, tolerance=TOLERANCE):
    """Match notes of equal pitch by onset, closest first.

    Greedy nearest-first rather than in order: two runs can report the same
    notes in a different sequence when one splits a note the other merged, and
    pairing by position would then cascade one offset into every later note.
    """
    candidates = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if a['pitch'] != b['pitch']:
                continue
            gap = abs(a['startSeconds'] - b['startSeconds'])
            if gap <= tolerance:
                candidates.append((gap, i, j))
    candidates.sort()
    used_left, used_right, pairs = set(), set(), []
    for _, i, j in candidates:
        if i in used_left or j in used_right:
            continue
        used_left.add(i); used_right.add(j)
        pairs.append((i, j))
    only_left = [i for i in range(len(left)) if i not in used_left]
    only_right = [j for j in range(len(right)) if j not in used_right]
    return pairs, only_left, only_right


def cell(note):
    string, fret = note.get('string'), note.get('fret')
    return 'unassigned' if string is None else f'{string}/{fret}'


def describe(name, note):
    return (f'{note["startSeconds"]:6.2f}s  {note["name"]:<4} '
            f'{cell(note):<10} {note.get("support", "?")}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--tolerance', type=float, default=TOLERANCE,
                        help='Seconds within which two notes of a pitch are the same note')
    parser.add_argument('--list', action='store_true',
                        help='List every differing note, not just the counts')
    parser.add_argument('--limit', type=int, default=12,
                        help='Notes to list per category with --list')
    args = parser.parse_args()

    before_summary, before = load(args.before)
    after_summary, after = load(args.after)
    pairs, dropped, added = pair(before, after, args.tolerance)

    moved = [(i, j) for i, j in pairs if cell(before[i]) != cell(after[j])]
    resupported = [(i, j) for i, j in pairs
                   if before[i].get('support') != after[j].get('support')
                   and cell(before[i]) == cell(after[j])]

    print(f'before: {args.before}  ({len(before)} notes)')
    print(f'after:  {args.after}  ({len(after)} notes)\n')
    print(f'  {len(pairs):>4} notes in both')
    print(f'  {len(dropped):>4} only before (the after run no longer reports them)')
    print(f'  {len(added):>4} only after (new notes)')
    print(f'  {len(moved):>4} in both, placed on a different string or fret')
    print(f'  {len(resupported):>4} in both, same cell, different evidence')

    for label, summary in (('before', before_summary), ('after', after_summary)):
        counts = summary.get('support_counts')
        if counts:
            total = sum(counts.values()) or 1
            parts = ', '.join(f'{k} {v} ({v/total:.0%})'
                              for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
            print(f'\n  {label} support: {parts}')
            rate = summary.get('fingertip_support_rate')
            if rate is not None:
                print(f'  {label} fretted notes under a fingertip: {rate:.3f}')

    if any('audio_string' in (n.get('evidence') or {}) for n in after):
        carried = sum(1 for n in after if 'audio_string' in (n.get('evidence') or {}))
        print(f'\n  {carried}/{len(after)} after-notes carry an audio string posterior')

    if args.list:
        for label, rows, source in (
                ('only before', dropped, before), ('only after', added, after)):
            if rows:
                print(f'\n  {label}:')
                for index in rows[:args.limit]:
                    print(f'    {describe(label, source[index])}')
                if len(rows) > args.limit:
                    print(f'    ... and {len(rows) - args.limit} more')
        if moved:
            print('\n  moved:')
            for i, j in moved[:args.limit]:
                print(f'    {before[i]["startSeconds"]:6.2f}s  {before[i]["name"]:<4} '
                      f'{cell(before[i]):<10} -> {cell(after[j]):<10} '
                      f'({before[i].get("support")} -> {after[j].get("support")})')
            if len(moved) > args.limit:
                print(f'    ... and {len(moved) - args.limit} more')

    flags = Counter(flag for note in after for flag in note.get('flags', ()))
    if flags:
        print('\n  after flags: ' + ', '.join(f'{k} {v}' for k, v in flags.most_common()))


if __name__ == '__main__':
    main()
