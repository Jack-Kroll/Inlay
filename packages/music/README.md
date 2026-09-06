# Shared music contract

`performance.schema.json` is a versioned JSON Schema for an initial timed performance.
Both Python and the future TypeScript app can consume it. It is deliberately
smaller than a full score model: it does not yet represent measures, tempo maps,
voices, spelling, ties, bends, slides, or repeats.

- Times are seconds from the beginning of the source media.
- Pitch is MIDI sounding pitch; guitar notation's octave convention is a display concern.
- String numbers start at 1, highest string first in standard tuning.
- Tuning entries are open-string MIDI pitches in that same order.
- String and fret are optional together: unassigned pitches remain editable.
- Confidence is optional; it must not be invented for user-entered notes.

JSON Schema checks structure. Application logic must also check string bounds
against tuning and, for ordinary unbent notes, pitch = tuning[string - 1] + fret.
Do not apply that equality to future bend/harmonic events without modeling them.

Next: define a separate metrical score representation and explicit conversion
from performance time to beats. Do not silently round seconds into notation.
