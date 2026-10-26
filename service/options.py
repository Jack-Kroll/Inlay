"""What a client may ask of a transcription, and how that becomes ``Options``.

A job's options arrive as JSON from outside, so only a few named fields are
accepted -- the musical ones a player would change, plus the overlay -- and
every model path, device and decoder threshold stays the server's choice.
Field names are camelCase, like the performance document's.
"""
from __future__ import annotations
from dataclasses import replace
import math

from processing.music.options import Options

# JSON field -> (Options field, type check)
FIELDS = {
    'tuning': ('tuning', lambda v: isinstance(v, list) and all(
        isinstance(p, int) and not isinstance(p, bool) for p in v)),
    'maxFret': ('max_fret', lambda v: isinstance(v, int) and not isinstance(v, bool)),
    'stringOrder': ('string_order', lambda v: isinstance(v, str)),
    'verifierThreshold': ('verifier_threshold', lambda v: _number(v)),
    'audioStringWeight': ('audio_string_weight', lambda v: _number(v)),
    'openAudioNeutral': ('open_audio_neutral', lambda v: _number(v)),
    'overlay': ('write_video', lambda v: isinstance(v, bool)),
}


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def options_for(request, settings):
    """``Options`` for one job: the server's settings, then the client's
    fields. Raises ValueError naming every problem."""
    if not isinstance(request, dict):
        raise ValueError('options must be an object')
    unknown = sorted(set(request) - set(FIELDS))
    if unknown:
        raise ValueError(f'Unknown option(s): {", ".join(unknown)}; '
                         f'accepted: {", ".join(FIELDS)}')
    base = Options(model=settings.model, device=settings.device,
                   verifier=settings.verifier, write_video=settings.write_overlay)
    changes = {}
    for name, value in request.items():
        field, valid = FIELDS[name]
        if not valid(value):
            raise ValueError(f'Invalid {name}: {value!r}')
        changes[field] = tuple(value) if name == 'tuning' else value
    options = replace(base, **changes)
    # Paths the server owns are checked when the worker runs, not here: the API
    # may run without the checkpoints present.
    problems = [p for p in replace(options, verifier=None, use_verifier=False).problems()]
    if problems:
        raise ValueError('; '.join(problems))
    return options
