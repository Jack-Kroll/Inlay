"""Simulate a different guitar, amp and recording chain for one waveform.

The verifier's string head reads timbre -- that is the whole reason it beats
musical context. Measured on IDMT-SMT-GUITAR it also reads the *wrong* timbre:
it gained 22.9 points of string accuracy on GuitarSet's held-out player and
only 9.3 on four electric guitars it never saw, and on three of eight IDMT
subsets it was worse than musical context alone. Trained on six acoustic
guitars through one microphone with no augmentation at all, it had no reason
to separate "this is the D string" from "this is that room".

So this builds a random recording chain per excerpt: level, spectral tilt, a
few resonances, a short cabinet impulse, amp saturation, hiss and an MP3
round-trip. What a string sounds like *relative to the other five* survives
all of them; what the guitar and the room sound like does not.

Measured, that is worth 2.7 points of IDMT string accuracy (.607 -> .634) and
3.8 of the string head's own argmax (.578 -> .616), plus 1.4 of note F1, for
two tenths of a point of GuitarSet string accuracy. A harder chain keeps
helping the exist head and starts hurting the string head, which is the
tension this module sits in the middle of; see docs/note-verifier.md.

Two things are deliberately absent. Pitch shifting and time stretching would
move the labels the verifier is trained against -- GuitarSet annotates a
sounding pitch and an onset time, and both heads predict exactly those. And
every impulse response here is direct-dominant and under 40 ms, because a
reverb tail long enough to be audible is also long enough to smear the attack
the exist head is asked to find.

Used by ``processing.tools.variants``, which renders a cache of augmented
posteriorgrams and CQTs; see docs/note-verifier.md.
"""
from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path
import numpy as np

SAMPLE_RATE = 22050

# Ranges are wide enough to cover a bright single-coil through a small combo
# and a dark humbucker direct into an interface, which is roughly the spread
# IDMT-SMT-GUITAR covers and GuitarSet does not.
GAIN_DB = (-12., 6.)
TILT_DB = (-6., 6.)
RESONANCES = (2, 4)
RESONANCE_DB = 6.
RESONANCE_HZ = (80., 6000.)
RESONANCE_Q = (.7, 3.)
DRIVE = (1., 12.)
SNR_DB = (20., 45.)
BITRATE = (64, 96, 128)
# Probability each stage is applied at all, so a run sees clean-ish chains and
# heavily mangled ones rather than only the average of the two.
PROBABILITY = dict(tilt=.9, resonance=.8, cabinet=.6, saturate=.4, hiss=.6,
                   codec=.35)


def _biquad(wave, b, a):
    from scipy.signal import lfilter
    # lfilter returns a tuple only when given initial conditions, which this
    # does not; asarray says so to a type checker as well as copying nothing.
    return np.asarray(lfilter(b, a, wave), np.float32)


def peaking(wave, hz, gain_db, q=1., sample_rate=SAMPLE_RATE):
    """An RBJ peaking EQ: one resonance, as a pickup or a guitar body has."""
    amplitude = 10.**(gain_db/40.)
    omega = 2*np.pi*hz/sample_rate
    alpha = np.sin(omega)/(2*q)
    cosine = np.cos(omega)
    b = [1 + alpha*amplitude, -2*cosine, 1 - alpha*amplitude]
    a = [1 + alpha/amplitude, -2*cosine, 1 - alpha/amplitude]
    return _biquad(wave, np.array(b)/a[0], np.array(a)/a[0])


def shelf(wave, hz, gain_db, high=False, sample_rate=SAMPLE_RATE):
    """An RBJ low or high shelf: everything above or below ``hz`` moves."""
    amplitude = 10.**(gain_db/40.)
    omega = 2*np.pi*hz/sample_rate
    cosine, sine = np.cos(omega), np.sin(omega)
    alpha = sine/2*np.sqrt((amplitude + 1/amplitude)*(1/.9 - 1) + 2)
    root = 2*np.sqrt(amplitude)*alpha
    if high:
        b = [amplitude*((amplitude + 1) + (amplitude - 1)*cosine + root),
             -2*amplitude*((amplitude - 1) + (amplitude + 1)*cosine),
             amplitude*((amplitude + 1) + (amplitude - 1)*cosine - root)]
        a = [(amplitude + 1) - (amplitude - 1)*cosine + root,
             2*((amplitude - 1) - (amplitude + 1)*cosine),
             (amplitude + 1) - (amplitude - 1)*cosine - root]
    else:
        b = [amplitude*((amplitude + 1) - (amplitude - 1)*cosine + root),
             2*amplitude*((amplitude - 1) - (amplitude + 1)*cosine),
             amplitude*((amplitude + 1) - (amplitude - 1)*cosine - root)]
        a = [(amplitude + 1) + (amplitude - 1)*cosine + root,
             -2*((amplitude - 1) + (amplitude + 1)*cosine),
             (amplitude + 1) + (amplitude - 1)*cosine - root]
    return _biquad(wave, np.array(b)/a[0], np.array(a)/a[0])


def tilt(wave, gain_db, sample_rate=SAMPLE_RATE):
    """Broadband brightness: lows down and highs up by ``gain_db``, or back.

    This is the cue a single-coil and a humbucker differ in most, and it is
    also one of the cues that separates a wound string from a plain one. The
    head has to keep the second while losing the first, which is the point.
    """
    wave = shelf(wave, 300., -gain_db, high=False, sample_rate=sample_rate)
    return shelf(wave, 2000., gain_db, high=True, sample_rate=sample_rate)


def cabinet_impulse(rng, sample_rate=SAMPLE_RATE, milliseconds=40.):
    """A short direct-dominant impulse response: a speaker in a small room.

    Real cabinet and close-mic responses are a direct arrival plus a handful
    of early reflections, coloured by the driver. Synthesising one rather than
    shipping a measured library keeps the repo free of impulse responses whose
    licences would have to be tracked, and the model only needs the *variety*.
    """
    length = int(sample_rate*milliseconds/1000)
    impulse = np.zeros(length, np.float32)
    impulse[0] = 1.
    # Early reflections: a few comb notches, which is what moving a microphone
    # a few centimetres actually does to a spectrum.
    for _ in range(rng.integers(2, 5)):
        delay = int(rng.integers(1, length))
        impulse[delay] += rng.uniform(-.5, .5)
    # Driver colouration, as a decaying noise floor under the reflections.
    tail = rng.normal(0, 1, length).astype(np.float32)
    tail *= np.exp(-np.arange(length)/(length/4)).astype(np.float32)
    impulse += .15*tail
    for _ in range(2):
        impulse = peaking(impulse, float(rng.uniform(200., 5000.)),
                          float(rng.uniform(-9., 9.)), float(rng.uniform(.7, 2.)),
                          sample_rate)
    return impulse/max(1e-6, float(np.abs(impulse).max()))


def convolve(wave, impulse):
    """Apply an impulse response without moving the attack.

    ``impulse[0]`` is the direct arrival, so trimming the convolution to the
    original length leaves every onset where it was; only the decay after it
    changes.
    """
    from scipy.signal import fftconvolve
    out = fftconvolve(wave, impulse)[:len(wave)].astype(np.float32)
    return out


def saturate(wave, drive):
    """A tanh waveshaper, level-matched: an amp pushed into its own ceiling.

    Saturation invents harmonics, so it changes the partial structure the
    string head reads more violently than any filter does. That makes it the
    strongest single test of whether the head learned a string or a tone.
    """
    peak = max(1e-6, float(np.abs(wave).max()))
    shaped = np.tanh(drive*wave/peak).astype(np.float32)
    return shaped*(peak/max(1e-6, float(np.abs(shaped).max())))


def hiss(wave, snr_db, pink=False, rng=None):
    """Additive noise at a signal-to-noise ratio, white or pink."""
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, 1, len(wave)).astype(np.float32)
    if pink:
        # A -3 dB/octave slope, close enough for amp and room noise.
        spectrum = np.fft.rfft(noise)
        bins = np.arange(len(spectrum))
        spectrum /= np.sqrt(np.maximum(bins, 1))
        noise = np.fft.irfft(spectrum, len(wave)).astype(np.float32)
    power = float(np.mean(wave**2))
    noise *= np.sqrt(power/10.**(snr_db/10.))/max(1e-9, float(np.std(noise)))
    return (wave + noise).astype(np.float32)


def _realign(original, decoded):
    """Shift ``decoded`` back onto ``original``'s timeline.

    An MP3 encoder prepends its own delay -- around 1,100 samples for LAME,
    which is 50 ms at 22,050 Hz and twice the verifier's onset tolerance. The
    delay is not reliably reported through a decode, so it is measured: the lag
    that maximises cross-correlation over the first few seconds.
    """
    window = min(len(original), 4*SAMPLE_RATE)
    limit = 4096
    padded = np.zeros(window + 2*limit, np.float32)
    head = decoded[:window + limit]
    padded[limit:limit + len(head)] = head
    correlation = np.correlate(padded, original[:window], mode='valid')
    lag = int(np.argmax(correlation)) - limit
    out = np.zeros_like(original)
    if lag >= 0:
        taken = decoded[lag:lag + len(original)]
        out[:len(taken)] = taken
    else:
        taken = decoded[:len(original) + lag]
        out[-lag:-lag + len(taken)] = taken
    return out


def codec(wave, bitrate=96, sample_rate=SAMPLE_RATE):
    """An MP3 round-trip, realigned. Returns ``wave`` unchanged without ffmpeg.

    This one is not only about IDMT. Inlay's own input is a recorded video,
    whose audio has already been through a lossy codec, while every excerpt it
    was trained on is uncompressed WAV.
    """
    import soundfile
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory)/'in.wav'
        compressed = Path(directory)/'out.mp3'
        soundfile.write(str(source), wave, sample_rate)
        try:
            subprocess.run(['ffmpeg', '-v', 'quiet', '-y', '-i', str(source),
                            '-b:a', f'{bitrate}k', str(compressed)], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return wave
        decoded, _ = soundfile.read(str(compressed), dtype='float32', always_2d=False)
    return _realign(wave, np.asarray(decoded, np.float32))


def chain(wave, rng, sample_rate=SAMPLE_RATE, stages=PROBABILITY, strength=1.):
    """One random recording chain over a whole excerpt.

    Sampled once per excerpt, not per frame: a guitar through an amp through a
    microphone is one fixed colouration for the length of a take, and a model
    asked to undo a different one every frame would learn to ignore timbre
    rather than to normalise it.

    ``strength`` scales every filter's decibels and pushes each stage's
    probability towards certainty, so "was the chain simply too gentle?" is a
    question one render can answer rather than a second set of constants.

    Returns (wave, description) so a rendered cache can say what it is.
    """
    applied = []
    harder = lambda chance: 1 - (1 - chance)**strength
    gain = float(rng.uniform(*GAIN_DB))
    wave = (wave*10.**(gain/20.)).astype(np.float32)
    applied.append(f'gain{gain:+.1f}dB')
    if rng.random() < harder(stages['tilt']):
        amount = float(rng.uniform(*TILT_DB))*strength
        wave = tilt(wave, amount, sample_rate)
        applied.append(f'tilt{amount:+.1f}dB')
    if rng.random() < harder(stages['resonance']):
        for _ in range(int(rng.integers(*RESONANCES))):
            hz = float(np.exp(rng.uniform(*np.log(RESONANCE_HZ))))
            gain_db = float(rng.uniform(-RESONANCE_DB, RESONANCE_DB))*strength
            quality = float(rng.uniform(*RESONANCE_Q))
            wave = peaking(wave, hz, gain_db, quality, sample_rate)
            applied.append(f'peak{hz:.0f}Hz{gain_db:+.1f}dB')
    if rng.random() < harder(stages['cabinet']):
        wave = convolve(wave, cabinet_impulse(rng, sample_rate))
        applied.append('cabinet')
    if rng.random() < harder(stages['saturate']):
        drive = 1 + (float(rng.uniform(*DRIVE)) - 1)*strength
        wave = saturate(wave, drive)
        applied.append(f'drive{drive:.1f}')
    if rng.random() < harder(stages['hiss']):
        snr = float(rng.uniform(*SNR_DB))/strength
        wave = hiss(wave, snr, bool(rng.random() < .5), rng)
        applied.append(f'snr{snr:.0f}dB')
    if rng.random() < harder(stages['codec']):
        bitrate = int(rng.choice(BITRATE))
        wave = codec(wave, bitrate, sample_rate)
        applied.append(f'mp3{bitrate}k')
    # Clipping would be a distortion the chain did not ask for, so the result
    # is scaled back under full scale instead of being clamped at it.
    peak = float(np.abs(wave).max())
    if peak > .99:
        wave = (wave*(.99/peak)).astype(np.float32)
    return wave, ' '.join(applied)



# How far apart two strings' levels may be pushed in a remix. Pickup pole
# height, pick attack and string age all move this by a few dB in reality.
STRING_SPREAD_DB = 7.


def remix(stems, rng, spread_db=STRING_SPREAD_DB):
    """Sum per-string stems at random relative levels.

    GuitarSet's debleeded hex pickup records each string on its own channel --
    measured over isolated notes, 98% of a string's energy lands on its own
    channel -- so a mixture built here has string labels that are exact rather
    than inferred. Drawing a fresh balance each time also takes away a cue the
    head could otherwise lean on: on a microphone in front of an acoustic
    guitar the wound strings are simply louder, which says nothing about which
    string a note was played on any other instrument.

    Returns (mix, description).
    """
    gains = rng.uniform(-spread_db, spread_db, len(stems))
    mix = np.zeros(max(len(stem) for stem in stems), np.float32)
    for stem, gain in zip(stems, gains):
        mix[:len(stem)] += (stem*10.**(gain/20.)).astype(np.float32)
    peak = float(np.abs(mix).max())
    if peak > .99:
        mix *= .99/peak
    return mix, 'remix ' + ' '.join(f'{gain:+.1f}' for gain in gains)
