"""Basic Pitch's network in PyTorch, so that it can be fine-tuned on guitar.

Basic Pitch ships Keras training code and runs inference through Core ML.
Neither is something the rest of inlay trains, and the 2022 checkpoint has
never been asked to tell a guitar from a piano: every audio decision the
verifier makes is read out of features tuned for MAESTRO, MedleyDB, iKala,
Slakh and one acoustic guitar corpus. This is the same network -- the same
constant-Q transform, harmonic stacking and three small convolutional heads,
about 21k weights -- written in PyTorch and loaded from the released
checkpoint, so the posteriorgram itself can be taught guitar.

Its outputs keep Basic Pitch's format exactly: ``note`` and ``onset`` at
(frames, 88), ``contour`` at (frames, 264), 22,050 Hz input and a 256-sample
hop, windowed and stitched the way ``basic_pitch.inference.run_inference``
does it. Everything downstream -- the verifier, peak-picking, the eval tools
-- reads it unchanged. See docs/front-end-finetune-plan.md.

The released weights are converted once, from the Keras checkpoint inside the
basic-pitch package, by ``processing.tools.convert_basic_pitch``.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import scipy.signal
import torch
from torch import nn
import torch.nn.functional as F

SAMPLE_RATE = 22050
HOP = 256
# Basic Pitch's training and inference window: two seconds, less one hop.
WINDOW = SAMPLE_RATE*2 - HOP
FRAMES = 172
# Basic Pitch's own label and trim rate. The network's true frame rate is
# 22050/256 = 86.13; the package rounds it down, and so does this.
FPS = SAMPLE_RATE // HOP
OVERLAP_FRAMES = 30
NOTE_BINS = 88
CONTOUR_BINS = 264
BINS_PER_SEMITONE = 3
FMIN = 27.5
# CQT geometry: n_harmonics=8 would ask for 124 semitones; Nyquist caps it at 103.
CQT_SEMITONES = 103
CQT_BINS = CQT_SEMITONES*BINS_PER_SEMITONE
BINS_PER_OCTAVE = 12*BINS_PER_SEMITONE
HARMONICS = (.5, 1, 2, 3, 4, 5, 6, 7)
# Keras' BatchNormalization default.
BN_EPSILON = 1e-3

SHIPPED = Path('runs/frontend/basic-pitch.pt')


def _cqt_kernels(q, sample_rate, fmin, bins, bins_per_octave):
    """Time-domain CQT kernels for one octave, as nnAudio's CQT2010v2 builds them."""
    fft_length = 2**int(np.ceil(np.log2(np.ceil(q*sample_rate/fmin))))
    frequencies = fmin*2.**(np.r_[0:bins]/float(bins_per_octave))
    kernels = np.zeros((bins, fft_length), dtype=np.complex64)
    for k, frequency in enumerate(frequencies):
        length = np.ceil(q*sample_rate/frequency)
        start = int(np.ceil(fft_length/2. - length/2.)) - int(length % 2)
        signal = (scipy.signal.get_window('hann', int(length), fftbins=True)
                  * np.exp(np.r_[-length//2:length//2]*1j*2*np.pi*frequency/sample_rate)
                  / length)
        kernels[k, start:start + int(length)] = signal/np.linalg.norm(signal, 1)
    return kernels, fft_length


class CQT(nn.Module):
    """nnAudio's CQT2010v2 with Basic Pitch's settings: one octave of kernels,
    applied to the audio halved eight times. Fixed DSP; nothing here trains."""
    real: torch.Tensor
    imag: torch.Tensor
    lowpass: torch.Tensor
    scale: torch.Tensor
    TAPS = 256

    def __init__(self):
        super().__init__()
        q = 1./(2**(1/BINS_PER_OCTAVE) - 1)
        self.octaves = int(np.ceil(CQT_BINS/BINS_PER_OCTAVE))
        top = FMIN*2**(self.octaves - 1)
        remainder = CQT_BINS % BINS_PER_OCTAVE
        top_max = top*2**(((remainder or BINS_PER_OCTAVE) - 1)/BINS_PER_OCTAVE)
        top = top_max/2**(1 - 1/BINS_PER_OCTAVE)
        kernels, self.fft_length = _cqt_kernels(q, SAMPLE_RATE, top, BINS_PER_OCTAVE,
                                                BINS_PER_OCTAVE)
        frequencies = FMIN*2.**(np.r_[0:CQT_BINS]/float(BINS_PER_OCTAVE))
        lengths = np.ceil(q*SAMPLE_RATE/frequencies)
        lowpass = scipy.signal.firwin2(self.TAPS, [0., .5/1.001, .5*1.001, 1.], [1., 1., 0., 0.])
        self.register_buffer('real', torch.tensor(np.expand_dims(kernels.real, 1),
                                                  dtype=torch.float32))
        self.register_buffer('imag', torch.tensor(np.expand_dims(kernels.imag, 1),
                                                  dtype=torch.float32))
        self.register_buffer('lowpass', torch.tensor(lowpass, dtype=torch.float32).reshape(1, 1, -1))
        self.register_buffer('scale', torch.tensor(np.sqrt(lengths).reshape(1, -1, 1),
                                                   dtype=torch.float32))

    def _octave(self, x, hop):
        padded = F.pad(x, (self.fft_length//2, self.fft_length//2), mode='reflect')
        return (F.conv1d(padded, self.real, stride=hop),
                -F.conv1d(padded, self.imag, stride=hop))

    def forward(self, audio):
        """(batch, samples) -> magnitude (batch, frames, CQT_BINS)."""
        x = audio[:, None]
        hop = HOP
        reals, imags = [], []
        real, imag = self._octave(x, hop)
        reals.append(real)
        imags.append(imag)
        pad = (self.TAPS - 1)//2
        for _ in range(self.octaves - 1):
            hop //= 2
            x = F.conv1d(F.pad(x, (pad, pad)), self.lowpass, stride=2)
            real, imag = self._octave(x, hop)
            reals.insert(0, real)
            imags.insert(0, imag)
        real = torch.cat(reals, 1)[:, -CQT_BINS:]*self.scale
        imag = torch.cat(imags, 1)[:, -CQT_BINS:]*self.scale
        return torch.sqrt(real**2 + imag**2).transpose(1, 2)


def normalized_log(magnitude):
    """Power in dB, rescaled to 0..1 per example, as Basic Pitch's NormalizedLog."""
    power = 10*torch.log10(magnitude**2 + 1e-10)
    low = power.amin(dim=(1, 2), keepdim=True)
    offset = power - low
    high = offset.amax(dim=(1, 2), keepdim=True)
    return torch.where(high > 0, offset/high.clamp_min(1e-12), torch.zeros_like(offset))


def harmonic_stack(x):
    """(batch, 1, frames, CQT_BINS) -> (batch, 8, frames, CONTOUR_BINS): the
    spectrum shifted down by each harmonic, so a fundamental and its partials
    line up in one column."""
    channels = []
    for harmonic in HARMONICS:
        shift = int(np.round(12.*BINS_PER_SEMITONE*np.log2(harmonic)))
        if shift > 0:
            channels.append(F.pad(x[..., shift:], (0, shift)))
        elif shift < 0:
            channels.append(F.pad(x[..., :shift], (-shift, 0)))
        else:
            channels.append(x)
    return torch.cat(channels, 1)[..., :CONTOUR_BINS]


def _same(kernel, stride, size):
    """TensorFlow's SAME padding along one axis, as (before, after)."""
    out = -(-size//stride)
    total = max((out - 1)*stride + kernel - size, 0)
    return total//2, total - total//2


class Strided(nn.Module):
    """A Keras ``Conv2D(padding='same', strides=(1, 3))`` over (frames, bins)."""

    def __init__(self, channels, filters, kernel, bins):
        super().__init__()
        self.conv = nn.Conv2d(channels, filters, kernel, stride=(1, 3))
        time = (kernel[0] - 1)//2
        self.padding = (*_same(kernel[1], 3, bins), time, time)

    def forward(self, x):
        return self.conv(F.pad(x, self.padding))


class BasicPitch(nn.Module):
    """The ICASSP 2022 network, layer for layer. ``forward`` takes Basic
    Pitch's two-second windows and returns its three heads; ``logits=True``
    returns them before the sigmoid, for a numerically stable loss."""

    def __init__(self):
        super().__init__()
        self.cqt = CQT()
        self.input_norm = nn.BatchNorm2d(1, eps=BN_EPSILON)
        self.contour_conv = nn.Conv2d(8, 8, (3, 39), padding=(1, 19))
        self.contour_norm = nn.BatchNorm2d(8, eps=BN_EPSILON)
        self.contour_out = nn.Conv2d(8, 1, (5, 5), padding=(2, 2))
        self.note_conv = Strided(1, 32, (7, 7), CONTOUR_BINS)
        self.note_out = nn.Conv2d(32, 1, (7, 3), padding=(3, 1))
        self.onset_conv = Strided(8, 32, (5, 5), CONTOUR_BINS)
        self.onset_norm = nn.BatchNorm2d(32, eps=BN_EPSILON)
        self.onset_out = nn.Conv2d(33, 1, (3, 3), padding=(1, 1))

    def features(self, audio):
        """The harmonic-stacked, normalised CQT every head starts from."""
        with torch.no_grad():
            spectrum = normalized_log(self.cqt(audio))
        return harmonic_stack(self.input_norm(spectrum[:, None]))

    def forward(self, audio, logits=False):
        x = self.features(audio)
        contour = F.relu(self.contour_norm(self.contour_conv(x)))
        contour = self.contour_out(contour)
        note = self.note_out(F.relu(self.note_conv(torch.sigmoid(contour))))
        onset = F.relu(self.onset_norm(self.onset_conv(x)))
        onset = self.onset_out(torch.cat([torch.sigmoid(note), onset], 1))
        heads = {'note': note[:, 0], 'onset': onset[:, 0], 'contour': contour[:, 0]}
        return heads if logits else {k: torch.sigmoid(v) for k, v in heads.items()}

    @torch.no_grad()
    def renormalize(self):
        """Put every convolution kernel back on the unit sphere, filter by filter.

        Basic Pitch trains under Keras' ``UnitNorm`` constraint, so each output
        filter's kernel has norm 1 and its scale lives in the bias and the
        batch norm that follows. Without the constraint an optimiser's fixed
        step is the size of a whole weight -- they average .03 in the widest
        layer -- and a fine-tune drifts off the released model in a few
        hundred steps. Called after every step, as Keras applies it."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight
                norms = weight.flatten(1).norm(dim=1).clamp_min(1e-12)
                weight.div_(norms[:, None, None, None])

    def freeze_statistics(self):
        """Keep every batch-norm layer on its released running statistics.

        Fine-tuning draws small batches from a few hours of one instrument;
        re-estimating the statistics from that would move every layer's
        operating point before the weights had learnt anything."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self


def keras_state_dict(variables):
    """Map the Keras checkpoint's ``layer_with_weights-N`` arrays onto ``BasicPitch``.

    The order is the order Keras built the layers in models.py. The first
    contour convolution there is dead code -- its output is overwritten before
    use -- so it was never saved and has no place here."""
    def conv(n):
        kernel = variables[f'layer_with_weights-{n}/kernel/.ATTRIBUTES/VARIABLE_VALUE']
        bias = variables[f'layer_with_weights-{n}/bias/.ATTRIBUTES/VARIABLE_VALUE']
        return torch.tensor(np.transpose(kernel, (3, 2, 0, 1)).copy()), torch.tensor(bias)

    def norm(n):
        get = lambda name: torch.tensor(variables[f'layer_with_weights-{n}/{name}/.ATTRIBUTES/VARIABLE_VALUE'])
        return {'weight': get('gamma'), 'bias': get('beta'),
                'running_mean': get('moving_mean'), 'running_var': get('moving_variance')}

    state = {}
    for name, n in (('input_norm', 0), ('contour_norm', 2), ('onset_norm', 6)):
        state.update({f'{name}.{k}': v for k, v in norm(n).items()})
    for name, n in (('contour_conv', 1), ('contour_out', 3), ('note_conv.conv', 4),
                    ('onset_conv.conv', 5), ('note_out', 7), ('onset_out', 8)):
        weight, bias = conv(n)
        state[f'{name}.weight'], state[f'{name}.bias'] = weight, bias
    return state


def load(path=SHIPPED, device=None):
    """A ``BasicPitch`` from a checkpoint written by this module or by
    ``processing.tools.convert_basic_pitch``, in eval mode."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{path} not found; convert the released weights with '
                                '`uv run --with tensorflow==2.16.2 python -m '
                                'processing.tools.convert_basic_pitch`')
    stored = torch.load(path, map_location='cpu', weights_only=True)
    model = BasicPitch()
    model.load_state_dict(stored['model'])
    return model.to(device or 'cpu').eval()


def tag(path):
    """A short name for the front end at ``path``, for cache directories:
    None for Basic Pitch's own Core ML model, otherwise the run directory,
    e.g. ``ft1``, with the checkpoint's stem when it is not ``best``."""
    if path is None:
        return None
    path = Path(path)
    return path.parent.name if path.stem == 'best' else f'{path.parent.name}-{path.stem}'


def cache_dir(root, front_end=None):
    """Where posteriorgrams from ``front_end`` live under a cache root.

    Basic Pitch's own keep the bare ``posteriorgrams`` name, so every cache
    already on disk still resolves; any other front end gets a sibling, so
    two front ends' caches never mix and a verifier trained on one is never
    quietly measured on the other."""
    name = 'posteriorgrams' if front_end is None else f'posteriorgrams@{tag(front_end)}'
    return Path(root)/name


_LOADED = {}


def activations(audio, front_end, device=None):
    """The three heads for an audio file from the checkpoint at ``front_end``,
    loading it once per process."""
    import torch as _torch
    if device is None:
        device = 'mps' if _torch.backends.mps.is_available() else 'cpu'
    key = (str(front_end), str(device))
    if key not in _LOADED:
        _LOADED[key] = load(front_end, device)
    return infer(_LOADED[key], load_audio(audio), device)


def windows(wave):
    """Basic Pitch's inference windows over a mono 22,050 Hz waveform:
    (n, WINDOW) float32, overlapping by OVERLAP_FRAMES frames."""
    overlap = OVERLAP_FRAMES*HOP
    hop = WINDOW - overlap
    padded = np.concatenate([np.zeros(overlap//2, np.float32), np.asarray(wave, np.float32)])
    starts = range(0, len(padded), hop)
    out = np.zeros((len(starts), WINDOW), np.float32)
    for i, start in enumerate(starts):
        chunk = padded[start:start + WINDOW]
        out[i, :len(chunk)] = chunk
    return out


def stitch(output, samples):
    """Undo ``windows`` on one head: drop half the overlap from each side of
    every window, concatenate, and trim to the audio's length."""
    half = OVERLAP_FRAMES//2
    output = output[:, half:-half]
    frames = int(np.floor(samples*(FPS/SAMPLE_RATE)))
    return output.reshape(-1, output.shape[-1])[:frames]


@torch.no_grad()
def infer(model, wave, device=None, batch=32):
    """The three heads for a whole waveform, stitched as ``run_inference``
    stitches them: a dict of float32 arrays."""
    device = device or next(model.parameters()).device
    chunks = windows(wave)
    outputs = {'note': [], 'onset': [], 'contour': []}
    for start in range(0, len(chunks), batch):
        heads = model(torch.from_numpy(chunks[start:start + batch]).to(device))
        for key, value in heads.items():
            outputs[key].append(value.float().cpu().numpy())
    return {key: stitch(np.concatenate(value), len(wave)) for key, value in outputs.items()}


def load_audio(path):
    """Mono 22,050 Hz float32, read the way Basic Pitch reads it."""
    import librosa
    wave, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return np.asarray(wave, np.float32)
