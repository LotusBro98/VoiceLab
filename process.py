import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import erfinv

SAVE_FREQ = 200
FREQ_STEP = 2 ** (1/12)
FREQ_RES = 2
MIN_FREQ = 0
MAX_FREQ = 11000
MEL_N_FEATS = 128

HEAR_SENSE_THRESHOLD = 1e-2


def get_window(n_save, win_size, shift=0):
    window = (torch.arange(n_save) + shift + n_save // 2) % n_save - n_save // 2
    prob_outside = 1e-2
    std = (win_size / 2) / erfinv(1 - prob_outside)
    window = torch.exp(-0.5 * torch.square(window / std))
    window -= window.min()
    window /= window.max()

    # f = torch.fft.fft(window).real
    # plt.plot(window / window.max())
    # plt.plot(f / f.max())
    # plt.show()
    return window


def get_subset(spec_all, fn, win_size, n_save):
    win = get_window(n_save, win_size)
    if len(spec_all.shape) == 2:
        win = win[None, ...]
    win = win.to(spec_all.device)

    fni = int(fn)
    idx_from = torch.arange(fni, fni + n_save)
    idx_from -= n_save // 2
    idx_from %= spec_all.shape[-1]

    idx_to = torch.arange(0, n_save)
    idx_to -= n_save // 2
    idx_to %= n_save

    spec = win * spec_all[..., idx_from[idx_to]]

    return spec


def mel_to_freq(mel):
    if not isinstance(mel, torch.Tensor):
        mel = torch.tensor(mel)
    return 700 * (torch.exp(mel / 1127) - 1)

def freq_to_mel(freq):
    if not isinstance(freq, torch.Tensor):
        freq = torch.tensor(freq)
    return 1127 * torch.log(1 + freq / 700)

def get_mel_n_feats(mel_min, mel_max, fstep, superres):
    mel_mid = (mel_min + mel_max) / 2
    freq_mid = mel_to_freq(mel_mid)
    dmel = mel_mid - freq_to_mel(freq_mid / fstep)
    n_feats = int((mel_max - mel_min) / dmel) * superres

    return n_feats

def get_n_freqs(fmin=MIN_FREQ, fmax=MAX_FREQ, fstep=FREQ_STEP, superres=FREQ_RES):
    if MEL_N_FEATS is not None:
        return MEL_N_FEATS

    mel_min = freq_to_mel(fmin)
    mel_max = freq_to_mel(fmax)

    n_feats = get_mel_n_feats(mel_min, mel_max, fstep, superres)
    return n_feats

def freq_to_sample(freq, fmin=MIN_FREQ, fmax=MAX_FREQ, n_feats=None, fstep=FREQ_STEP, superres=FREQ_RES):
    mel_min = freq_to_mel(fmin)
    mel_max = freq_to_mel(fmax)
    mel_x = freq_to_mel(freq)

    if n_feats is None:
        n_feats = get_mel_n_feats(mel_min, mel_max, fstep, superres)

    fn = (mel_x - mel_min) / (mel_max - mel_min) * n_feats
    return fn


def sample_to_freq(fn, fmin=MIN_FREQ, fmax=MAX_FREQ, n_feats=None, fstep=FREQ_STEP, superres=FREQ_RES):
    mel_min = freq_to_mel(fmin)
    mel_max = freq_to_mel(fmax)

    if n_feats is None:
        n_feats = get_mel_n_feats(mel_min, mel_max, fstep, superres)

    mel_x = fn / n_feats * (mel_max - mel_min) + mel_min
    freq = mel_to_freq(mel_x)

    return freq


def get_mel_scale(fmin=MIN_FREQ, fmax=MAX_FREQ, n_feats=MEL_N_FEATS, fstep=FREQ_STEP, superres=FREQ_RES):
    mel_min = freq_to_mel(fmin)
    mel_max = freq_to_mel(fmax)

    if n_feats is None:
        n_feats = get_mel_n_feats(mel_min, mel_max, fstep, superres)

    mels = torch.linspace(mel_min, mel_max, n_feats)
    freqs = mel_to_freq(mels)

    return freqs


def get_log_scale(fmin, fmax, fstep, superres=1):
    log_min = np.log(fmin)
    log_max = np.log(fmax)
    log_step = np.log(fstep) / superres

    logs = np.arange(log_min, log_max, log_step)
    freqs = np.exp(logs)
    return freqs


def to_freq_diff_repr(spectrum: torch.Tensor) -> torch.Tensor:
    df = spectrum.angle()

    df = df.diff(1, -2, prepend=torch.zeros_like(df[..., :1, :]))
    # df = df.diff(1, -1, prepend=torch.zeros_like(df[..., :, :1]))
    df = torch.atan2(df.sin(), df.cos())
    # df[0, :] = 0

    ampl = spectrum.abs()

    spectrum = (
        ampl *
        torch.exp(1j * df)
    )

    return spectrum


def from_freq_diff_repr(spectrum: torch.Tensor) -> torch.Tensor:
    df = spectrum.angle()

    # df[0, :] = 0
    # df = df.cumsum(-1)
    df = df.cumsum(-2)

    ampl = spectrum.abs()

    spectrum = (
        ampl *
        torch.exp(1j * df)
    )

    return spectrum


def to_bel_scale(spectrum, hear_sense_threshold=HEAR_SENSE_THRESHOLD) -> torch.Tensor:
    ampl = spectrum.abs()
    ampl = torch.log10(1 + ampl / hear_sense_threshold)

    spectrum = (
        ampl *
        torch.exp(1j * spectrum.angle())
    )

    return spectrum


def from_bel_scale(spectrum) -> torch.Tensor:
    ampl = spectrum.abs()
    ampl = HEAR_SENSE_THRESHOLD * (torch.pow(10, ampl) - 1)

    spectrum = (
        ampl *
        torch.exp(1j * spectrum.angle())
    )

    return spectrum


def build_spectrogram(x, sample_rate, fsave=SAVE_FREQ, fmin=MIN_FREQ, fmax=MAX_FREQ):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    x_len = x.shape[-1]

    n_save = int(x_len * fsave / sample_rate)
    spec_all = torch.fft.fft(x, dim=-1)
    spec_all_freq_res = sample_rate / spec_all.shape[-1]
    fn = get_mel_scale(fmin, fmax) / spec_all_freq_res
    df = torch.gradient(fn)[0] * FREQ_RES

    log_spec = []
    for i in range(len(fn)):
        spec = get_subset(spec_all, fn[i], df[i], n_save)
        ampl = torch.fft.ifft(spec, dim=-1)
        log_spec.append(ampl)
    log_spec = torch.stack(log_spec, dim=-1)

    log_spec = to_freq_diff_repr(log_spec)
    log_spec = to_bel_scale(log_spec)

    return log_spec


def set_subset(spec_all, fn, win_size, n_save, spec, weights):
    win = get_window(n_save, win_size).numpy()

    fni = int(fn)
    idx_from = np.arange(fni, fni + n_save)
    idx_from -= n_save // 2
    idx_from %= spec_all.shape[-1]

    idx_to = np.arange(0, n_save)
    idx_to -= n_save // 2
    idx_to %= n_save

    spec_all[idx_from] += spec[idx_to]
    weights[idx_from] += win[idx_to]

    return spec


def generate_sound(spectrum, sample_rate, fsave=SAVE_FREQ, fmin=MIN_FREQ, fmax=MAX_FREQ):
    n_all = int(len(spectrum) * sample_rate / fsave)
    n_save = spectrum.shape[0]

    spec_all_freq_res = sample_rate / n_all

    fn = get_mel_scale(fmin, fmax, fstep=FREQ_STEP, superres=FREQ_RES) / spec_all_freq_res
    df = np.gradient(fn) * FREQ_RES

    spectrum = from_bel_scale(spectrum)
    spectrum = from_freq_diff_repr(spectrum)

    spec_all = np.zeros((n_all,), dtype=np.complex128)
    weights_all = np.zeros((n_all,), dtype=np.complex128)
    
    for i in range(len(fn)):
        ampl = spectrum[:,i]
        spec = np.fft.fft(ampl)
        set_subset(spec_all, fn[i], df[i], n_save, spec, weights_all)
    spec_all[len(spec_all) // 2:] = 0
    spec_all /= weights_all.clip(weights_all.max() / 5, None)
    spec_all[1:] += np.conj(spec_all)[1:][::-1]
    track = np.fft.ifft(spec_all)
    track = np.real(track)
    return track
