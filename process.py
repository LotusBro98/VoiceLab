import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import erfinv

SAVE_FREQ = 1000
FREQ_STEP = 2 ** (1/12)
FREQ_RES = 10
MIN_FREQ = 0
MAX_FREQ = 11000


def get_window(n_save, win_size):
    window = (torch.arange(n_save) + n_save // 2) % n_save - n_save // 2
    prob_outside = 1e-2
    std = (win_size / 2) / erfinv(1 - prob_outside)
    window = torch.exp(-0.5 * torch.square(window / std))
    # window -= window.min()

    # window *= np.clip(win_size / n_save, 1, None)

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


def freq_to_sample(freq, fmin, fmax, n_feats=None, fstep=FREQ_STEP, superres=FREQ_RES):
    mel_min = freq_to_mel(fmin)
    mel_max = freq_to_mel(fmax)
    mel_x = freq_to_mel(freq)

    if n_feats is None:
        mel_mid = (mel_min + mel_max) / 2
        freq_mid = mel_to_freq(mel_mid)
        dmel = mel_mid - freq_to_mel(freq_mid / fstep)
        n_feats = int((mel_max - mel_min) / dmel) * superres

    fn = (mel_x - mel_min) / (mel_max - mel_min) * n_feats
    return fn


def get_mel_scale(fmin, fmax, n_feats=None, fstep=FREQ_STEP, superres=FREQ_RES):
    mel_min = freq_to_mel(fmin)
    mel_max = freq_to_mel(fmax)

    if n_feats is None:
        mel_mid = (mel_min + mel_max) / 2
        freq_mid = mel_to_freq(mel_mid)
        dmel = mel_mid - freq_to_mel(freq_mid / fstep)
        n_feats = int((mel_max - mel_min) / dmel) * superres

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


def get_dfreq(freq_scale, superres=1):
    dfreq = freq_scale[2:] - freq_scale[:-2]
    np.gradient(freq_scale)


def log_spectrum(x, sample_rate, fsave=SAVE_FREQ, fmin=MIN_FREQ, fmax=MAX_FREQ):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    x_len = x.shape[-1]

    n_save = int(x_len * fsave / sample_rate)
    spec_all = torch.fft.fft(x, dim=-1)
    spec_all_freq_res = sample_rate / spec_all.shape[-1]
    fn = get_mel_scale(fmin, fmax, fstep=FREQ_STEP, superres=FREQ_RES) / spec_all_freq_res
    df = np.gradient(fn) * FREQ_RES

    log_spec = []
    for i in range(len(fn)):
        spec = get_subset(spec_all, fn[i], df[i], n_save)
        ampl = torch.fft.ifft(spec, dim=-1)
        log_spec.append(ampl)
    log_spec = torch.stack(log_spec, dim=-1)
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


def generate(spectrum, fs, fsave=SAVE_FREQ):
    n_all = int(len(spectrum) * fs / fsave)
    n_save = spectrum.shape[0]
    spec_all = np.zeros((n_all,), dtype=np.complex128)
    spec_all_freq_res = fs / spec_all.shape[-1]
    n_feats = spectrum.shape[-1]
    weights_all = np.zeros((n_all,), dtype=np.complex128)
    fn = get_mel_scale(fsave, fs / 2, n_feats) / spec_all_freq_res
    for i in range(len(fn) - 2):
        ampl = spectrum[:,i]
        spec = np.fft.fft(ampl)
        set_subset(spec_all, fn[i+1], (fn[i+2] - fn[i]), n_save, spec, weights_all)
    spec_all[len(spec_all) // 2:] = 0
    spec_all /= weights_all.clip(weights_all.max() / 5, None)
    spec_all[1:] += np.conj(spec_all)[1:][::-1]
    track = np.fft.ifft(spec_all)
    track = np.real(track)
    return track
