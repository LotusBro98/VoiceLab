import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import erfinv

SAVE_FREQ = 100


def get_window(n_save, win_size):
    win_size = win_size.clip(None, n_save)

    # window = (torch.arange(n_save) + n_save // 2) % n_save - n_save // 2
    # window = window / (win_size / 2)
    # window = torch.sinc(window)

    # window = (torch.arange(n_save) + n_save // 2) % n_save - n_save // 2
    # window = window / (n_save / (win_size / 2))
    # window = torch.cos(np.pi / 2 * window.clip(-1, 1)).square()
    # window = torch.fft.fft(window).real

    # window = (torch.arange(n_save) + n_save // 2) % n_save - n_save // 2
    # window = window / (win_size / 2)
    # window = torch.cos(np.pi / 2 * window.clip(-1, 1)).square()

    # if win_size > n_save:
    #     window = (torch.arange(n_save) + n_save // 2) % n_save - n_save // 2
    #     window = window / (n_save / 2)
    #     window = torch.cos(window * np.pi / 2).square()
    #     window = (window * n_save + win_size - n_save) / (win_size)
    # else:
    #     window = (torch.arange(n_save) + n_save // 2) % n_save - n_save // 2
    #     window = window / (n_save / (win_size / 2))
    #     window = torch.cos(np.pi / 2 * window.clip(-1, 1)).square()
    #     window = torch.fft.fft(window).real
    #     # window = (torch.arange(n_save) + n_save // 2) % n_save - n_save // 2
    #     # prob_outside = 1e-6
    #     # std = (win_size / 2) / erfinv(1 - prob_outside)
    #     # window = torch.exp(-0.5 * torch.square(window / std))

    window = (torch.arange(n_save) + n_save // 2) % n_save - n_save // 2
    prob_outside = 1e-2
    std = (win_size / 2) / erfinv(1 - prob_outside)
    window = torch.exp(-0.5 * torch.square(window / std))
    window -= window.min()

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


def get_mel_scale(fmin, fmax, n_feats):
    def mel_to_freq(mel):
        if not isinstance(mel, torch.Tensor):
            mel = torch.tensor(mel)
        return 700 * (torch.exp(mel / 1127) - 1)

    def freq_to_mel(freq):
        if not isinstance(freq, torch.Tensor):
            freq = torch.tensor(freq)
        return 1127 * torch.log(1 + freq / 700)

    mel_min = freq_to_mel(fmin)
    mel_max = freq_to_mel(fmax)
    mels = torch.linspace(mel_min, mel_max, n_feats + 2)
    freqs = mel_to_freq(mels)

    return freqs


def log_spectrum(x, fs, fsave=SAVE_FREQ, n_feats=80):
    n_save = int(x.shape[-1] * fsave / fs)
    spec_all = torch.fft.fft(x, dim=-1)
    spec_all_freq_res = fs / spec_all.shape[-1]
    fn = get_mel_scale(fsave, fs / 2, n_feats) / spec_all_freq_res
    log_spec = []
    for i in range(len(fn) - 2):
        spec = get_subset(spec_all, fn[i+1], (fn[i+2] - fn[i]), n_save)
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
