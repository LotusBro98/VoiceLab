import numpy as np

import notes

FREQ_STEP = np.power(2, 1 / 12 / 3)
WINDOW_FREQ_STEP = np.power(2, 1 / 12)
MIN_FREQ = notes.NOTES["C1"]
SAVE_FREQ = 256


def get_window(n_save, win_size):
    window = (np.arange(n_save) + n_save // 2) % n_save - n_save // 2
    std = win_size / 3
    window = np.exp(-0.5 * np.square(window / std))
    return window


def get_subset(spec_all, fn, fstep, n_save):
    win_size = fn * (fstep - 1)
    spec = get_window(n_save, win_size)
    spec = np.complex64(spec)
    
    fni = int(fn)
    idx_from = np.arange(fni, fni + n_save)
    idx_from -= n_save // 2
    idx_from %= spec_all.shape[-1]

    idx_to = np.arange(0, n_save)
    idx_to -= n_save // 2
    idx_to %= n_save

    spec[idx_to] *= spec_all[idx_from]

    return spec


def log_spectrum(x, fs, fstep=FREQ_STEP, win_fstep=WINDOW_FREQ_STEP, fmin=MIN_FREQ, fsave=SAVE_FREQ):
    n_save = int(len(x) * fsave / fs)
    spec_all = np.fft.fft(x)
    nmax = int(np.ceil(np.log(fs / 2 / fmin) / np.log(fstep)))
    fn = len(x) / fs * fmin * np.power(fstep, np.arange(0, nmax + 1))
    log_spec = []
    for i in range(len(fn)):
        spec = get_subset(spec_all, fn[i], win_fstep, n_save)
        ampl = np.fft.ifft(spec)
        log_spec.append(ampl)
    log_spec = np.stack(log_spec, axis=-1)
    return log_spec


def set_subset(spec_all, fn, fstep, n_save, spec, weights):
    win_size = fn * (fstep - 1)
    win = get_window(n_save, win_size)

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


def generate(spectrum, fs, fstep=FREQ_STEP, win_fstep=WINDOW_FREQ_STEP, fmin=MIN_FREQ, fsave=SAVE_FREQ):
    n_all = int(len(spectrum) * fs / fsave)
    n_save = spectrum.shape[0]
    spec_all = np.zeros((n_all,), dtype=np.complex128)
    weights_all = np.zeros((n_all,), dtype=np.complex128)
    nmax = int(np.ceil(np.log(fs / 2 / fmin) / np.log(fstep)))
    fn = n_all / fs * fmin * np.power(fstep, np.arange(0, nmax + 1))
    for i in range(len(fn) - 1):
        ampl = spectrum[:,i]
        spec = np.fft.fft(ampl)
        set_subset(spec_all, fn[i], win_fstep, n_save, spec, weights_all)
    spec_all[len(spec_all) // 2:] = 0
    spec_all /= weights_all.clip(1e-3, None)
    spec_all[1:] += np.conj(spec_all)[1:][::-1]
    track = np.fft.ifft(spec_all)
    track = np.real(track)
    return track
