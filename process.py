import matplotlib.pyplot as plt
import numpy as np
import notes

FREQ_STEP = np.power(2, 1/12 / 1)
MIN_FREQ = notes.NOTES["C2"]
SAVE_FREQ = 500


def log_spectrum(x, fs, fstep=FREQ_STEP, fmin=MIN_FREQ, fsave=SAVE_FREQ):
    n_save = int(len(x) * fsave / fs)
    spec_all = np.fft.fft(x)
    nmax = int(np.log(fs / 2 / fmin) / np.log(fstep))
    fn = len(x) / fs * fmin * np.power(fstep, np.arange(0, nmax + 1))
    log_spec = []
    for i in range(len(fn)):
        subset = spec_all[int(fn[i]/np.sqrt(fstep)): int(fn[i]*np.sqrt(fstep))]
        spec = np.zeros((n_save,), dtype=np.complex128)
        center = len(subset)//2
        end = min(len(subset), center + (n_save//2))
        start = max(0, center - (n_save//2))
        spec[:end-center] = subset[center:end]
        spec[-(center-start):] = subset[start:center]
        ampl = np.fft.ifft(spec)
        log_spec.append(ampl)
    log_spec = np.stack(log_spec, axis=-1)
    return log_spec


def generate(spectrum, fs, fstep=FREQ_STEP, fmin=MIN_FREQ, fsave=SAVE_FREQ):
    n_all = int(len(spectrum) * fs / fsave)
    n_save = spectrum.shape[0]
    spec_all = np.zeros((n_all,), dtype=np.complex128)
    nmax = int(np.log(fs / 2 / fmin) / np.log(fstep))
    fn = n_all / fs * fmin * np.power(fstep, np.arange(0, nmax + 1))
    for i in range(len(fn) - 1):
        ampl = spectrum[:,i]
        spec = np.fft.fft(ampl)
        subset = spec_all[int(fn[i]/np.sqrt(fstep)): int(fn[i]*np.sqrt(fstep))]
        center = len(subset)//2
        end = min(len(subset), center + (n_save//2))
        start = max(0, center - (n_save//2))
        subset[center:end] = spec[:end-center]
        subset[start:center] = spec[-(center-start):]
    spec_all[1:] += np.conj(spec_all)[1:][::-1]
    track = np.fft.ifft(spec_all)
    track = np.real(track)
    return track


def harmonics(spectrum, fstep=FREQ_STEP, n_harms=9):
    mask_idx = np.int32(np.round(np.log(range(1, n_harms + 1)) / np.log(fstep)))
    harms = np.stack([np.take(spectrum, mask_idx + i, axis=-1, mode='clip') for i in range(spectrum.shape[-1])], axis=-2)

    for t in range(harms.shape[0]):
        max_stacks = []
        for i in range(1):
            power = np.sum(np.abs(harms[t]), axis=-1)
            power_max_i = np.argmax(power)
            max_stacks.append((power_max_i, harms[t, power_max_i].copy()))

    return harms
