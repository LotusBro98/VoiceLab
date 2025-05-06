import math
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import erfinv
from scipy.interpolate import interp1d
import cv2 as cv

# SAVE_FREQ = 200
# FREQ_STEP = 2 ** (1/12)
# FREQ_RES = 2
# MIN_FREQ = 0
# MAX_FREQ = 11000
# MEL_N_FEATS = 128

# HEAR_SENSE_THRESHOLD = 1e-2


# def get_subset(spec_all, fn, win_size, n_save):
#     win = get_window(n_save, win_size)
#     if len(spec_all.shape) == 2:
#         win = win[None, ...]
#     win = win.to(spec_all.device)

#     fni = int(fn)
#     idx_from = torch.arange(fni, fni + n_save)
#     idx_from -= n_save // 2
#     idx_from %= spec_all.shape[-1]

#     idx_to = torch.arange(0, n_save)
#     idx_to -= n_save // 2
#     idx_to %= n_save

#     spec = win * spec_all[..., idx_from[idx_to]]

#     return spec


def mel_to_freq(mel):
    if not isinstance(mel, torch.Tensor):
        mel = torch.tensor(mel)
    return 700 * (torch.exp(mel / 1127) - 1)

def freq_to_mel(freq):
    if not isinstance(freq, torch.Tensor):
        freq = torch.tensor(freq)
    return 1127 * torch.log(1 + freq / 700)

# def get_mel_n_feats(mel_min, mel_max, fstep, superres):
#         mel_mid = (mel_min + mel_max) / 2
#         freq_mid = mel_to_freq(mel_mid)
#         dmel = mel_mid - freq_to_mel(freq_mid / fstep)
#         n_feats = int((mel_max - mel_min) / dmel) * superres

#         return n_feats

# def get_n_freqs(fmin=MIN_FREQ, fmax=MAX_FREQ, fstep=FREQ_STEP, superres=FREQ_RES):
#     if MEL_N_FEATS is not None:
#         return MEL_N_FEATS

#     mel_min = freq_to_mel(fmin)
#     mel_max = freq_to_mel(fmax)

#     n_feats = get_mel_n_feats(mel_min, mel_max, fstep, superres)
#     return n_feats


class SpectrogramBuilder(nn.Module):
    def __init__(self, 
                 sample_rate: float, 
                 fsave: float = 400,
                 n_feats: int = 128,
                 hear_sense_threshold: float = 1e-2):
        super().__init__()

        self.sample_rate = sample_rate
        self.fsave = fsave
        self.hear_sense_threshold = hear_sense_threshold
        self.n_feats = n_feats

        self.fmin = 0
        self.fmax = sample_rate / 2

        self.build_kernel()

    def build_kernel(self):
        fn = self.get_mel_scale()
        df = torch.gradient(fn)[0] * 2

        win_size_T = 2 / torch.min(df)

        t = torch.arange(0, win_size_T, 1 / self.sample_rate) - win_size_T / 2

        W = torch.exp(2j * torch.pi * fn[:, None] * t[None, :])

        ksize = W.shape[-1]
        stride = int(self.sample_rate / self.fsave)

        windows = self.get_window(W.shape[-1], (self.sample_rate / df)[:, None], W.shape[-1] // 2)
        mask = self.get_window_mask(ksize, stride, windows)

        W *= windows
        W /= W.abs().square().sum(-1, keepdim=True).sqrt()

        W_enc = W

        W_dec = W / mask

        # plt.imshow(self.complex_picture(W))
        # plt.savefig("kernel.png")
        # plt.close()

        self.kernel_encode = torch.concat([
            W_enc.real,
            W_enc.imag
        ], dim=0)[:, None, :]

        self.kernel_decode = torch.concat([
            W_dec.real,
            W_dec.imag
        ], dim=0)[:, None, :]

    def _encode_conv(self, signal: torch.Tensor) -> torch.Tensor:
        sig_shape = signal.shape
        sig_len = signal.shape[-1]
        stride = int(self.sample_rate / self.fsave)

        x = signal.reshape(-1, 1, sig_len)

        spec = F.conv1d(
            x, 
            weight=self.kernel_encode, 
            bias=None, 
            stride=stride,
            padding=0,
        )

        spec_real, spec_imag = spec.chunk(2, dim=1)
        spec = torch.complex(spec_real, spec_imag)

        spec = spec.reshape(sig_shape[:-1] + spec.shape[-2:])
        return spec
    
    def _decode_conv(self, spec: torch.Tensor) -> torch.Tensor:
        spec_shape = spec.shape
        stride = int(self.sample_rate / self.fsave)

        spec = spec.reshape(-1, *spec_shape[-2:])

        spec = torch.concat([
            spec.real,
            spec.imag
        ], dim=1)

        signal = F.conv_transpose1d(
            spec, 
            weight=self.kernel_decode, 
            bias=None, 
            stride=stride,
            padding=0,
        )

        signal = signal.reshape(*spec_shape[:-2], signal.shape[-1])
        return signal
    
    def reconstruct_phase(self, spec_abs: torch.Tensor) -> torch.Tensor:
        # Griffin-Lim algorithm for signal reconstruction from magnitude-only spectrogram

        spec = torch.complex(spec_abs, torch.zeros_like(spec_abs))
        
        for i in range(20):
            signal = self._decode_conv(spec)
            spec = self._encode_conv(signal)
            spec = spec_abs * torch.exp(1j * spec.angle())

        return spec

    def encode(self, signal: torch.Tensor) -> torch.Tensor:
        spec = self._encode_conv(signal)
        spec = spec.abs()
        spec = self.to_bel_scale(spec)
        return spec
    
    def decode(self, spec: torch.Tensor) -> torch.Tensor:
        spec = self.from_bel_scale(spec)
        spec = self.reconstruct_phase(spec)
        signal = self._decode_conv(spec)

        return signal
    
    def get_window(self, n_save, win_size, shift=0):
        window = (torch.arange(n_save) + shift + n_save // 2) % n_save - n_save // 2

        window = (torch.cos((window / win_size).clip(-1, 1) * torch.pi) + 1) / 2

        # prob_outside = 1e-2
        # std = (win_size / 2) / erfinv(1 - prob_outside)
        # window = torch.exp(-0.5 * torch.square(window / std))

        window -= window.abs().min()
        window /= window.sum(-1, keepdim=True) / win_size
        # window /= window.abs().max()

        return window

    def get_window_mask(self, ksize, stride, windows):
        center = ksize // 2
        idxs = torch.concatenate([
            torch.arange(center, 0, -stride)[1:].__reversed__() - center,
            torch.arange(center, ksize, stride) - center,
        ])

        mask = torch.zeros_like(windows)

        for i in idxs:
            mask[:, max(i, 0): min(ksize + i, ksize)] += windows[:, max(-i, 0): min(ksize - i, ksize)]

        # plt.plot((windows / mask).T)
        # plt.savefig("mask.png")
        # plt.close()

        return mask

    def freq_to_sample(self, freq):
        mel_min = freq_to_mel(self.fmin)
        mel_max = freq_to_mel(self.fmax)
        mel_x = freq_to_mel(freq)

        fn = (mel_x - mel_min) / (mel_max - mel_min) * self.n_feats
        return fn

    def sample_to_freq(self, fn):
        mel_min = freq_to_mel(self, self.fmin)
        mel_max = freq_to_mel(self, self.fmax)

        mel_x = fn / self.n_feats * (mel_max - mel_min) + mel_min
        freq = mel_to_freq(mel_x)

        return freq

    def get_mel_scale(self):
        mel_min = freq_to_mel(self.fmin)
        mel_max = freq_to_mel(self.fmax)

        mels = torch.linspace(mel_min, mel_max, self.n_feats)
        freqs = mel_to_freq(mels)

        return freqs

    def to_freq_diff_repr(self, spectrum: torch.Tensor) -> torch.Tensor:
        f = self.get_mel_scale() / self.fsave
        phase0 = torch.exp(2j * torch.pi * f[:, None])

        df = spectrum.angle()
        ampl = spectrum.abs()

        # df = df.diff(1, -1, prepend=torch.zeros_like(df[..., :1]))

        # df = (torch.exp(1j * df) * phase0).angle()

        spectrum = (
            ampl *
            torch.exp(1j * df)
        )

        return spectrum

    def from_freq_diff_repr(self, spectrum: torch.Tensor) -> torch.Tensor:
        f = self.get_mel_scale() / self.fsave
        phase0 = torch.exp(-2j * torch.pi * f[:, None])

        df = spectrum.angle()
        ampl = spectrum.abs()

        # df = (torch.exp(1j * df) * phase0).angle()

        # df = df.cumsum(-1)

        spectrum = (
            ampl *
            torch.exp(1j * df)
        )

        return spectrum

    def to_bel_scale(self, spectrum) -> torch.Tensor:
        ampl = spectrum.abs()
        ampl = torch.log10(1 + ampl / self.hear_sense_threshold)

        if not torch.is_complex(spectrum):
            return ampl
        
        return (
            ampl *
            torch.exp(1j * spectrum.angle())
        )

    def from_bel_scale(self, spectrum) -> torch.Tensor:
        ampl = spectrum.abs()
        ampl = self.hear_sense_threshold * (torch.pow(10, ampl) - 1)

        if not torch.is_complex(spectrum):
            return ampl

        return (
            ampl *
            torch.exp(1j * spectrum.angle())
        )
    
    def complex_picture(self, spectrum: torch.Tensor, ampl_cap: Literal["max", "std"] = "std"):
        ampl = spectrum.abs()
        if ampl_cap == "std":
            ampl /= ampl.abs().square().mean().sqrt() * 3
        elif ampl_cap == "max":
            ampl /= ampl.max()

        phase = spectrum.angle()

        image = torch.stack([
            (phase + torch.pi).rad2deg(),
            (1 - ampl).clip(None, 0).exp(),
            ampl.clip(0, 1),
        ], dim=-1).float().numpy()

        image = cv.cvtColor(image, cv.COLOR_HSV2RGB)

        return image


# def build_spectrogram(x, sample_rate, fsave=SAVE_FREQ, fmin=MIN_FREQ, fmax=MAX_FREQ):
#     if not isinstance(x, torch.Tensor):
#         x = torch.tensor(x)

#     x_len = x.shape[-1]

#     n_save = int(x_len * fsave / sample_rate)
#     spec_all = torch.fft.fft(x, dim=-1)
#     spec_all_freq_res = sample_rate / spec_all.shape[-1]
#     fn = get_mel_scale(fmin, fmax) / spec_all_freq_res
#     df = torch.gradient(fn)[0] * FREQ_RES

#     log_spec = []
#     for i in range(len(fn)):
#         spec = get_subset(spec_all, fn[i], df[i], n_save)
#         log_spec.append(spec)
#     log_spec = torch.stack(log_spec, dim=-1)
#     log_spec = torch.fft.ifft(log_spec, dim=-2)

#     log_spec = to_freq_diff_repr(log_spec)
#     log_spec = to_bel_scale(log_spec)

#     if SAVE_FREQ != REPR_FREQ:
#         t = np.linspace(0, 1, log_spec.shape[-2] - 1)
#         f = interp1d(t, log_spec[..., 1:, :], axis=-2, kind='quadratic')
#         t_new = np.linspace(0, 1, int((log_spec.shape[-2] - 1) * REPR_FREQ / SAVE_FREQ))
#         log_spec = torch.concat([log_spec[..., :1, :], torch.tensor(f(t_new), dtype=torch.complex64)], dim=-2)

#     return log_spec


# def generate_sound(spectrum, sample_rate, fsave=SAVE_FREQ, fmin=MIN_FREQ, fmax=MAX_FREQ):
#     if SAVE_FREQ != REPR_FREQ:
#         t = np.linspace(0, 1, spectrum.shape[-2] - 1)
#         f = interp1d(t, spectrum[..., 1:, :], axis=-2, kind='quadratic')
#         t_new = np.linspace(0, 1, int((spectrum.shape[-2] - 1) * SAVE_FREQ / REPR_FREQ))
#         spectrum = torch.concat([spectrum[..., :1, :], torch.tensor(f(t_new), dtype=torch.complex64)], dim=-2)
    
#     n_all = int(len(spectrum) * sample_rate / fsave)
#     n_save = spectrum.shape[0]

#     spec_all_freq_res = sample_rate / n_all

#     fn = get_mel_scale(fmin, fmax, fstep=FREQ_STEP, superres=FREQ_RES) / spec_all_freq_res
#     df = np.gradient(fn) * FREQ_RES

#     spectrum = from_bel_scale(spectrum)
#     spectrum = from_freq_diff_repr(spectrum)

#     spec_all = np.zeros((n_all,), dtype=np.complex128)
#     weights_all = np.zeros((n_all,), dtype=np.complex128)
    
#     for i in range(len(fn)):
#         ampl = spectrum[:,i]
#         spec = np.fft.fft(ampl)
#         set_subset(spec_all, fn[i], df[i], n_save, spec, weights_all)
#     spec_all[len(spec_all) // 2:] = 0

#     # plt.plot(weights_all)
#     # plt.savefig("window.png")
#     # plt.close()

#     spec_all /= weights_all.clip(weights_all.max() / 5, None)
#     spec_all[1:] += np.conj(spec_all)[1:][::-1]
#     track = np.fft.ifft(spec_all)
#     track = np.real(track)
#     return track
