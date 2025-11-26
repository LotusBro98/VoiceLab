import math
from typing import Literal, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import erfinv
from scipy.interpolate import interp1d
import torchvision
import torchvision.transforms.functional


def complex_picture(spectrum: torch.Tensor, ampl_cap: Literal["max", "std", None] = "std", noise: torch.Tensor = None):
    ampl = spectrum.abs()
    if noise is not None:
        snr = ampl / (ampl.square() + noise.square()).sqrt()
        ampl = (ampl.square() + noise.square()).sqrt()
    if ampl_cap is None:
        pass
    elif ampl_cap == "std":
        ampl /= ampl.abs().square().mean().sqrt() * 3
    elif ampl_cap == "max":
        ampl /= ampl.max()

    phase = spectrum.angle()

    if noise is not None:
        saturation = snr
    else:
        saturation = (1 - ampl).clip(None, 0).exp()
    image = torch.stack([
        (phase + torch.pi) / (2 * torch.pi),
        saturation,
        ampl.clip(0, 1),
    ], dim=0).float()

    image = torchvision.transforms.functional.to_pil_image(image, mode="HSV")
    rgb_image = image.convert(mode="RGB")
    rgb_image = np.array(rgb_image)

    return rgb_image

def create_gaussian_kernel(kernel_size):
    kernel_size = np.array(kernel_size)
    sigma = kernel_size / 3
    # Create a grid
    k = kernel_size // 2
    x = torch.linspace(-k[0], k[0], kernel_size[0])
    y = torch.linspace(-k[1], k[1], kernel_size[0])
    x, y = torch.meshgrid(x, y)
    
    # Compute the Gaussian kernel
    gaussian = torch.exp(-((x**2 / (2.0 * sigma[0]**2) + y**2 / (2.0 * sigma[1]**2))))
    gaussian /= gaussian.sum()
    
    return gaussian


class SpectrogramBuilder(nn.Module):
    def __init__(self, 
                 sample_rate: float, 
                 fsave: float = 400,
                 n_feats: int = 160,
                 freq_res: float = 2,
                 fmax: Optional[float] = None,
                 magnitude: bool = True,
                 freq_diff: bool = True,
                 hear_sense_threshold: float = 1e-0,
                 combo_scale: bool = True,
                 power_by_freq_scale: bool = True,         
                 use_noise_masking: bool = True,
        ):
        super().__init__()

        self.sample_rate = sample_rate
        self.fsave = fsave
        self.freq_res = freq_res
        self.hear_sense_threshold = hear_sense_threshold
        self.n_feats = n_feats
        self.magnitude = magnitude
        self.freq_diff = freq_diff
        self.combo_scale = combo_scale
        self.power_by_freq_scale = power_by_freq_scale
        self.use_noise_masking = use_noise_masking

        self.fmin = 0
        self.fmax = fmax if fmax is not None else sample_rate / 2

        self.fn = self.get_mel_scale()
        self.build_kernel()

    def build_kernel(self):
        fn = self.get_mel_scale()
        df = torch.gradient(fn)[0]

        win_sizes = (self.sample_rate / df)[:, None]
        ksize = int(math.ceil(2 * torch.max(win_sizes)))
        self.stride = int(self.sample_rate / self.fsave)

        windows = self.get_window(ksize, win_sizes / self.freq_res, ksize // 2)
        mask = self.get_window_mask(ksize, self.stride, windows.square())
        # mask = 0.0044 #* win_sizes ** 2

        t = (torch.arange(0, ksize) - ksize / 2) / self.sample_rate
        W = torch.exp(2j * torch.pi * fn[:, None] * t[None, :])
        W *= windows
        W /= win_sizes.sqrt()

        W_enc = W
        W_dec = W / mask * 2

        self.kernel_encode = torch.concat([
            W_enc.real,
            W_enc.imag
        ], dim=0)[:, None, :]

        self.kernel_decode = torch.concat([
            W_dec.real,
            W_dec.imag
        ], dim=0)[:, None, :]

    def _encode_conv(self, signal: torch.Tensor) -> torch.Tensor:
        if self.kernel_encode.device != signal.device:
            self.kernel_encode = self.kernel_encode.to(signal.device)
        sig_shape = signal.shape
        sig_len = signal.shape[-1]

        x = signal.reshape(-1, 1, sig_len)

        spec = F.conv1d(
            x, 
            weight=self.kernel_encode, 
            bias=None, 
            stride=self.stride,
            padding=(self.kernel_encode.shape[-1]-1)//2,
        )

        spec_real, spec_imag = spec.chunk(2, dim=1)
        spec = torch.complex(spec_real, spec_imag)

        spec = spec.reshape(sig_shape[:-1] + spec.shape[-2:])
        return spec
    
    def _decode_conv(self, spec: torch.Tensor) -> torch.Tensor:
        if self.kernel_decode.device != spec.device:
            self.kernel_decode = self.kernel_decode.to(spec.device)
        spec_shape = spec.shape

        spec = spec.reshape(-1, *spec_shape[-2:])

        spec = torch.concat([
            spec.real,
            spec.imag
        ], dim=1)

        signal = F.conv_transpose1d(
            spec, 
            weight=self.kernel_decode, 
            bias=None, 
            stride=self.stride,
            padding=(self.kernel_encode.shape[-1]-1)//2,
        )

        signal = signal.reshape(*spec_shape[:-2], signal.shape[-1])
        return signal
    
    def reconstruct_phase(self, spec_abs: torch.Tensor) -> torch.Tensor:
        # Griffin-Lim algorithm adaptation for signal reconstruction from magnitude-only spectrogram

        spec = torch.complex(spec_abs, torch.zeros_like(spec_abs))
        
        for i in range(100):
            signal = self._decode_conv(spec)
            spec = self._encode_conv(signal)
            spec = spec_abs * torch.exp(1j * spec.angle())

        return spec
    
    def signal_noise_decomposition(self, spec: torch.Tensor, K=None, n_feats=None, snr=1) -> Tuple[torch.Tensor, torch.Tensor]:
        spec_shape = spec.shape
        spec = spec.reshape(-1, 1, *spec_shape[-2:])

        ksize = (5, 5)
        if K is None:
            with torch.no_grad():
                patches = torch.nn.functional.unfold(spec, ksize).transpose(0, 1).reshape(np.prod(ksize), -1)
                patches -= patches.mean(-1, keepdim=True)
                cov = patches @ patches.H / patches.shape[-1]

                U, S, V = torch.linalg.svd(cov)
                n_feats = (S.cumsum(-1) < snr * S.flip(-1).cumsum(-1).flip(-1)).sum()
                # print(n_feats)
                K = V
                K = K.reshape(K.shape[0], *ksize)
                K *= create_gaussian_kernel(ksize).to(K.device)
                K = K / math.sqrt(np.prod(ksize))

        padding = ((ksize[0]-1)//2, (ksize[1]-1)//2)
        spec_scaled = F.conv2d(spec, K[:, None, :, :], padding="same")
        noise = spec_scaled[:, n_feats:, :, :].abs().square().sum(1).sqrt()
        spec_scaled = F.conv_transpose2d(spec_scaled[:, :n_feats], K[:n_feats, None, :, :].conj(), padding=padding)


        # plt.imshow(noise[0, :, :200].detach().cpu().numpy()[::-1])
        # plt.savefig("noise.png")
        # plt.close()

        # plt.imshow(complex_picture(spec_scaled[0, 0, :, :200].detach().cpu())[::-1])
        # plt.savefig("voice.png")
        # plt.close()

        spec = spec.reshape(spec_shape)
        spec_scaled = spec_scaled.reshape(spec_shape)
        noise = noise.reshape(spec_shape)

        return spec_scaled, noise, K, n_feats

    def encode(self, signal: torch.Tensor, snr=False) -> torch.Tensor:
        spec = self._encode_conv(signal)
        if self.magnitude:
            spec = spec.abs()
        if not self.magnitude and self.freq_diff:
            spec = self.to_freq_diff_repr(spec)
        
        spec = self.to_bel_scale(spec)
        # if snr:
        #     spec_coher, spec_noise = self.signal_noise_decomposition(spec)

        #     spec_coher = self.from_bel_scale(spec_coher)
        #     spec_noise = self.from_bel_scale(spec_noise)

        #     plt.figure(figsize=(15, 5))
        #     plt.imshow(complex_picture(self.to_bel_scale(spec_coher), noise=self.to_bel_scale(spec_noise))[::-1], interpolation="nearest")
        #     plt.savefig("with_noise.png")
        #     plt.close()
            
        #     spec = spec_coher + spec_noise * torch.randn_like(spec_coher)
        #     spec = self.to_bel_scale(spec)
        # spec = self.to_bel_scale(spec)
        
        return spec
    
    def decode(self, spec: torch.Tensor) -> torch.Tensor:
        if self.use_noise_masking:
            f_thresh = 3000
            mask = F.sigmoid((self.fn.to(spec.device) - f_thresh) / self.n_feats)
            mask = mask[:, None] * torch.ones_like(spec)
            mask[spec.abs() < 1 * self.hear_sense_threshold] = 1

        if self.magnitude:
            spec = spec.clip(0, None)

        spec = self.from_bel_scale(spec)
        if not self.magnitude and self.freq_diff:
            spec = self.from_freq_diff_repr(spec)
        if self.magnitude:
            spec = self.reconstruct_phase(spec)
        
        if self.use_noise_masking:
            noise = torch.randn_like(spec) * spec.abs() * 1
            spec = spec * (1 - mask) + noise * mask

        spec[..., :1, :] = 0
        signal = self._decode_conv(spec)
        return signal
    
    def get_window(self, n_save, win_size, shift=0, kind: Literal["flat", "hann", "gaussian", "invflat", "invhann"] = "gaussian"):
        window = (torch.arange(n_save) + shift + n_save // 2) % n_save - n_save // 2

        if kind == "hann":
            window = (torch.cos((window / win_size).clip(-1, 1) * torch.pi) + 1) / 2
        elif kind == "flat":
            window = (window.abs() < win_size).float()
        elif kind == "invflat":
            win_size = n_save / win_size
            window = (window.abs() < win_size).float()
            window = torch.fft.ifft(window.roll(-shift)).clone().roll(shift)
        elif kind == "invhann":
            win_size = n_save / win_size
            window = (torch.cos((window / win_size).clip(-1, 1) * torch.pi) + 1) / 2
            window = torch.fft.ifft(window.roll(-shift)).clone().roll(shift)
        elif kind == "gaussian":
            prob_outside = 1e-2
            std = (win_size / 2) / erfinv(1 - prob_outside)
            window = torch.exp(-0.5 * torch.square(window / std))

        if kind not in ["invflat", "invhann"]:
            window -= window.abs().min()
        window /= window.max(-1, keepdim=True).values

        return window

    def get_window_mask(self, ksize, stride, windows):
        idxs = torch.concatenate([
            torch.arange(-ksize // stride * stride, ksize // stride * stride, stride)[1:-1]
        ])

        mask = torch.zeros_like(windows)

        for i in idxs:
            mask[:, max(i, 0): min(ksize + i, ksize)] += windows[:, max(-i, 0): min(ksize - i, ksize)]

        return mask

    def freq_to_sample(self, freq):
        mel_min = self.freq_to_mel(self.fmin)
        mel_max = self.freq_to_mel(self.fmax)
        mel_x = self.freq_to_mel(freq)

        fn = (mel_x - mel_min) / (mel_max - mel_min) * self.n_feats
        return fn

    def sample_to_freq(self, fn):
        mel_min = self.freq_to_mel(self, self.fmin)
        mel_max = self.freq_to_mel(self, self.fmax)

        mel_x = fn / self.n_feats * (mel_max - mel_min) + mel_min
        freq = self.mel_to_freq(mel_x)

        return freq
    
    def _linear_tail_freq_scale(self, freqs):
        f_thresh = self.fsave / 4

        ii = torch.arange(1, self.n_feats)
        fs = freqs[1:]
        df = freqs[1:] - freqs[:-1]
        n_i = ii + (self.fmax - fs) / df
        scale = self.n_feats / n_i
        i_thresh = sum(df / scale < f_thresh)
        scale = scale[i_thresh]

        freqs = torch.concat([
            freqs[:i_thresh],
            torch.arange(freqs[i_thresh], self.fmax, f_thresh * scale)
        ])

        freqs = F.interpolate(freqs[None, None], self.n_feats, mode="linear")[0, 0]

        return freqs

    def get_mel_scale(self):
        mel_min = self.freq_to_mel(self.fmin)
        mel_max = self.freq_to_mel(self.fmax)

        mels = torch.linspace(mel_min, mel_max, self.n_feats)
        freqs = self.mel_to_freq(mels)

        if self.combo_scale:
            freqs = self._linear_tail_freq_scale(freqs)

        return freqs

    def to_freq_diff_repr(self, spectrum: torch.Tensor) -> torch.Tensor:
        f = self.get_mel_scale() / self.fsave
        phase0 = torch.exp(2j * torch.pi * f[:, None]).to(device=spectrum.device)

        # df = spectrum.angle()
        ampl = spectrum.abs()

        df = spectrum.angle() - (
            1.0 * spectrum.roll(1, -1).roll(1, -2) + 
            1.0 * spectrum.roll(1, -1).roll(0, -2) + 
            1.0 * spectrum.roll(1, -1).roll(-1, -2)
        ).angle()

        # df = df.diff(1, -1, prepend=torch.zeros_like(df[..., :1]))

        df = (torch.exp(1j * df) * phase0).angle()

        spectrum = (
            ampl *
            torch.exp(1j * df)
        )

        return spectrum

    def from_freq_diff_repr(self, spectrum: torch.Tensor) -> torch.Tensor:
        f = self.get_mel_scale() / self.fsave
        phase0 = torch.exp(-2j * torch.pi * f.to(spectrum.device)[:, None])

        df = spectrum.angle()
        ampl = spectrum.abs()

        df = (torch.exp(1j * df) * phase0).angle()

        # df = df.cumsum(-1)

        df1 = df[..., 0:1]
        spec1 = (ampl * torch.exp(1j * df))[..., 0:1]
        for i in range(1, df.shape[-1]):
            dfi = df[..., i:i+1] + (
                1.0 * spec1.roll(1, -2) +
                1.0 * spec1.roll(0, -2) +
                1.0 * spec1.roll(-1, -2)
            ).angle()
            spec1 = ampl[..., i:i+1] * torch.exp(1j * dfi)
            df1 = torch.concat([df1, dfi], dim=-1)
        df = df1

        spectrum = (
            ampl *
            torch.exp(1j * df)
        )

        return spectrum

    def to_bel_scale(self, spectrum) -> torch.Tensor:
        ampl = spectrum.abs()
        if self.power_by_freq_scale:
            ampl = ampl * self.fn.to(ampl.device)[:, None]
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
        if self.power_by_freq_scale:
            ampl = ampl / self.fn.to(ampl.device)[:, None].clip(1e-9, None)

        if not torch.is_complex(spectrum):
            return ampl

        return (
            ampl *
            torch.exp(1j * spectrum.angle())
        )

    def mel_to_freq(self, mel):
        if not isinstance(mel, torch.Tensor):
            mel = torch.tensor(mel)
        return 700 * (torch.exp(mel / 1127) - 1)

    def freq_to_mel(self, freq):
        if not isinstance(freq, torch.Tensor):
            freq = torch.tensor(freq)
        return 1127 * torch.log(1 + freq / 700)
