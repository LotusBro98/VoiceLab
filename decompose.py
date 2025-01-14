import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import cv2 as cv
from scipy.special import erfinv

from display import complex_picture
from process import sample_to_freq, freq_to_sample, get_window, to_freq_diff_repr, FREQ_RES


def get_sample_scale_window(n_freqs, fni):
    win_size = FREQ_RES
    window = torch.arange(n_freqs) - fni
    prob_outside = 1e-2
    std = (win_size / 2) / erfinv(1 - prob_outside)
    window = torch.exp(-0.5 * torch.square(window / std))

    return window


def get_voice_basis(n_freqs, n_basis=400):
    min_voice = 50
    max_voice = 1000
    max_voice_full = 10000
    base_freqs = sample_to_freq(
        torch.linspace(
            freq_to_sample(min_voice), 
            freq_to_sample(max_voice),
            n_basis
        )
    )
    
    sample_freqs = sample_to_freq(torch.arange(n_freqs))

    basis = []
    for freq in base_freqs:
        harmonics = torch.zeros(n_freqs, dtype=torch.complex64)
        for i in range(1, 30):
            freq_i = freq * i
            if freq_i >= max_voice_full:
                break

            fni = freq_to_sample(freq_i)
            win = get_sample_scale_window(n_freqs, fni)
            ampl = win
            df = (sample_freqs - freq_i) / freq
            harmonics += ampl * torch.exp(2j * torch.pi * df)
        basis.append(harmonics)
    basis = torch.stack(basis, dim=0)

    return basis


def matmul_complex(t1: torch.Tensor, t2: torch.Tensor):
    def ensure_complex(t: torch.Tensor):
        if not t.is_complex():
            return torch.complex(t, torch.zeros_like(t))
        else:
            return t
        
    t1 = ensure_complex(t1)
    t2 = ensure_complex(t2)

    return torch.view_as_complex(torch.stack((t1.real @ t2.real - t1.imag @ t2.imag, t1.real @ t2.imag + t1.imag @ t2.real),dim=2))


def voice_model(base_voice_freq, resonator_ampl, basis):
    main_voice = torch.softmax(base_voice_freq, dim=-1)
    voice = matmul_complex(main_voice, basis)

    res_ampl = torch.sigmoid(resonator_ampl)
    voice = voice * res_ampl

    return voice, main_voice, res_ampl


def decompose_voice(spectrum, basis):
    n_time, n_freq = spectrum.shape
    n_basis = basis.shape[0]

    base_voice_freq = nn.Parameter(torch.randn(n_time, n_basis))
    resonator_ampl = nn.Parameter(0.01 * torch.randn(n_time, n_freq))

    opt = torch.optim.Adam([base_voice_freq, resonator_ampl], lr=1e-1)

    for i in range(2000):
        opt.zero_grad()

        voice, main_voice, res_ampl = voice_model(base_voice_freq, resonator_ampl, basis)

        loss = (voice - spectrum).abs().square().mean().sqrt() / spectrum.abs().std()

        f = torch.arange(n_basis)[None, :]
        f_mean = (f * main_voice).sum(-1, keepdim=True)
        f_std = ((f - f_mean).square() * main_voice).sum(-1, keepdim=True).sqrt()
        loss += f_std.mean() / n_basis
        
        loss += res_ampl.diff(1, -1).square().mean().sqrt() / spectrum.std()
        loss += res_ampl.diff(1, -2).square().mean().sqrt() / spectrum.std()

        loss.backward()
        opt.step()

        with torch.no_grad():
            base_voice_freq -= base_voice_freq.mean(-1, keepdim=True)

        print(i, loss.item())

    return main_voice.data.detach(), res_ampl.data.detach(), voice.detach()


def extract_voice(spectrum):

    basis = get_voice_basis(spectrum.shape[-1], n_basis=200)

    spectrum *= torch.linspace(0, 1, spectrum.shape[-1])
    spectrum /= spectrum.abs().max()

    spectrum = to_freq_diff_repr(spectrum)

    base, res, voice = decompose_voice(spectrum, basis)

    voice_diff = voice - spectrum
    voice_diff /= voice_diff.abs().max()

    f, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax[0, 0].imshow(complex_picture(spectrum))
    ax[1, 0].imshow(complex_picture(basis))
    ax[2, 0].imshow(complex_picture(voice))

    ax[0, 1].imshow(base)
    ax[1, 1].imshow(res)
    ax[2, 1].imshow(complex_picture(voice_diff))
    plt.show()
