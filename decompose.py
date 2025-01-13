import matplotlib.pyplot as plt

import torch
from torch import nn
import numpy as np
import cv2 as cv
from scipy.special import erfinv

from display import complex_picture
from process import sample_to_freq, freq_to_sample, get_window, FREQ_RES


def to_freq_diff_repr(spectrum) -> torch.Tensor:
    spectrum = (
        spectrum.abs() *
        torch.exp(1j * (spectrum.angle() - spectrum.roll(-1, -2).angle()))
    )

    return spectrum


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
    max_voice_full = 5000
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


def extract_voice(spectrum):

    basis = get_voice_basis(spectrum.shape[-1])

    spectrum *= torch.linspace(0, 1, spectrum.shape[-1])
    spectrum /= spectrum.abs().max()

    spectrum = to_freq_diff_repr(spectrum)

    f, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].imshow(complex_picture(spectrum))
    ax[1].imshow(complex_picture(basis))
    plt.show()
