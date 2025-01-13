import matplotlib.pyplot as plt

import torch
from torch import nn
import numpy as np
import cv2 as cv

from display import complex_picture
from process import sample_to_freq, freq_to_sample, get_window, FREQ_RES


def to_freq_diff_repr(spectrum):
    spectrum = (
        spectrum.abs() *
        torch.exp(1j * (spectrum.angle() - spectrum.roll(-1, -2).angle()))
    )

    return spectrum


def get_voice_basis(n_freqs):
    min_voice = 50
    max_voice = 1000
    base_freqs = sample_to_freq(
        torch.linspace(
            freq_to_sample(min_voice), 
            freq_to_sample(max_voice),
            200
        )
    )

    basis = [torch.zeros(n_freqs)]
    for freq in base_freqs:
        harmonics = torch.zeros(n_freqs)
        for i in range(1, 20):
            fni = int(freq_to_sample(freq * i))
            if fni >= n_freqs:
                break
            win = get_window(n_freqs, FREQ_RES)
            harmonics += win.roll(fni)
        basis.append(harmonics)
    basis = torch.stack(basis, dim=0)

    return basis


def extract_voice(spectrum):

    spectrum *= torch.linspace(0, 1, spectrum.shape[-1])
    spectrum /= spectrum.abs().max()

    spectrum = to_freq_diff_repr(spectrum)

    image = complex_picture(spectrum)

    plt.imshow(image)
    plt.show()

    # f, ax = plt.subplots