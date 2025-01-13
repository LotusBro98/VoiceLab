import matplotlib.pyplot as plt

import torch
from torch import nn
import numpy as np
import cv2 as cv

from display import complex_picture


def to_freq_diff_repr(spectrum):
    spectrum = (
        spectrum.abs() *
        torch.exp(1j * (spectrum.angle() - spectrum.roll(-1, -2).angle()))
    )

    return spectrum

def extract_voice(spectrum):

    spectrum *= torch.linspace(0, 1, spectrum.shape[-1])
    spectrum /= spectrum.abs().max()

    spectrum = to_freq_diff_repr(spectrum)

    image = complex_picture(spectrum)

    plt.imshow(image)
    plt.show()

    # f, ax = plt.subplots