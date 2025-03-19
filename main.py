import librosa
from scipy.io.wavfile import write
import numpy as np
import matplotlib.pyplot as plt
import torch

import notes
from display import render_plot, complex_picture
from process import build_spectrogram, generate_sound, freq_to_mel, mel_to_freq
from decompose import extract_voice

# SOURCE = "data/Detektivbyrn_-_Om_Du_Mter_Varg_63265005.mp3"
# SOURCE = "data/kukla_kolduna.mp3"
# SOURCE = "data/Alan_Jackson_-_Chattahoochee_48072565.mp3"
# SOURCE = "data/Lenka.mp3"
SOURCE = "data/podcast.mp3"
TIME_START = 100
TIME_WINDOW = 5

track, sample_rate = librosa.load(SOURCE)
track = track[int(TIME_START*sample_rate): int((TIME_START+TIME_WINDOW)*sample_rate)]
track0 = track

# track += 1 * np.random.randn(*track.shape)

# t = np.linspace(0, TIME_WINDOW, len(track), dtype=np.float32)
# # f1 = mel_to_freq(np.linspace(freq_to_mel(0), freq_to_mel(3000), len(track), dtype=np.float32)).numpy()
# # f1 = np.linspace(550, 550, len(track), dtype=np.float32)
# f1 = np.linspace(500, 600, len(track), dtype=np.float32)
# f2 = np.linspace(1500, 2000, len(track), dtype=np.float32)
# f2[:50000] = 0
# # f2 = np.linspace(1000, 1000, len(track), dtype=np.float32)
# track = np.sin(2 * np.pi * f1 * t)
# track += np.sin(2 * np.pi * f2 * t)

spectrum = build_spectrogram(track, sample_rate)

# spectrum += 0.1 * torch.randn_like(spectrum)

print(spectrum.shape)
plt.figure(figsize=(20, 10))
plt.imshow(complex_picture(spectrum[50:-50]).swapaxes(0, 1)[::-1], aspect=2, interpolation="nearest")
plt.savefig("complex_pic.png")
plt.close()

render_plot(np.abs(spectrum), 0, sample_rate)

track2 = generate_sound(spectrum, sample_rate)
write("track.wav", sample_rate, (track2*2**31).astype(np.int32))
write("orig.wav", sample_rate, (track*2**31).astype(np.int32))