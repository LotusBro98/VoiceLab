import librosa
from scipy.io.wavfile import write
import numpy as np
import matplotlib.pyplot as plt
import torch

import notes
# from display import render_plot, complex_picture
from process import SpectrogramBuilder#, build_spectrogram, generate_sound, freq_to_mel, mel_to_freq
# from decompose import extract_voice

# SOURCE = "data/Detektivbyrn_-_Om_Du_Mter_Varg_63265005.mp3"
# SOURCE = "data/kukla_kolduna.mp3"
# SOURCE = "data/Alan_Jackson_-_Chattahoochee_48072565.mp3"
# SOURCE = "data/Lenka.mp3"
SOURCE = "data/podcast.mp3"
TIME_START = 200
TIME_WINDOW = 3

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

builder = SpectrogramBuilder(sample_rate, magnitude=False, use_noise_masking=False, combo_scale=False)

# f = np.linspace(5000, 5000, len(track), dtype=np.float32)
# track = (np.sin(np.linspace(0, 1, len(track), dtype=np.float32) * f))

print(track.shape)
spec0 = builder.encode(torch.tensor(track, device="cuda"), snr=False)
spectrum = builder.encode(torch.tensor(track, device="cuda"), snr=True)
print(spectrum.shape)

# spectrum *= torch.rand_like(spectrum.abs()) < 0.9
# spectrum += torch.randn_like(spectrum) * 0.001

# spectrum = spectrum[:, 50:-50]
# spec2 = spec2[50:-50:2]
# spectrum += 0.1 * torch.randn_like(spectrum)

track2 = builder.decode(spectrum)
spec2 = builder.encode(track2)

f, ax = plt.subplots(3, figsize=(15, 10))
ax[0].imshow(builder.complex_picture(spec0)[::-1], aspect=1, interpolation="nearest")
ax[1].imshow(builder.complex_picture(spec2)[::-1], aspect=1, interpolation="nearest")
ax[2].imshow(builder.complex_picture(spec2 - spec0)[::-1], aspect=1, interpolation="nearest")
plt.savefig("complex_pic.png")
plt.close()

# render_plot(np.abs(spectrum), 0, sample_rate)


write("track.wav", sample_rate, (track2.cpu().numpy()*2**31).astype(np.int32))
write("orig.wav", sample_rate, (track*2**31).astype(np.int32))