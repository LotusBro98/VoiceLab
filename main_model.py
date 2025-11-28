import librosa
from scipy.io.wavfile import write
import numpy as np
import matplotlib.pyplot as plt
import torch

from process import complex_picture
from upscale_model import UpsamplerTrainable

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

codec = UpsamplerTrainable()
codec.model.load_state_dict(torch.load("upsampler.pth"))
codec.model.eval()

spec = codec.encode(track)
track2 = codec.decode(spec)
spec2 = codec.encode(track2)

f, ax = plt.subplots(3, figsize=(15, 10))
ax[0].imshow(complex_picture(spec)[::-1], aspect=1, interpolation="nearest")
ax[1].imshow(complex_picture(spec2)[::-1], aspect=1, interpolation="nearest")
ax[2].imshow(complex_picture(spec2 - spec)[::-1], aspect=1, interpolation="nearest")
plt.savefig("complex_pic.png")
plt.close()

write("track.wav", sample_rate, (track2.cpu().numpy()*2**31).astype(np.int32))
write("orig.wav", sample_rate, (track*2**31).astype(np.int32))