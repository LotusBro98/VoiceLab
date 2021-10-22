from pyffmpeg import FFmpeg
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import numpy as np

from process import generate, log_spectrum

SOURCE = "data/Detektivbyrn_-_Om_Du_Mter_Varg_63265005.mp3"
# SOURCE = "Lenka.mp3"
TEMP = "temp.wav"

ff = FFmpeg()
ff.convert(SOURCE, TEMP)

sample_rate, track = read(TEMP)
track = track[:,0]
track = track / 2**15
track0 = track


sec = 0
dsec = 10
track = track0[(sec)*sample_rate : (sec + dsec)*sample_rate]
spectrum = log_spectrum(track, sample_rate)
print(spectrum.shape)
plt.figure(figsize=(20, 10))
plt.imshow(np.abs(spectrum).T[::-1], aspect=1/2, extent=[0, 1, 0, 1])
plt.show()

track2 = generate(spectrum, sample_rate)
write("track.wav", sample_rate, track2.astype(np.int16))

spectrum2 = log_spectrum(track2, sample_rate)
print(spectrum2.shape)
plt.figure(figsize=(20, 10))
plt.imshow(np.abs(spectrum2).T[::-1], aspect=1/2, extent=[0, 1, 0, 1])
plt.show()