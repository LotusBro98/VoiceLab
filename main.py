from pyffmpeg import FFmpeg
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import numpy as np

import notes
from display import render_plot
from process import generate, log_spectrum, harmonics, SAVE_FREQ

# SOURCE = "data/Detektivbyrn_-_Om_Du_Mter_Varg_63265005.mp3"
# SOURCE = "data/kukla_kolduna.mp3"
SOURCE = "data/Alan_Jackson_-_Chattahoochee_48072565.mp3"
# SOURCE = "Lenka.mp3"
# SOURCE = "data/Lenka.mp3"
TEMP = "temp.wav"
TIME_START = 0
TIME_WINDOW = 10

ff = FFmpeg()
ff.convert(SOURCE, TEMP)

sample_rate, track = read(TEMP)
track = track[:,0]
track = track / 2**15
track = track[int(TIME_START*sample_rate): int((TIME_START+TIME_WINDOW)*sample_rate)]
track0 = track

spectrum = log_spectrum(track, sample_rate)

harms = harmonics(spectrum)
harms = harms[int(1.125 * SAVE_FREQ)]
plt.plot(np.sum(np.abs((harms)), axis=-1))
plt.show()

render_plot(np.abs(spectrum), 0, sample_rate)
render_plot(np.abs(harms.T), 0, sample_rate)

track2 = generate(spectrum, sample_rate)
write("track.wav", sample_rate, (track2*2**15).astype(np.int16))