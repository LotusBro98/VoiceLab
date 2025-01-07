import librosa
from scipy.io.wavfile import write
import numpy as np

import notes
from display import render_plot
from process import generate, log_spectrum

# SOURCE = "data/Detektivbyrn_-_Om_Du_Mter_Varg_63265005.mp3"
# SOURCE = "data/kukla_kolduna.mp3"
# SOURCE = "data/Alan_Jackson_-_Chattahoochee_48072565.mp3"
# SOURCE = "data/Lenka.mp3"
SOURCE = "data/podcast.mp3"
TIME_START = 20
TIME_WINDOW = 2

track, sample_rate = librosa.load(SOURCE)
track = track[int(TIME_START*sample_rate): int((TIME_START+TIME_WINDOW)*sample_rate)]
track0 = track

spectrum = log_spectrum(track, sample_rate)

render_plot(np.abs(spectrum), 0, sample_rate)

track2 = generate(spectrum, sample_rate)
write("track.wav", sample_rate, (track2*2**31).astype(np.int32))
write("orig.wav", sample_rate, (track*2**31).astype(np.int32))