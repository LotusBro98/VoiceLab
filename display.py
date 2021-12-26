import numpy as np
from matplotlib import pyplot as plt

import notes
from process import MIN_FREQ, FREQ_STEP, SAVE_FREQ

BPM = 112
FOURTHS_IN_BEAT = 4
MINOR_TICK_FOURTH = 1 / 2

BEAT_TIME = FOURTHS_IN_BEAT / (BPM / 60)
MINOR_TICK_BEAT = MINOR_TICK_FOURTH / FOURTHS_IN_BEAT

TIME_STEP_MAJOR = BEAT_TIME
TIME_STEP_MINOR = BEAT_TIME * MINOR_TICK_BEAT

TIME_RES = TIME_STEP_MINOR / 8
FREQ_RES_WINDOW_TIME = TIME_RES * 2

PLOT_RES_TIME = 10*1/6
PLOT_RES_NOTE = 1/5
GRID_COLOR = 'gray'

HALF_NOTE = np.power(2, 1/12/2)


def render_plot(spectrum, time_start, fs, fstep=FREQ_STEP, fmin=MIN_FREQ, fsave=SAVE_FREQ):
    dt = 1 / fsave
    max_time = time_start + spectrum.shape[0] * dt

    notes_inside = dict([(note, spectrum.shape[-1]-1 - np.log(notes.NOTES[note] / fmin) / np.log(fstep)) for note in notes.NOTES.keys() if
                         notes.NOTES[note] >= MIN_FREQ and notes.NOTES[note] <= fs])

    n_notes = len(notes_inside)

    im_size_x = PLOT_RES_TIME * ((max_time - time_start) / TIME_STEP_MINOR)
    im_size_y = PLOT_RES_NOTE * n_notes

    padding_x = 5
    padding_y = 5

    figsize_x = im_size_x + padding_x
    figsize_y = im_size_y + padding_y

    fig = plt.figure(figsize=(figsize_x, figsize_y), tight_layout={'pad': 3})
    ax = fig.add_subplot(111)

    ax.set_ylim((list(notes_inside.values())[0], list(notes_inside.values())[-1]))

    aspect1 = ((max_time - time_start) / dt) / ((np.log(fs) - np.log(fmin)) / np.log(fstep))
    aspect = aspect1 * im_size_y / im_size_x

    xticks_major = np.arange(time_start, max_time, TIME_STEP_MAJOR) / dt
    xticks_minor = np.arange(time_start, max_time, TIME_STEP_MINOR) / dt
    ax.set_xticks(xticks_minor, minor=True)
    ax.set_xticks(xticks_major)
    ax.set_xticklabels(np.arange(1, len(xticks_major) + 1, 1))

    ax.xaxis.grid(True, which='minor', color=GRID_COLOR, linestyle="--", alpha=0.3)
    ax.xaxis.grid(True, which='major', color=GRID_COLOR, linewidth=2, alpha=0.5)

    ax.set_yticks(list(notes_inside.values()), minor=True)
    ax.set_yticklabels(list(notes_inside.keys()), minor=True)

    ax.yaxis.grid(True, which='minor', color=GRID_COLOR, linestyle="--", alpha=0.3)

    ax.imshow(spectrum.T[::-1], aspect=aspect, interpolation='nearest')

    plt.show()