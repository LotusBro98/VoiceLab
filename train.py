import os

import librosa
import random
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

from dataset import AudioDataset
from model import Autoencoder
from display import render_plot, complex_picture
from process import generate_sound


def main():
    dataset = AudioDataset(["data/podcast.mp3"])
    train_loader = DataLoader(
        dataset, 
        batch_size=4, 
        num_workers=os.cpu_count(), 
        persistent_workers=True
    )

    autoencoder = Autoencoder(dataset.get_n_freqs())

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        # limit_train_batches=10000, 
        max_epochs=1,
        # accelerator="cpu"
        devices=1
    )

    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    autoencoder.eval()

    random.seed(42)
    chunk, spec, sample_rate = random.choice(dataset)

    with torch.no_grad():
        spec, spec_pred = autoencoder(spec[None, :])
        spec, spec_pred = spec[0], spec_pred[0]

    f, ax = plt.subplots(3, 1, figsize=(20, 20))
    ax[0].imshow(complex_picture(spec.T)[::-1])
    ax[1].imshow(complex_picture(spec_pred.T)[::-1])
    ax[2].imshow(complex_picture((spec - spec_pred).T)[::-1])
    plt.savefig("complex_picture.png")
    plt.close()

    render_plot(spec_pred, 0, sample_rate)

    track = generate_sound(spec, sample_rate)
    track_pred = generate_sound(spec_pred, sample_rate)

    write("track.wav", sample_rate, (track_pred*2**31).astype(np.int32))
    write("orig_back.wav", sample_rate, (track*2**31).astype(np.int32))
    write("orig.wav", sample_rate, (chunk.numpy()*2**31).astype(np.int32))


if __name__ == "__main__":
    main()
