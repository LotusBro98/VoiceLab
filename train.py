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
from process import SpectrogramBuilder


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

    # random.seed(42)
    # chunk, spec, sample_rate = random.choice(dataset)
    chunk, spec, sample_rate = dataset[40]

    builder = SpectrogramBuilder(sample_rate)

    with torch.no_grad():
        spec, spec_pred = autoencoder(spec[None, :])
        spec, spec_pred = spec[0], spec_pred[0]

    f, ax = plt.subplots(3, 1, figsize=(40, 20))
    ax[0].imshow(builder.complex_picture(spec)[::-1], aspect=1)
    ax[1].imshow(builder.complex_picture(spec_pred)[::-1], aspect=1)
    ax[2].imshow(builder.complex_picture((spec - spec_pred))[::-1], aspect=1)
    plt.savefig("complex_picture.png")
    plt.close()

    track = builder.decode(spec).numpy()
    track_pred = builder.decode(spec_pred).numpy()

    write("track.wav", sample_rate, (track_pred*2**31).astype(np.int32))
    write("orig_back.wav", sample_rate, (track*2**31).astype(np.int32))
    write("orig.wav", sample_rate, (chunk.numpy()*2**31).astype(np.int32))


if __name__ == "__main__":
    main()
