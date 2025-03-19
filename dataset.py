import glob
from typing import List
import librosa
import torch
from torch.utils.data import Dataset

from process import build_spectrogram, get_n_freqs

class AudioDataset(Dataset):
    def __init__(self, paths: List[str], chunk_len: float = 5):
        super().__init__()

        # paths = glob.glob(f"{root}/*.mp3", recursive=True)

        self.chunks = []

        for path in paths:
            track, sample_rate = librosa.load(path)

            step = int(chunk_len * sample_rate)
            for i in range(0, len(track) - step, step):
                chunk = track[i : i + step]
                self.chunks.append((chunk, sample_rate))


    def __getitem__(self, index) -> torch.Tensor:
        chunk, sample_rate = self.chunks[index]

        spec = build_spectrogram(chunk, sample_rate)

        return chunk, spec, sample_rate

    def __len__(self):
        return len(self.chunks)
    
    def get_n_freqs(self):
        return get_n_freqs()
