import librosa
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from process import SpectrogramBuilder, complex_picture
from scipy.io.wavfile import write


class ResBlock(nn.Module):
    def __init__(self, cin, cout, bottleneck=0.25):
        super().__init__()
        cmid = int(round(bottleneck * max(cin, cout)))
        self.do_skip_conv = cin != cout

        self.conv1 = nn.Conv2d(cin, cmid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cmid)

        self.conv2 = nn.Conv2d(cmid, cmid, 3, bias=False, padding="same")
        self.bn2 = nn.BatchNorm2d(cmid)

        self.conv3 = nn.Conv2d(cmid, cout, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)

        self.conv_skip = nn.Conv2d(cin, cout, 1, bias=False) if self.do_skip_conv else None
        self.bn_skip = nn.BatchNorm2d(cout) if self.do_skip_conv else None

        self.act = nn.ReLU()

    def forward(self, x):
        x0 = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.do_skip_conv:
            x0 = self.conv_skip(x0)
            x0 = self.bn_skip(x0)

        x = x0 + x

        return x


class SpecUpsampler(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()

        self.layers = nn.ModuleList([
            ResBlock(2, d_model),
            ResBlock(d_model, d_model),
            ResBlock(d_model, d_model),
            ResBlock(d_model, 2),
        ])

    def forward(self, spec: torch.Tensor):
        spec_shape = spec.shape
        spec = spec.reshape(-1, 1, *spec_shape[-2:])
        x = torch.cat([spec.real, spec.imag], dim=1)

        x = F.interpolate(x, scale_factor=(2, 2), mode="nearest")

        for layer in self.layers:
            x = layer(x)
        
        spec = torch.complex(x[:, 0:1], x[:, 1:2])
        spec = spec.reshape(spec_shape[:-2] + spec.shape[-2:])
        return spec


def main():
    sample_rate = 22050

    builder_encode = SpectrogramBuilder(
        sample_rate,
        fsave=200,
        n_feats=80,
        freq_res=1,
        magnitude=False,
        combo_scale=False,
        use_noise_masking=False,
    )

    builder_decode = SpectrogramBuilder(
        sample_rate,
        fsave=400,
        n_feats=160,
        freq_res=2,
        magnitude=False,
        combo_scale=False,
        use_noise_masking=False,
    )

    model = SpecUpsampler().to("cuda")

    SOURCE = "data/podcast.mp3"
    TIME_START = 200
    TIME_WINDOW = 3

    track, sample_rate = librosa.load(SOURCE)
    track = track[int(TIME_START*sample_rate): int((TIME_START+TIME_WINDOW)*sample_rate)]
    track = torch.tensor(track, device="cuda")

    spec0 = builder_encode.encode(track)
    with torch.no_grad():
        spec1 = model(spec0)

    track2 = builder_decode.decode(spec1)
    write("track.wav", sample_rate, (track2.cpu().numpy()*2**31).astype(np.int32))
    
    f, ax = plt.subplots(3, figsize=(15, 10))
    ax[0].imshow(complex_picture(spec0)[::-1], aspect=1, interpolation="nearest")
    ax[1].imshow(complex_picture(spec1)[::-1], aspect=1, interpolation="nearest")
    # ax[2].imshow(complex_picture(spec2 - spec0)[::-1], aspect=1, interpolation="nearest")
    plt.savefig("complex_pic.png")
    plt.close()


if __name__ == "__main__":
    main()
