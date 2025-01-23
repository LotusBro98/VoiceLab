import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F


class Downsample(nn.Module):
    def __init__(self, cin: int, cout: int, ksize: int = 2, stride: int = 2):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=cin, 
            out_channels=cout,
            kernel_size=ksize,
            stride=stride,
            bias=False
        )
        self.norm = nn.BatchNorm1d(cout)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)

        return x


class Upsample(nn.Module):
    def __init__(self, cin: int, cout: int, ksize: int = 2):
        super().__init__()

        self.conv = nn.ConvTranspose1d(
            in_channels=cin, 
            out_channels=cout,
            kernel_size=ksize,
            stride=ksize,
            bias=False
        )
        self.norm = nn.BatchNorm1d(cout)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)

        return x


class ResNetBlock(nn.Module):
    def __init__(self, cin, bottleneck=1, ksize=3):
        super().__init__()
        cmid = cin // bottleneck

        self.conv1 = nn.Conv1d(cin, cmid, kernel_size=ksize, padding="same", bias=False)
        self.conv2 = nn.Conv1d(cmid, cmid, kernel_size=ksize, padding="same", bias=False)
        self.conv3 = nn.Conv1d(cmid, cin, kernel_size=ksize, padding="same", bias=False)

        self.norm1 = nn.BatchNorm1d(cmid)
        self.norm2 = nn.BatchNorm1d(cmid)
        self.norm3 = nn.BatchNorm1d(cmid)

        # self.dropout = nn.Dropout(0.3)

        self.act = nn.ReLU()

    def forward(self, x):

        x0 = x

        # x = self.dropout(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x = x + x0

        return x
        

class Encoder(nn.Module):
    def __init__(self, n_freqs: int):
        super().__init__()

        self.in_conv = nn.Conv1d(2 * n_freqs, 1024, kernel_size=1, bias=False)

        self.test = nn.Parameter(torch.zeros(3 * n_freqs, 1))

        self.blocks = nn.ModuleList([
            ResNetBlock(512),

            Downsample(512, 512),
            ResNetBlock(512),
            ResNetBlock(512),

            # Downsample(128, 256),
            # ResNetBlock(256),
            # ResNetBlock(256),

            # Downsample(256, 512),
            # ResNetBlock(512),
            # ResNetBlock(512),
        ])

    def forward(self, spectrogram: torch.Tensor):
        spectrogram = spectrogram.transpose(-1, -2)

        # spectrogram = torch.log(spectrogram)
        # x = torch.concat([
        #     spectrogram.abs().log(),
        #     spectrogram.real / spectrogram.abs(),
        #     spectrogram.imag / spectrogram.abs(),
        # ], dim=1)

        x = torch.concat([
            spectrogram.real,
            spectrogram.imag
        ], dim=1)

        # x = x + self.test
        x = self.in_conv(x)

        # for block in self.blocks:
        #     x = block(x)

        return x
    

class Decoder(nn.Module):
    def __init__(self, n_freqs: int):
        super().__init__()

        self.blocks = nn.ModuleList([
            # ResNetBlock(512),
            # ResNetBlock(512),
            # Upsample(512, 256),

            # ResNetBlock(256),
            # ResNetBlock(256),
            # Upsample(256, 128),

            ResNetBlock(512),
            ResNetBlock(512),
            Upsample(512, 512),

            ResNetBlock(512),
        ])

        self.out_conv = nn.Conv1d(1024, 2 * n_freqs, bias=True, kernel_size=1)
        # self.out_norm = nn.BatchNorm1d(2 * n_freqs)

    def forward(self, x: torch.Tensor):
        # for block in self.blocks:
        #     x = block(x)
        x = self.out_conv(x)
        # x = self.out_norm(x)
        
        
        # spec_abs, spec_x, spec_y = x.chunk(3, dim=1)
        # spectrogram = spec_abs.exp() * torch.complex(
        #     spec_x / (spec_x.square() + spec_y.square()).sqrt(), 
        #     spec_y / (spec_x.square() + spec_y.square()).sqrt()
        # )

        spec_real, spec_imag = x.chunk(2, dim=1)
        spectrogram = torch.complex(spec_real, spec_imag)
        # spectrogram = torch.exp(spectrogram)

        # spectrogram = (
        #     spec_real *
        #     torch.exp(1j * spec_imag)
        # )

        spectrogram = spectrogram.transpose(-1, -2)

        return spectrogram
    

class Autoencoder(pl.LightningModule):
    def __init__(self, n_freqs: int):
        super().__init__()

        self.encoder = Encoder(n_freqs)
        self.decoder = Decoder(n_freqs)

    def forward(self, spectrogram: torch.Tensor):
        feats = self.encoder(spectrogram)
        reconstructed_spec = self.decoder(feats)

        return reconstructed_spec
    
    def training_step(self, batch, batch_idx):
        spec, sr = batch

        feats = self.encoder(spec)
        pred_spec = self.decoder(feats)

        # loss = (pred_spec - spec).abs().square().mean().sqrt() / spec.std()

        loss = ((pred_spec - spec).abs() / spec.abs().clip(1e-6, None)).mean()

        # eps = 1e-6
        # loss = 0
        # loss += F.mse_loss((eps + pred_spec.abs()).log(), (eps + spec.abs()).log()) / (eps + spec.abs()).log().std()
        # loss += ((
        #     pred_spec / pred_spec.abs().clip(1e-4, None) - 
        #     spec / spec.abs().clip(1e-4, None)
        # ).abs().square() * spec.abs().square()).mean().sqrt() / spec.abs().square().mean().sqrt()
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
