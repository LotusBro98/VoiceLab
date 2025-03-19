from itertools import chain
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F

from special_layers import GradientReverse


class Downsample(nn.Module):
    def __init__(self, cin: int, cout: int, ksize: int = 3, stride: int = 2, norm=True, act=False):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=cin, 
            out_channels=cout,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            bias=not norm
        )
        self.norm = nn.BatchNorm1d(cout) if norm else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class Upsample(nn.Module):
    def __init__(self, cin: int, cout: int, ksize: int = 3, stride : int = 2, norm = True, act=False):
        super().__init__()

        self.conv = nn.ConvTranspose1d(
            in_channels=cin,
            out_channels=cout,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            bias=not norm
        )
        self.norm = nn.BatchNorm1d(cout) if norm else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class ResNetBlock(nn.Module):
    def __init__(self, cin, bottleneck=1, ksize=3, norm=True):
        super().__init__()
        cmid = cin // bottleneck

        self.conv1 = nn.Conv1d(cin, cmid, kernel_size=ksize, padding="same", bias=False)
        self.conv2 = nn.Conv1d(cmid, cmid, kernel_size=ksize, padding="same", bias=False)
        self.conv3 = nn.Conv1d(cmid, cin, kernel_size=ksize, padding="same", bias=False)

        self.norm1 = nn.BatchNorm1d(cmid)
        self.norm2 = nn.BatchNorm1d(cmid)
        self.norm3 = nn.BatchNorm1d(cmid) if norm else nn.Identity()

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

        self.in_conv = nn.Conv1d(2 * n_freqs, 256, kernel_size=1, bias=False)

        self.blocks = nn.ModuleList([
            ResNetBlock(256),

            Downsample(256, 256),
            ResNetBlock(256),
            ResNetBlock(256),

            Downsample(256, 256),
            ResNetBlock(256),
            ResNetBlock(256),

            Downsample(256, 256),
            ResNetBlock(256),
            ResNetBlock(256),
        ])

        self.out_conv = nn.Conv1d(256, 64, bias=False, kernel_size=1)
        self.out_norm = nn.BatchNorm1d(64)

    def forward(self, spectrogram: torch.Tensor):
        spectrogram = spectrogram.transpose(-1, -2)

        x = torch.concat([
            spectrogram.real,
            spectrogram.imag
        ], dim=1)

        x = self.in_conv(x)

        for block in self.blocks:
            x = block(x)

        x = self.out_conv(x)
        x = self.out_norm(x)

        return x
    

class Decoder(nn.Module):
    def __init__(self, n_freqs: int):
        super().__init__()

        self.in_conv = nn.Conv1d(64, 256, bias=False, kernel_size=1)
        self.in_norm = nn.BatchNorm1d(256)

        self.blocks = nn.ModuleList([
            ResNetBlock(256),
            ResNetBlock(256),
            
            Upsample(256, 256),
            ResNetBlock(256),
            ResNetBlock(256),
            
            Upsample(256, 256),
            ResNetBlock(256),
            ResNetBlock(256),

            Upsample(256, 256),
            ResNetBlock(256),
            ResNetBlock(256),

            ResNetBlock(256),
        ])

        self.out_conv = nn.Conv1d(256, 2 * n_freqs, bias=False, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.in_conv(x)
        x = self.in_norm(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.out_conv(x)

        spec_real, spec_imag = x.chunk(2, dim=1)
        spectrogram = torch.complex(spec_real, spec_imag)

        spectrogram = spectrogram.transpose(-1, -2)

        return spectrogram
    

class Autoencoder(pl.LightningModule):
    def __init__(self, n_freqs: int):
        super().__init__()

        self.encoder = Encoder(n_freqs)
        self.decoder = Decoder(n_freqs)
        self.in_norm = nn.BatchNorm1d(2 * n_freqs, affine=False)

    def normalize_input(self, spec):
        spec = spec.transpose(1, -1)
        spec = torch.concat([
            spec.real,
            spec.imag
        ], dim=1)

        spec = self.in_norm(spec)

        spec_real, spec_imag = spec.chunk(2, dim=1)
        spec = torch.complex(spec_real, spec_imag)
        spec = spec.transpose(1, -1)
        return spec
    
    def denormalize_output(self, spec):
        spec = spec.transpose(1, -1)
        spec = torch.concat([
            spec.real,
            spec.imag
        ], dim=1)

        spec = spec * self.in_norm.running_var[:, None].sqrt() + self.in_norm.running_mean[:, None]
        
        spec_real, spec_imag = spec.chunk(2, dim=1)
        spec = torch.complex(spec_real, spec_imag)
        spec = spec.transpose(1, -1)
        return spec

    def forward(self, spectrogram: torch.Tensor):
        spectrogram = self.normalize_input(spectrogram)
        feats = self.encoder(spectrogram)
        reconstructed_spec = self.decoder(feats)
        reconstructed_spec = self.denormalize_output(reconstructed_spec)
        spectrogram = self.denormalize_output(spectrogram)

        spectrogram = spectrogram[:, :reconstructed_spec.shape[1], :]
        reconstructed_spec = reconstructed_spec[:, :spectrogram.shape[1], :]
        return spectrogram, reconstructed_spec
    
    def training_step(self, batch: torch.Tensor, batch_idx):
        chunk, spec, sr = batch

        spec, pred_spec = self(spec)

        loss_ae = (pred_spec - spec).abs().square().mean().sqrt() / spec.std()
        # loss_ae = (pred_spec - spec).abs().mean() / spec.std()
        loss = loss_ae

        self.log("Lae", loss_ae, prog_bar=True)
        self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 1 - 5e-4)
        return [optimizer], [scheduler]
