from itertools import chain
import math
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F

from process import to_bel_scale
from special_layers import GradientReverse


class Downsample(nn.Module):
    def __init__(self, cin: int, cout: int, ksize: int = 3, stride: int = 2, norm=True, act=False, dropout=0):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=cin, 
            out_channels=cout,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            bias=not norm
        )
        self.norm = nn.LazyInstanceNorm1d() if norm else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x


class Upsample(nn.Module):
    def __init__(self, cin: int, cout: int, ksize: int = 3, stride : int = 2, norm = True, act=True):
        super().__init__()

        self.conv = nn.ConvTranspose1d(
            in_channels=cin,
            out_channels=cout,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            bias=not norm
        )
        self.norm = nn.LazyInstanceNorm1d() if norm else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()

        # self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        # x = self.upsample(x)

        return x


class ResNetBlock(nn.Module):
    def __init__(self, cin, bottleneck=1, ksize=3, norm=True, dropout=0):
        super().__init__()
        cmid = int(cin * bottleneck)

        self.conv1 = nn.Conv1d(cin, cmid, kernel_size=ksize, padding="same", bias=False)
        self.conv2 = nn.Conv1d(cmid, cmid, kernel_size=ksize, padding="same", bias=False)
        self.conv3 = nn.Conv1d(cmid, cin, kernel_size=ksize, padding="same", bias=False)

        self.norm1 = nn.LazyInstanceNorm1d()
        self.norm2 = nn.LazyInstanceNorm1d()
        self.norm3 = nn.LazyInstanceNorm1d() if norm else nn.Identity()

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):

        x0 = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x = x + x0
        x = self.act(x)

        return x
        

class Encoder(nn.Module):
    def __init__(self, n_freqs: int):
        super().__init__()

        d_emb = 256
        d_model = 1

        self.in_conv = nn.Conv1d(2 * n_freqs, d_model, kernel_size=1, bias=False)

        self.blocks = nn.ModuleList([
            # ResNetBlock(256),

            Downsample(d_model, d_model),
            # ResNetBlock(256),
            # ResNetBlock(256),

            Downsample(d_model, d_model),
            # ResNetBlock(256),
            # ResNetBlock(256),

            Downsample(d_model, d_model),
            # ResNetBlock(256),
            # ResNetBlock(256),
        ])

        self.out_conv = nn.Conv1d(d_model, d_emb, bias=False, kernel_size=1)
        self.out_norm = nn.LazyInstanceNorm1d()

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

        d_emb = 256
        d_model = 256
        ksize_out = 33
        ksize = 3

        self.in_conv = nn.Conv1d(d_emb, d_model, bias=False, kernel_size=1)
        self.in_norm = nn.LazyInstanceNorm1d()

        self.blocks = nn.ModuleList([
            ResNetBlock(d_model, ksize=ksize),
            ResNetBlock(d_model, ksize=ksize),
            
            Upsample(d_model, d_model, ksize=ksize),
            ResNetBlock(d_model, ksize=ksize),
            ResNetBlock(d_model, ksize=ksize),
            ResNetBlock(d_model, ksize=ksize),
            ResNetBlock(d_model, ksize=ksize),
            
            Upsample(d_model, d_model, ksize=ksize),
            ResNetBlock(d_model, ksize=ksize),
            ResNetBlock(d_model, ksize=ksize),
            ResNetBlock(d_model, ksize=ksize),
            ResNetBlock(d_model, ksize=ksize),

            Upsample(d_model, d_model, ksize=ksize),
            ResNetBlock(d_model, ksize=ksize),
            ResNetBlock(d_model, ksize=ksize),
            ResNetBlock(d_model, ksize=ksize),

            # ResNetBlock(d_model, ksize=ksize, norm=False),
        ])

        self.out_conv = nn.Conv1d(d_model, 2 * n_freqs, bias=False, kernel_size=ksize_out)
        self.out_norm = nn.LazyInstanceNorm1d(affine=False)

    def forward(self, x: torch.Tensor):
        x = self.in_conv(x)
        x = self.in_norm(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.out_conv(x)
        x = self.out_norm(x)

        spec_real, spec_imag = x.chunk(2, dim=1)
        spectrogram = torch.complex(spec_real, spec_imag)

        spectrogram = spectrogram.transpose(-1, -2)

        return spectrogram
    

class Discriminator(nn.Module):
    def __init__(self, n_freqs: int):
        super().__init__()

        d_model = 128
        ksize_in = 15
        ksize = 3

        self.in_conv = nn.Conv1d(2 * n_freqs, d_model, kernel_size=ksize_in, bias=False)
        self.in_norm = nn.LazyInstanceNorm1d()

        self.blocks = nn.ModuleList([
            # ResNetBlock(256, dropout=0.5),

            Downsample(d_model, d_model, ksize=ksize, act=True),
            # ResNetBlock(d_model, ksize=ksize),

            Downsample(d_model, d_model, ksize=ksize, act=True),
            # ResNetBlock(d_model, ksize=ksize),

            Downsample(d_model, d_model, ksize=ksize, act=True),
            # ResNetBlock(d_model, ksize=ksize),
        ])

        self.out_conv = nn.Conv1d(d_model, 1, bias=False, kernel_size=ksize)
        self.out_norm = nn.BatchNorm1d(1)

        self.grad_rev = GradientReverse()

    def forward(self, *args: torch.Tensor):
        spectrogram = torch.concat(args, dim=0)
        spectrogram = spectrogram.transpose(-1, -2)

        x = torch.concat([
            spectrogram.real,
            spectrogram.imag
        ], dim=1)

        x = self.in_conv(x)
        x = self.in_norm(x)

        for block in self.blocks:
            x = block(x)

        x = self.out_conv(x)
        x = self.out_norm(x)

        if (len(args) > 1):
            x = x.chunk(len(args))

        return x
    
    def loss_disc(self, spec_real, spec_fake):
        spec_fake = self.grad_rev(spec_fake)
        pred_real, pred_fake = self(spec_real, spec_fake)

        loss_real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
        loss_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
        loss = (loss_real + loss_fake) / 2

        return loss
    

class Autoencoder(pl.LightningModule):
    def __init__(self, n_freqs: int):
        super().__init__()

        self.encoder = Encoder(n_freqs)
        self.decoder = Decoder(n_freqs)
        self.discriminator = Discriminator(n_freqs)

        self.in_norm = nn.BatchNorm1d(2 * n_freqs, affine=False)

    @torch.no_grad
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
    
    @torch.no_grad
    def denormalize_output(self, spec):
        spec = spec.transpose(1, -1)
        spec = torch.concat([
            spec.real,
            spec.imag
        ], dim=1)

        mean = self.in_norm.running_mean[:, None].detach()
        var = self.in_norm.running_var[:, None].sqrt().detach()
        spec = spec * var + mean
        
        spec_real, spec_imag = spec.chunk(2, dim=1)
        spec = torch.complex(spec_real, spec_imag)
        spec = spec.transpose(1, -1)
        return spec

    def forward(self, spectrogram: torch.Tensor, norm=True, crop=True):
        if norm:
            spectrogram = self.normalize_input(spectrogram)

        feats = self.encoder(spectrogram)
        feats = torch.randn_like(feats)
        reconstructed_spec = self.decoder(feats)

        if norm:
            reconstructed_spec = self.denormalize_output(reconstructed_spec)
            spectrogram = self.denormalize_output(spectrogram)

        if crop:
            pad = spectrogram.shape[1] - reconstructed_spec.shape[1]
            spectrogram = spectrogram[:, pad // 2: spectrogram.shape[1] - (pad - pad // 2), :]
        return spectrogram, reconstructed_spec
    
    def training_step(self, batch: torch.Tensor, batch_idx):
        chunk, spec, sr = batch

        spec0 = self.normalize_input(spec)

        spec, pred_spec = self(spec0, norm=False)
        loss_disc = self.discriminator.loss_disc(spec, pred_spec)
        loss = loss_disc

        self.log("Ld", loss_disc, prog_bar=True)
        self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True)
        return loss

    def configure_optimizers(self):
        base_lr = 1e-4

        opt_g = optim.Adam(self.parameters(), lr=base_lr)

        T1 = 100
        T2 = 10000
        sch_g = optim.lr_scheduler.SequentialLR(opt_g, [
            optim.lr_scheduler.LinearLR(opt_g, 1 / T1, 1, total_iters=T1),
            optim.lr_scheduler.ExponentialLR(opt_g, math.exp(-1 / T2))
        ], milestones=[T1])
        sch_g = {
            'scheduler': sch_g,
            'interval': 'step', # or 'epoch'
            'frequency': 1
        }
        return [opt_g], [sch_g]
