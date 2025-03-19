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
            ResNetBlock(256, norm=True),

            Downsample(256, 256, norm=True),
            # ResNetBlock(512),
            # ResNetBlock(512),

            # Downsample(512, 512),
            # ResNetBlock(512),
            # ResNetBlock(512),

            # Downsample(512, 512),
            # ResNetBlock(512),
            # ResNetBlock(512),
        ])

    def forward(self, spectrogram: torch.Tensor):
        spectrogram = spectrogram.transpose(-1, -2)

        x = torch.concat([
            spectrogram.real,
            spectrogram.imag
        ], dim=1)

        x = self.in_conv(x)

        for block in self.blocks:
            x = block(x)

        return x
    

class Decoder(nn.Module):
    def __init__(self, n_freqs: int):
        super().__init__()

        self.blocks = nn.ModuleList([
            # ResNetBlock(512),
            # ResNetBlock(512),
            
            # Upsample(512, 512),
            # ResNetBlock(512),
            # ResNetBlock(512),
            
            # Upsample(512, 512),
            # ResNetBlock(512),
            # ResNetBlock(512),

            Upsample(256, 256, norm=True),
            ResNetBlock(256, norm=True),
            ResNetBlock(256, norm=True),

            ResNetBlock(256, norm=True),
        ])

        self.out_conv = nn.Conv1d(256, 2 * n_freqs, bias=False, kernel_size=1)

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        x = self.out_conv(x)

        spec_real, spec_imag = x.chunk(2, dim=1)
        spectrogram = torch.complex(spec_real, spec_imag)

        spectrogram = spectrogram.transpose(-1, -2)

        return spectrogram
    

class Discriminator(nn.Module):
    def __init__(self, n_freqs: int):
        super().__init__()
        ksize = 3

        self.in_conv = Downsample(2 * n_freqs, 256, ksize=ksize, stride=2, act=True)

        self.blocks = nn.ModuleList([
            ResNetBlock(256, ksize=ksize),

            Downsample(256, 256, ksize=ksize, stride=2, act=True),
            ResNetBlock(256, ksize=ksize),
            ResNetBlock(256, ksize=ksize),

            Downsample(256, 256, ksize=ksize, stride=2, act=True),
            ResNetBlock(256, ksize=ksize),
            ResNetBlock(256, ksize=ksize),

            Downsample(256, 256, ksize=ksize, stride=2, act=True),
            ResNetBlock(256, ksize=ksize),
            ResNetBlock(256, ksize=ksize),
        ])

        self.out_proj = nn.Conv1d(256, 1, 1, bias=False)
        self.out_norm = nn.BatchNorm1d(1)

        self.grad_rev = GradientReverse()

    def forward(self, *args) -> torch.Tensor:
        spectrogram = torch.concat([*args], dim=0)

        spectrogram = spectrogram.transpose(-1, -2)
        x = torch.concat([
            spectrogram.real,
            spectrogram.imag
        ], dim=1)

        x = self.in_conv(x)

        for block in self.blocks:
            x = block(x)

        x = self.out_proj(x)
        x = self.out_norm(x)

        if len(args) > 1:
            return x.chunk(len(args))
        else: 
            return x
    
    def discriminate(self, spec_real: torch.Tensor, spec_fake: torch.Tensor, gen: bool) -> torch.Tensor:
        # spec_fake = self.grad_rev(spec_fake)
        
        pred_real, pred_fake = self.forward(spec_real, spec_fake)

        loss_real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
        loss_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake) if gen else torch.zeros_like(pred_fake))
        loss = loss_real + loss_fake

        return loss
    
    def disc_loss(self, spec_real: torch.Tensor, spec_fake: torch.Tensor) -> torch.Tensor:
        pred_real, pred_fake = self.forward(spec_real, spec_fake)

        loss_real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
        loss_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
        loss = (loss_real + loss_fake) / 2

        return loss
    
    def gen_loss(self, spec_real: torch.Tensor, spec_fake: torch.Tensor) -> torch.Tensor:
        pred_real, pred_fake = self.forward(spec_real, spec_fake)

        loss_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))

        return loss_fake
    

class Autoencoder(pl.LightningModule):
    def __init__(self, n_freqs: int):
        super().__init__()

        self.encoder = Encoder(n_freqs)
        self.decoder = Decoder(n_freqs)
        self.discriminator = Discriminator(n_freqs)
        self.in_norm = nn.BatchNorm1d(2 * n_freqs, affine=False)

        self.automatic_optimization = False

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
        chunk, spec0, sr = batch

        opt_g, opt_d = self.optimizers()

        # Generator
        opt_g.zero_grad()

        spec, pred_spec = self(spec0)

        # loss_ae = (pred_spec - spec).abs().square().mean().sqrt() / spec.std()
        loss_ae = (pred_spec - spec).abs().mean() / spec.std()
        loss_disc_gen = 0#self.discriminator.gen_loss(spec, pred_spec)
        loss_gen = 10 * loss_ae + loss_disc_gen

        self.log("Lae", loss_ae, prog_bar=True)
        self.log("Lg", loss_disc_gen, prog_bar=True)
        self.manual_backward(loss_gen)
        opt_g.step()
        
        # # Discriminator
        # opt_d.zero_grad()

        # # spec, pred_spec = self(spec0)

        # loss_disc = self.discriminator.disc_loss(spec, pred_spec.detach())

        # self.log("Ld", loss_disc, prog_bar=True)
        # self.manual_backward(loss_disc)
        # opt_d.step()

    def configure_optimizers(self):
        opt_g = optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=1e-3)
        opt_d = optim.Adam(self.discriminator.parameters(), lr=1e-3)
        return [opt_g, opt_d], []
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 1 - 5e-4)
        # return [optimizer], [scheduler]
