from itertools import chain
import math
from typing import Literal, Optional
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F

from dataset import AudioDataset
from special_layers import GradientReverse


class Downsample(nn.Module):
    def __init__(self, cin: int, cout: int, ksize: int = 3, stride: int = 2, 
                 norm: Literal["batch", "instance", "none", None] = "batch", 
                 act: Literal["relu", "leakyrelu", "none", None] = "relu", 
                 bias: Optional[bool] = None,
                 dropout=0):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=cin, 
            out_channels=cout,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            bias=(not norm) if bias is None else bias
        )
        self.norm = nn.LazyBatchNorm1d() if norm == "batch" else nn.LazyInstanceNorm1d() if norm == "instance" else nn.Identity()
        self.act = nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2) if act == "leakyrelu" else nn.Identity()
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

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x0 = x

        x = self.conv(x)
        x = self.norm(x)

        x = x + self.upsample(x0)[..., :x.shape[-1]]

        x = self.act(x)

        return x


class ResNetBlock(nn.Module):
    def __init__(self, cin, cout=None, bottleneck=1, ksize=3, stride=1,
                 norm: Literal["batch", "instance", "none", None] = "batch",
                 act_mid: Literal["relu", "leakyrelu"] = "relu",
                 act_out: Literal["relu", "leakyrelu", "none", None] = None,
                 dropout=0, up=False):
        super().__init__()
        if cout is None:
            cout = cin
        cmid = int(max(cin, cout) * bottleneck)

        norm_class = nn.LazyBatchNorm1d if norm == "batch" else nn.LazyInstanceNorm1d if norm == "instance" else nn.Identity

        self.conv1 = nn.Conv1d(cin, cmid, kernel_size=ksize, padding="same", bias=False)
        self.conv2 = (nn.ConvTranspose1d if up else nn.Conv1d)(
            cmid, cmid, kernel_size=ksize, stride=stride, bias=False,
            padding="same" if stride == 1 else (ksize - 1) // 2)
        self.conv3 = nn.Conv1d(cmid, cin, kernel_size=ksize, padding="same", bias=False)
        self.conv_short = (
            nn.Identity() if stride == 1 and cin == cout else
            nn.Upsample(scale_factor=stride if up else 1 / stride) if cin == cout else
            nn.ConvTranspose1d(cin, cout, 1, stride=stride, bias=False) if up else
            nn.Conv1d(cin, cout, 1, stride=stride, bias=False)
        )


        self.norm1 = norm_class()
        self.norm2 = norm_class()
        self.norm3 = norm_class()
        self.norm_short = nn.Identity() if cin == cout else norm_class()

        self.act_mid = nn.LeakyReLU(0.2) if act_mid == "leakyrelu" else nn.ReLU()
        self.act_out = nn.ReLU() if act_out == "relu" else nn.LeakyReLU(0.2) if act_out == "leakyrelu" else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x0 = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act_mid(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act_mid(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.norm3(x)

        if isinstance(self.conv_short, nn.Upsample):
            x0 = F.interpolate(x0, x.shape[2:])
        elif not isinstance(self.conv_short, nn.Identity):
            x0 = self.conv_short(x0)
            x = x[..., :x0.shape[-1]]
            x0 = x0[..., :x.shape[-1]]
        x0 = self.norm_short(x0)

        x = x + x0

        x = self.act_out(x)

        return x
        

class Encoder(nn.Module):
    def __init__(self, n_freqs: int):
        super().__init__()

        d_emb = 512
        d_model = 512
        ksize = 3
        ksize_in = 7
        stride_in = 1

        self.in_conv = nn.Sequential(
            nn.Conv1d(2 * n_freqs, d_model, kernel_size=ksize_in, stride=stride_in, bias=False, padding=ksize_in//2),
            nn.LazyBatchNorm1d()
        )

        self.blocks = nn.ModuleList([
            # ResNetBlock(d_model, d_model, ksize=ksize, stride=2),
            # ResNetBlock(d_model, ksize=ksize),
            # ResNetBlock(d_model, ksize=ksize),

            # ResNetBlock(d_model, d_model, ksize=ksize, stride=2),
            # ResNetBlock(d_model, ksize=ksize),
            # ResNetBlock(d_model, ksize=ksize),

            # ResNetBlock(d_model, d_model, ksize=ksize, stride=2),
            # ResNetBlock(d_model, ksize=ksize),
            # ResNetBlock(d_model, ksize=ksize),
        ])

        self.out_conv = nn.Sequential(
            nn.Conv1d(d_model, d_emb, bias=False, kernel_size=ksize),
            nn.LazyBatchNorm1d(),
        )

    def forward(self, spectrogram_real: torch.Tensor, spectrogram_imag: torch.Tensor):
        x = torch.concat([
            spectrogram_real,
            spectrogram_imag
        ], dim=1)

        x = self.in_conv(x)

        for block in self.blocks:
            x = block(x)

        x = self.out_conv(x)

        return x
    

class Decoder(nn.Module):
    def __init__(self, n_freqs: int):
        super().__init__()

        d_emb = 512
        d_model = 512
        ksize_out = 7
        stride_out = 1
        ksize = 3

        self.in_conv = nn.Sequential(
            nn.Conv1d(d_emb, d_model, bias=False, kernel_size=1),
            nn.LazyBatchNorm1d(),
            # nn.ReLU(),
        )

        self.blocks = nn.ModuleList([
            # ResNetBlock(d_model, ksize=ksize),
            # ResNetBlock(d_model, ksize=ksize),
            
            # ResNetBlock(d_model, d_model, ksize=ksize, stride=2, up=True),
            # ResNetBlock(d_model, ksize=ksize),
            # ResNetBlock(d_model, ksize=ksize),
            # ResNetBlock(d_model, ksize=ksize),
            # ResNetBlock(d_model, ksize=ksize),
            
            # ResNetBlock(d_model, d_model, ksize=ksize, stride=2, up=True),
            # ResNetBlock(d_model, ksize=ksize),
            # ResNetBlock(d_model, ksize=ksize),
            # ResNetBlock(d_model, ksize=ksize),
            # ResNetBlock(d_model, ksize=ksize),

            # ResNetBlock(d_model, d_model, ksize=ksize, stride=2, up=True),
            # ResNetBlock(d_model, ksize=ksize),
            # ResNetBlock(d_model, ksize=ksize),
            # ResNetBlock(d_model, ksize=ksize),
        ])

        self.out_conv = nn.Sequential(
            nn.ConvTranspose1d(d_model, 2 * n_freqs, kernel_size=ksize_out, stride=stride_out, bias=False, padding=ksize_out//2),
            # nn.LazyBatchNorm1d(affine=False)
        )

    def forward(self, x: torch.Tensor):
        x = self.in_conv(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.out_conv(x)

        spec_real, spec_imag = x.chunk(2, dim=1)
        return spec_real, spec_imag
        # spectrogram = torch.complex(spec_real, spec_imag)
        # return spectrogram
    

class Discriminator(nn.Module):
    def __init__(self, n_freqs: int):
        super().__init__()

        d_model = 512
        ksize_in = 3
        stride_in = 1
        stride = 1
        ksize = 3

        self.in_conv = Downsample(2 * n_freqs, d_model, ksize=ksize_in, stride=stride_in, norm="instance", bias=False, act="leakyrelu")

        self.blocks = nn.ModuleList([
            # ResNetBlock(d_model, ksize=ksize),

            # Downsample(d_model, d_model, ksize=ksize, stride=stride, norm="instance", act="leakyrelu"),
            # ResNetBlock(d_model, ksize=ksize),

            # Downsample(d_model, d_model, ksize=ksize, stride=stride, norm="instance", act="leakyrelu"),
            # ResNetBlock(d_model, ksize=ksize),

            # Downsample(d_model, d_model, ksize=ksize, stride=stride, norm="instance", act="leakyrelu"),
            # ResNetBlock(d_model, ksize=ksize),
        ])

        self.out_conv = nn.Sequential(
            nn.Conv1d(d_model, 1, bias=False, kernel_size=ksize),
            nn.BatchNorm1d(1),
        )

        self.grad_rev = GradientReverse()

    def forward(self, *args: torch.Tensor):
        spectrogram = torch.concat(args, dim=0)

        x = torch.concat([
            spectrogram.real,
            spectrogram.imag
        ], dim=1)

        x = self.in_conv(x)

        for block in self.blocks:
            x = block(x)

        x = self.out_conv(x)

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
        spec = torch.concat([
            spec.real,
            spec.imag
        ], dim=1)

        spec = self.in_norm(spec)

        spec_real, spec_imag = spec.chunk(2, dim=1)
        spec = torch.complex(spec_real, spec_imag)
        return spec
    
    @torch.no_grad
    def denormalize_output(self, spec):
        spec = torch.concat([
            spec.real,
            spec.imag
        ], dim=1)

        mean = self.in_norm.running_mean[:, None].detach()
        var = self.in_norm.running_var[:, None].sqrt().detach()
        spec = spec * var + mean
        
        spec_real, spec_imag = spec.chunk(2, dim=1)
        spec = torch.complex(spec_real, spec_imag)
        return spec

    def forward(self, spectrogram: torch.Tensor, norm=True, crop=True):
        if norm:
            spectrogram = self.normalize_input(spectrogram)

        feats = self.encoder(spectrogram.real, spectrogram.imag)
        # feats = torch.randn_like(feats)
        reconstructed_spec = torch.complex(*self.decoder(feats))

        if norm:
            reconstructed_spec = self.denormalize_output(reconstructed_spec)
            spectrogram = self.denormalize_output(spectrogram)

        if crop:
            pad = spectrogram.shape[-1] - reconstructed_spec.shape[-1]
            spectrogram = spectrogram[..., pad // 2: spectrogram.shape[-1] - (pad - pad // 2)]
            # spectrogram = spectrogram[..., :spectrogram.shape[-1] - pad]
        return spectrogram, reconstructed_spec
    
    def training_step(self, batch: torch.Tensor, batch_idx):
        chunk, spec, sr = batch

        spec0 = self.normalize_input(spec)

        spec, pred_spec = self(spec0, norm=False)
        loss_ae = (spec - pred_spec).abs().square().mean().sqrt() / spec.std()
        # loss_ae = (
        #     (pred_spec.real - spec.real).square() / spec.real.std().square() +
        #     (pred_spec.imag - spec.imag).square() / spec.imag.std().square()
        # ).mean().sqrt()
        loss_disc = self.discriminator.loss_disc(spec, pred_spec)
        loss = 100 * loss_ae + loss_disc

        self.log("Lae", loss_ae, prog_bar=True)
        self.log("Ld", loss_disc, prog_bar=True)
        self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True)
        return loss

    def configure_optimizers(self):
        base_lr = 3e-4

        opt_g = optim.AdamW(self.parameters(), lr=base_lr, weight_decay=0.01)

        T1 = 500
        Tall = self.trainer.estimated_stepping_batches
        sch_g = optim.lr_scheduler.SequentialLR(opt_g, [
            optim.lr_scheduler.LinearLR(opt_g, 1 / T1, 1, total_iters=T1),
            optim.lr_scheduler.CosineAnnealingLR(opt_g, Tall - 1 - T1),
        ], milestones=[T1])
        sch_g = {
            'scheduler': sch_g,
            'interval': 'step', # or 'epoch'
            'frequency': 1
        }
        return [opt_g], [sch_g]


def test():
    dataset = AudioDataset(["data/podcast.mp3"])
    autoencoder = Autoencoder(dataset.get_n_freqs())

    chunk, spec, sample_rate = dataset[40]

    autoencoder(spec[None])
    feats = autoencoder.encoder(spec[None].real, spec[None].imag)

    torch.onnx.export(
        autoencoder.encoder,
        (spec[None].real, spec[None].imag),
        "encoder.onnx"
    )

    torch.onnx.export(
        autoencoder.decoder,
        feats,
        "decoder.onnx"
    )

if __name__ == "__main__":
    test()

