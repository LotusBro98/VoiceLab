import os
import librosa
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import AudioDataset
from process import SpectrogramBuilder, complex_picture
from scipy.io.wavfile import write
import pytorch_lightning as pl


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
    def __init__(self, d_model=128, n_hid_layers=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)

        self.layers = nn.ModuleList([
            nn.Conv2d(1, d_model, kernel_size=3, padding="same"),
            *(
                ResBlock(d_model, d_model)
                for i in range(n_hid_layers)
            ),
            nn.Conv2d(d_model, 2, kernel_size=3, padding="same"),
        ])

        self.basic_phase_k = nn.Parameter(torch.ones(160))

    def forward(self, spec: torch.Tensor):
        spec_shape = spec.shape
        spec = spec.reshape(-1, 1, *spec_shape[-2:])
        # x = torch.cat([spec.real, spec.imag], dim=1)
        x = torch.cat([spec.abs()], dim=1)

        x = self.upsample(x)
        for layer in self.layers:
            x = layer(x)
        
        # spec = torch.complex(x[:, 0:1], x[:, 1:2])
        ampl = F.gelu(x[:, 0:1])
        phase = x[:, 1:2] * self.basic_phase_k[:, None]
        spec = ampl * torch.complex(torch.cos(phase), torch.sin(phase))
        spec = spec.reshape(spec_shape[:-2] + spec.shape[-2:])
        return spec


def complex_mse_loss(pred, true, normalize=True, eps=1e-3):
    loss = (pred - true).abs().square().mean().sqrt()
    if normalize:
        loss = loss / true.std().clip(eps, None)
    return loss

class UpsamplerTrainable(pl.LightningModule):
    def __init__(self, sample_rate=22050):
        super().__init__()

        self.sample_rate = sample_rate

        self.builder_encode = SpectrogramBuilder(
            sample_rate,
            fsave=200,
            n_feats=80,
            freq_res=1,
            magnitude=True,
            combo_scale=False,
            use_noise_masking=False,
        )

        self.builder_decode = SpectrogramBuilder(
            sample_rate,
            fsave=400,
            n_feats=160,
            freq_res=2,
            magnitude=False,
            combo_scale=False,
            use_noise_masking=False,
        )

        self.model = SpecUpsampler().to("cuda")

    @torch.no_grad
    def encode(self, signal):
        spec = self.builder_encode.encode(signal)
        return spec
    
    @torch.no_grad
    def decode(self, spec):
        spec_pred = self.model(spec)
        signal = self.builder_decode.decode(spec_pred)
        return signal

    def training_step(self, batch: torch.Tensor, batch_idx):
        chunk, sr = batch

        spec_encode = self.builder_encode.encode(chunk)
        spec_decode = self.builder_decode.encode(chunk)

        spec_pred = self.model(spec_encode)

        signal_true, noise_true, K, nf = self.builder_decode.signal_noise_decomposition(spec_decode)
        signal_pred, noise_pred, K, nf = self.builder_decode.signal_noise_decomposition(spec_pred, K=K, n_feats=nf)

        loss_ae = complex_mse_loss(signal_pred, signal_true)
        loss_ae += complex_mse_loss(noise_pred, noise_true)
        # loss_ae = complex_mse_loss(spec_pred, spec_decode)
        # loss_ae += 0.1 * F.mse_loss(spec_pred.abs(), spec_decode.abs()).sqrt() / spec_decode.std()
        # loss_ae += 0.1 * (spec_pred / spec_pred.abs().clip(1e-6, None) - spec_decode / spec_decode.abs().clip(1e-6, None)).abs().square().mean().sqrt()

        # loss_disc = self.discriminator.loss_disc(spec, pred_spec)
        # loss = 100 * loss_ae + loss_disc
        loss = loss_ae

        self.log("Lae", loss_ae, prog_bar=True)
        # self.log("Ld", loss_disc, prog_bar=True)
        self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True)
        return loss

    def configure_optimizers(self):
        base_lr = 3e-4

        opt_g = optim.AdamW(self.parameters(), lr=base_lr, weight_decay=0.01)

        T1 = 100
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



def main():
    sample_rate = 22050
    chunk_len=4

    dataset = AudioDataset("data/podcast.mp3", chunk_len=chunk_len, sample_rate=sample_rate)
    train_loader = DataLoader(
        dataset, 
        batch_size=4, 
        num_workers=os.cpu_count(), 
        persistent_workers=True
    )

    model = UpsamplerTrainable(sample_rate=sample_rate)

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        # limit_train_batches=10000, 
        max_epochs=5,
        # accelerator="cpu"
        devices=[1],
        # callbacks=[
        #     # EarlyStopping("Lae", min_delta=0.01, patience=3)
        # ]
    )

    trainer.fit(model=model, train_dataloaders=train_loader)
    torch.save(model.model.state_dict(), "upsampler.pth")
    model.to("cuda:1").eval()

    ### Test

    print(model.model.basic_phase_k)

    track, _ = dataset[10]
    track = track.to("cuda")

    spec0 = model.builder_encode.encode(track)
    with torch.no_grad():
        spec1 = model.model(spec0)

    spec2 = model.builder_decode.encode(track)

    track2 = model.builder_decode.decode(spec1)
    write("track.wav", sample_rate, (track2.cpu().numpy()*2**31).astype(np.int32))
    
    f, ax = plt.subplots(3, figsize=(15, 10))
    ax[0].imshow(complex_picture(spec0)[::-1], aspect=1, interpolation="nearest")
    ax[1].imshow(complex_picture(model.builder_decode.from_freq_diff_repr(spec1))[::-1], aspect=1, interpolation="nearest")
    ax[2].imshow(complex_picture(spec1 - spec2)[::-1], aspect=1, interpolation="nearest")
    plt.savefig("complex_pic.png")
    plt.close()


if __name__ == "__main__":
    main()
