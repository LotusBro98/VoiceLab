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

from upscale_model import UpsamplerTrainable


class ResBlock(nn.Module):
    def __init__(self, cin, cout, ksize=33, bottleneck=1, stride=1):
        super().__init__()
        cmid = int(round(bottleneck * max(cin, cout)))
        self.do_skip_conv = cin != cout or stride != 1

        self.conv1 = nn.Conv1d(cin, cmid, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(cmid)

        self.conv2 = nn.Conv1d(cmid, cmid, ksize, stride=stride, groups=cmid, bias=False, padding=(ksize-1)//2)
        self.bn2 = nn.BatchNorm1d(cmid)

        self.conv3 = nn.Conv1d(cmid, cout, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(cout)

        self.conv_skip = nn.Conv1d(cin, cout, 1, stride=stride, bias=False) if self.do_skip_conv else None
        self.bn_skip = nn.BatchNorm1d(cout) if self.do_skip_conv else None

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


class Encoder(nn.Module):
    def __init__(self, d_model=128, d_spec=80, d_emb=512):
        super().__init__()

        self.layers = nn.ModuleList([
            ResBlock(d_spec, d_model),
            ResBlock(d_model, d_model, stride=2),
            ResBlock(d_model, d_model, stride=2),
            ResBlock(d_model, d_emb),
        ])

    def forward(self, spec: torch.Tensor):
        x = spec

        for layer in self.layers:
            x = layer(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model=128, d_spec=80, d_emb=512):
        super().__init__()

        self.layers = nn.ModuleList([
            ResBlock(d_emb, d_model),
            nn.Upsample(scale_factor=2),
            ResBlock(d_model, d_model),
            nn.Upsample(scale_factor=2),
            ResBlock(d_model, d_model),
            ResBlock(d_model, d_spec),
        ])

    def forward(self, emb: torch.Tensor):
        x = emb

        for layer in self.layers:
            x = layer(x)

        spec = F.gelu(x)
        return spec

class Autoencoder(pl.LightningModule):
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

        self.encoder = Encoder()
        self.decoder = Decoder()

    def training_step(self, batch: torch.Tensor, batch_idx):
        chunk, sr = batch

        spec_true = self.builder_encode.encode(chunk)

        emb = self.encoder(spec_true)
        spec_pred = self.decoder(emb)

        loss_ae = F.mse_loss(spec_pred[..., :spec_true.shape[-1]], spec_true[..., :spec_pred.shape[-1]]).sqrt() / spec_true.std()
        loss = loss_ae

        self.log("Lae", loss_ae, prog_bar=True)
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

    model = Autoencoder(sample_rate=sample_rate)

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
    model.to("cuda:1").eval()

    ### Test
    track, _ = dataset[10]
    track = track.to("cuda")

    spec0 = model.builder_encode.encode(track)
    with torch.no_grad():
        emb = model.encoder(spec0[None])
        spec1 = model.decoder(emb)[0]

    codec = UpsamplerTrainable()
    codec.model.load_state_dict(torch.load("upsampler.pth"))
    codec.model.eval()

    track2 = codec.decode(spec1)
    write("track.wav", sample_rate, (track2.cpu().numpy()*2**31).astype(np.int32))
    
    f, ax = plt.subplots(3, figsize=(15, 10))
    ax[0].imshow(complex_picture(spec0)[::-1], aspect=1, interpolation="nearest")
    ax[1].imshow(complex_picture(spec1)[::-1], aspect=1, interpolation="nearest")
    ax[2].imshow(complex_picture(spec1[..., :spec0.shape[-1]] - spec0[..., :spec1.shape[-1]])[::-1], aspect=1, interpolation="nearest")
    plt.savefig("complex_pic.png")
    plt.close()


if __name__ == "__main__":
    main()
