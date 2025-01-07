import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F

from process import freq_to_sample, sample_to_freq, FREQ_RES, get_window

def get_voice_basis(n_freqs):
    min_voice = 50
    max_voice = 1000
    base_freqs = sample_to_freq(
        torch.linspace(
            freq_to_sample(min_voice), 
            freq_to_sample(max_voice),
            200
        )
    )

    basis = [torch.zeros(n_freqs)]
    for freq in base_freqs:
        harmonics = torch.zeros(n_freqs)
        for i in range(1, 20):
            fni = int(freq_to_sample(freq * i))
            if fni >= n_freqs:
                break
            win = get_window(n_freqs, FREQ_RES)
            harmonics += win.roll(fni)
        basis.append(harmonics)
    basis = torch.stack(basis, dim=0)

    return basis


def smoothness_loss(x: torch.Tensor, df=5, dt=1):
    loss = 0
    loss += df * x.diff(1, dim=-1).abs().square().mean().sqrt() / x.std()
    loss += dt * x.diff(2, dim=-2).abs().square().mean().sqrt() / x.std()
    return loss


class VoicePrint(nn.Module):
    def __init__(self, n_freqs, n_time):
        super().__init__()

        self.basis = get_voice_basis(n_freqs)
        self.main_voice = nn.Parameter(torch.randn(n_time, self.basis.shape[0]))
        self.res_curve = nn.Parameter(torch.randn(n_time, n_freqs))
        self.noise_mask = nn.Parameter(torch.randn(n_time, n_freqs))

        self.normalize()

    def forward(self, spectrum):
        main_voice = F.softmax(self.main_voice, dim=-1)

        pred = (main_voice @ self.basis) * self.res_curve.exp()

        loss = F.mse_loss(pred, spectrum)

        loss += smoothness_loss(self.res_curve)
        # loss += smoothness_loss(self.noise_mask)

        return loss, pred
    
    @torch.no_grad()
    def normalize(self):
        # res_curve = self.res_curve.clone()
        # res_curve = torch.fft.fft(res_curve, dim=-1)
        # n = 5
        # res_curve[..., n:-n+1] *= 0
        # self.res_curve[:] = torch.fft.ifft(res_curve, dim=-1).abs()

        self.res_curve[:] -= self.res_curve.mean(dim=(-1, -2), keepdim=True)
        self.res_curve[:] /= self.res_curve.std(dim=(-1, -2), keepdim=True)

        self.main_voice[:] -= self.main_voice.mean(dim=-1, keepdim=True)


def extract_voice(spectrum):
    model = VoicePrint(spectrum.shape[1], spectrum.shape[0])

    optim = torch.optim.Adam(model.parameters(), lr=1e-1)

    
    for ep in range(2000):
        optim.zero_grad()

        loss, pred = model(spectrum.abs())
        loss.backward()

        optim.step()
        print(ep, loss.item())

        model.normalize()

    f, ax = plt.subplots(4, 1, figsize=(5, 10))
    ax[0].imshow(model.res_curve.exp().detach().cpu())
    ax[1].imshow(F.softmax(model.main_voice, dim=-1).detach().cpu())
    ax[2].imshow(spectrum.abs())
    # ax[3].imshow(model.basis.detach().cpu())
    ax[3].imshow(pred.detach().cpu())
    plt.savefig("model.png")
    plt.cla()

    return pred.detach().cpu().numpy()
