import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


__all__ = ["DDSPNetwork"]


class MLP(nn.Module):
    """MLP helper module"""

    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()

        self.dense1 = nn.Linear(input_size, output_size)
        self.norm1 = nn.LayerNorm(output_size)

        self.dense2 = nn.Linear(output_size, output_size)
        self.norm2 = nn.LayerNorm(output_size)

        self.dense3 = nn.Linear(output_size, output_size)
        self.norm3 = nn.LayerNorm(output_size)

    def forward(self, x):
        y = F.relu(self.norm1(self.dense1(x)))
        y = F.relu(self.norm2(self.dense2(y)))
        y = F.relu(self.norm3(self.dense3(y)))

        return y


LOG_10 = np.log(10.0)


class DDSPNetwork(nn.Module):
    """Main DDSP module, following original architecture in the paper.
    No z space."""

    def __init__(self, nb_harms, nb_noise_bands, dtype, device):
        super(DDSPNetwork, self).__init__()

        self.nb_harms = nb_harms
        self.nb_noise_bands = nb_noise_bands

        self.mlp_f0 = MLP(1, 512)
        self.mlp_lo = MLP(1, 512)
        self.gru = nn.GRU(2 * 512, 512, batch_first=True)
        # self.gru = nn.GRU(512, 512, batch_first=True)
        self.mlp = MLP(512, 512)
        self.dense_harm = nn.Linear(512, nb_harms + 1)
        self.dense_noise = nn.Linear(512, nb_noise_bands + 1)

        self.dtype = dtype
        self.device = device

    def forward(self, f0, lo):
        y_f0 = self.mlp_f0(f0)
        y_lo = self.mlp_lo(lo)

        y = torch.cat((y_f0, y_lo), dim=2)
        y = self.gru(y)[0]
        y = self.mlp(y)

        y_harm = self.dense_harm(y)
        y_harm = 2.0 * torch.sigmoid(y_harm) ** LOG_10 + 1e-7
        y_noise = self.dense_noise(y)
        y_noise = 2.0 * torch.sigmoid(y_noise) ** LOG_10 + 1e-7

        a0 = y_harm[..., 0]
        aa = y_harm[..., 1:]
        aa = aa.transpose(1, 2)
        h0 = y_noise[..., 0]
        hh = y_noise[..., 1:]
        hh = hh.transpose(1, 2)

        return a0, aa, h0, hh
