import os

import torch
from torch.utils.data import DataLoader
import numpy as np
from torch import optim

from .network import DDSPNetwork
from .harmonic import HarmonicSynth
from .noise import NoiseSynth
from .loss import compute_stft, spectral_loss

__all__ = ["train", "save_checkpoint", "restore_checkpoint"]

SCHEDULER_GAMMA = 0.99


class Training:
    def __init__(self, net, dataset, batch_size=12, learning_rate=0.0001):
        self.net = net
        self.data_loader = DataLoader(dataset, batch_size, shuffle=True)

        self.harm_synth = HarmonicSynth(
            net.nb_harms,
            audio_sr=dataset.audio_sr,
            frame_length=dataset.frame_length,
            dtype=net.dtype,
            device=net.device,
        )
        self.noise_synth = NoiseSynth(
            net.nb_noise_bands,
            frame_length=dataset.frame_length,
            dtype=net.dtype,
            device=net.device,
        )
        self.cur_epoch = 0

        self.optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, SCHEDULER_GAMMA
        )

    def run_epoch(self):
        nb_batchs = len(self.data_loader)
        epoch_loss = 0

        for i, fragments in enumerate(self.data_loader, 0):
            f0 = fragments["f0"].to(self.net.device).unsqueeze(-1)
            lo = fragments["lo"].to(self.net.device).unsqueeze(-1)

            a0, aa, h0, hh = self.net.forward(f0, lo)

            f0 = f0.squeeze(-1)
            harm_wf = self.harm_synth.synthetize(f0, a0, aa)
            noise_wf = self.noise_synth.synthetize(h0, hh)
            synth_wf = harm_wf + noise_wf
            synth_stfts = compute_stft(synth_wf)
            truth_stfts = [s.to(self.net.device) for s in fragments["stfts"]]
            loss = spectral_loss(synth_stfts, truth_stfts)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        self.cur_epoch += 1
        # TODO check why
        torch.cuda.empty_cache()

        self.scheduler.step()
        epoch_loss /= nb_batchs
        return epoch_loss


def save_checkpoint(training, out_path):
    state = {
        "net": training.net.state_dict(),
        "cur_epoch": training.cur_epoch,
        "optimizer": training.optimizer.state_dict(),
        "scheduler": training.scheduler.state_dict(),
    }
    torch.save(state, out_path)


def restore_checkpoint(path, dataset, nb_harms, nb_noise_bands, dtype, device):
    state = torch.load(path, map_location=device)

    net = DDSPNetwork(nb_harms, nb_noise_bands, dtype=dtype, device=device)
    net.load_state_dict(state["net"])
    net = net.type(dtype)
    net = net.to(device)
    net.dtype = dtype
    net.device = device

    training = Training(net, dataset)

    training.optimizer = optim.Adam(net.parameters())
    training.optimizer.load_state_dict(state["optimizer"])

    training.scheduler = optim.lr_scheduler.ExponentialLR(
        training.optimizer, SCHEDULER_GAMMA
    )
    training.scheduler.load_state_dict(state["scheduler"])

    training.cur_epoch = state["cur_epoch"]

    return training

