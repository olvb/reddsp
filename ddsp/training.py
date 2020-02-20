import os
from functools import lru_cache

import torch
from torch.utils.data import DataLoader
import torch.optim
import numpy as np

from .model.decoder import Decoder
from .spectral import MultiScaleSTFT
from .loss import SpectralLinLogLoss

__all__ = ["Training", "save_checkpoint", "restore_checkpoint"]


class Training:
    def __init__(
        self,
        model,
        dataset,
        batch_size=12,
        train_test_split=0.8,
        learning_rate=1e-3,
        scheduler_gamma=0.98,
        device=torch.device("cpu"),
    ):
        self.model = model.to(device=device)

        # split into train/test
        train_length = round(train_test_split * len(dataset))
        test_length = len(dataset) - train_length
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, (train_length, test_length)
        )

        self.train_data_loader = DataLoader(
            train_dataset, batch_size, shuffle=True
        )
        self.test_data_loader = DataLoader(
            test_dataset, batch_size, shuffle=False
        )

        self.epoch = 0
        self.train_loss = None
        self.test_loss = None

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, scheduler_gamma
        )

        self.spectral_transform = MultiScaleSTFT()
        self.spectral_loss = SpectralLinLogLoss(dtype=model.dtype)

        self.device = device

    def train_epoch(self):
        self.train_loss = self._run_epoch(self.train_data_loader, train=True)
        self.epoch += 1
        self.scheduler.step()

    def test_epoch(self):
        self.test_loss = self._run_epoch(self.test_data_loader, train=False)

    def _run_epoch(self, data_loader, train=True):
        torch.set_grad_enabled(train)

        nb_batchs = len(data_loader)
        loss_sum = 0

        for fragment in data_loader:
            truth_audio = fragment["audio"]
            f0 = fragment["f0"]
            f0_scaled = fragment["f0_scaled"]
            lo = fragment["lo"]

            # send to device
            truth_audio = truth_audio.to(device=self.device)
            f0 = f0.to(device=self.device)
            f0_scaled = f0_scaled.to(device=self.device)
            lo = lo.to(device=self.device)

            resynth_audio, _, _ = self.model.forward(f0, f0_scaled, lo)
            resynth_stfts = self.spectral_transform(resynth_audio)
            truth_stfts = self._get_truth_spectral(truth_audio)
            # TODO send truth stft to proper device
            # [s.to(self.model.device) for s in fragments["stfts"]]
            loss = self.spectral_loss(resynth_stfts, truth_stfts)

            loss_sum += loss.item()

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        loss = loss_sum / nb_batchs
        return loss

    @lru_cache()
    def _get_truth_spectral(self, fragments_audio):
        return self.spectral_transform(fragments_audio)


def save_checkpoint(training, out_path):
    state = {
        "model_state_dict": training.model.state_dict(),
        "optimizer_state_dict": training.optimizer.state_dict(),
        "scheduler_state_dict": training.scheduler.state_dict(),
        "epoch": training.epoch,
        "test_loss": training.test_loss,
    }
    torch.save(state, out_path)


# def restore_checkpoint(path, dataset, device=torch.device("cpu")):
#     state = torch.load(path, map_location=device)

#     model = Decoder()
#     model.load_state_dict(state["model_state_dict"])

#     training = Training(decoder, dataset)

#     training.optimizer = torch.optim.Adam(decoder.parameters())
#     training.optimizer.load_state_dict(state["optimizer_state_dict"])

#     scheduler_gamma = state["scheduler_state_dict"]["gama"]
#     training.scheduler = torch.optim.lr_scheduler.ExponentialLR(
#         training.optimizer, scheduler_gamma
#     )
#     training.scheduler.load_state_dict(state["scheduler_state_dict"])

#     training.epoch = state["epoch"]
#     traing.epoch_loss = state["epoch_loss"]

#     return training
