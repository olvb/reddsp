import torch
from torch.nn import functional as F
import numpy as np


__all__ = ["SpectralLinLogLoss"]


class SpectralLinLogLoss:
    def __init__(self, dtype=torch.float32):
        self.l1_loss = torch.nn.L1Loss(reduction="none")
        self.dtype = dtype
        self.epsilon = torch.finfo(dtype).eps

    def __call__(self, stft_synth_all, stft_truth_all):
        loss = torch.tensor(0.0, dtype=self.dtype)

        for stft_synth, stft_truth in zip(stft_synth_all, stft_truth_all):
            stft_synth_log = torch.log(stft_synth + self.epsilon)
            stft_truth_log = torch.log(stft_truth + self.epsilon)

            loss_lin = self.l1_loss(stft_synth, stft_truth)
            loss_log = self.l1_loss(stft_synth_log, stft_truth_log)

            # # sum on time and freq, mean on batch
            loss_lin = torch.mean(torch.sum(torch.mean(loss_lin, 2), 1))
            loss_log = torch.mean(torch.sum(torch.mean(loss_log, 2), 1))

            loss += loss_lin + loss_log

        loss /= len(stft_synth_all)

        return loss
