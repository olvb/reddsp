import torch
from torch.nn import functional as F
import numpy as np


__all__ = ["SpectralLinLogLoss"]


class SpectralLinLogLoss:
    def __init__(self, dtype=torch.float):
        self.epsilon = torch.finfo(dtype).eps

    def __call__(self, stft_resynth_all, stft_truth_all):
        loss = 0.0

        for stft_resynth, stft_truth in zip(stft_resynth_all, stft_truth_all):
            stft_resynth_log = torch.log(stft_resynth + self.epsilon)
            stft_truth_log = torch.log(stft_truth + self.epsilon)

            loss_lin = F.l1_loss(stft_resynth, stft_truth, reduction="none")
            loss_log = F.l1_loss(
                stft_resynth_log, stft_truth_log, reduction="none"
            )

            # mean on spectrogram dims, sum on batch
            # TODO problematic because loss magnitude varies with batch size?
            loss_lin = torch.sum(torch.mean(loss_lin, dim=(1, 2)))
            loss_log = torch.sum(torch.mean(loss_log, dim=(1, 2)))

            loss += loss_lin + loss_log

        return loss
