import torch
from torch.nn import functional as F
import numpy as np


__all__ = ["compute_stft", "spectral_loss"]

EPSILON = 10 ** (-96 / 20)


# TODO check magnitude
def spectral_loss(stft_synth_all, stft_truth_all):
    loss = 0.0

    for stft_synth, stft_truth in zip(stft_synth_all, stft_truth_all):
        stft_synth_log = torch.log(stft_synth + EPSILON)
        stft_truth_log = torch.log(stft_truth + EPSILON)

        loss_lin = F.l1_loss(stft_synth, stft_truth, reduction="mean")
        loss_log = F.l1_loss(stft_synth_log, stft_truth_log, reduction="mean")

        loss += loss_lin + loss_log

    return loss


# TODO investigate normalization
def compute_stft(
    wf, fft_sizes=[2048, 1024, 512, 256, 128, 64], normalized=False
):
    """Computes multiscale stfts to use for  multiscale spectral loss."""
    stft_all = []

    for fft_size in fft_sizes:
        stft = torch.stft(
            wf,
            fft_size,
            hop_length=fft_size // 4,
            center=True,
            pad_mode="reflect",
            normalized=normalized,
            onesided=True,
        )
        stft = torch.sum(stft ** 2, dim=-1)
        # TODO check this
        stft = stft[:, 1, :]  # remove DC
        stft_all.append(stft)

    return stft_all



