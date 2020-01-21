import torch
from torch.nn import functional as F
import numpy as np


__all__ = ["compute_stft", "spectral_loss"]

EPSILON = 10 ** (-96 / 20)


# def spectral_loss(stft_synth_all, stft_truth_all):
#     loss = 0.0

#     for stft_synth, stft_truth in zip(stft_synth_all, stft_truth_all):
#         stft_synth_log = torch.log(stft_synth + EPSILON)
#         stft_truth_log = torch.log(stft_truth + EPSILON)

#         loss_lin = F.l1_loss(stft_synth, stft_truth, reduction="mean")
#         loss_log = F.l1_loss(stft_synth_log, stft_truth_log, reduction="mean")

#         loss += loss_lin + loss_log

#     return loss


# def compute_stft(
#     wf, fft_sizes=[2048, 1024, 512, 256, 128, 64], normalized=False
# ):
#     """Computes multiscale stfts to use for  multiscale spectral loss."""
#     stft_all = []

#     for fft_size in fft_sizes:
#         stft = torch.stft(
#             wf,
#             fft_size,
#             hop_length=fft_size // 4,
#             center=True,
#             pad_mode="reflect",
#             normalized=normalized,
#             onesided=True,
#         )
#         stft = torch.sum(stft ** 2, dim=-1)
#         stft = stft[:, 1, :]  # remove DC
#         stft_all.append(stft)

#     return stft_all


# Lambda for computing squared amplitude
amp = lambda x: x[..., 0] ** 2 + x[..., 1] ** 2


def compute_stft(waveform, fft_sizes=[2048, 1024, 512, 256, 128, 64]):
    """Computes multiscale stfts to use for  multiscale spectral loss."""
    stfts = []
    overlap = 0.75

    for fft_size in fft_sizes:
        win = torch.hann_window(fft_size)
        stft = torch.stft(
            waveform,
            n_fft=fft_size,
            window=win,
            hop_length=int((1 - overlap) * fft_size),
            center=False,
        )
        stfts.append(amp(stft))

    return stfts


def spectral_loss(stft_synth_all, stft_truth_all):
    """Computes multiscale spectral loss before reconstructed and groundtruth signals.
    Excepts a list of tensors containing stfts computed with different fft sizes."""

    lin_loss = 0
    log_loss = 0.0

    nb_ffts = len(stft_synth_all)
    for i in range(nb_ffts):
        stft_synth = stft_synth_all[i]
        stft_truth = stft_truth_all[i]

        nb_batchs = stft_synth.size()[0]
        for j in range(nb_batchs):
            lin_diff = stft_truth[j] - stft_synth[j]
            lin_loss += torch.mean(torch.abs(lin_diff))
            log_diff = torch.log(stft_truth[j] + 1e-4) - torch.log(
                stft_synth[j] + 1e-4
            )
            log_loss += torch.mean(torch.abs(log_diff))

    return lin_loss + log_loss
