#!/usr/bin/env python3

import os
import argparse
import pathlib

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import scipy.signal
import scipy.fft
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ddsp.model.decoder import Decoder
from ddsp.dataset import Dataset
from ddsp.training import Training, save_checkpoint  # , restore_checkpoint


def plot_wf(audio, sr):
    fig = plt.figure()
    ax = fig.subplots()
    t = np.arange(len(audio)) * 1 / sr
    ax.set_ylim((-1.0, 1.0))
    ax.plot(t, audio)
    return fig


def plot_spectrogram(audio, sr):
    eps = np.finfo(audio.dtype).eps

    fig = plt.figure()
    ax = fig.subplots()
    f, t, spec = scipy.signal.spectrogram(
        audio, sr, nperseg=2048, noverlap=1536
    )
    spec = 20 * np.log10(spec + eps)

    ax.pcolormesh(t, f, spec, cmap="inferno")
    ax.set_yscale("log")
    ax.set_yticks([20, 100, 200, 1000, 2000, 10000])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim((20, max(f)))

    return fig


def plot_f0(f0, frame_sr, audio_sr):
    fig = plt.figure()
    ax = fig.subplots()
    t = np.arange(len(audio)) * 1 / sr
    ax.plot(t, f0)

    ax.set_yscale("log")
    ax.set_yticks([20, 100, 200, 1000, 2000, 10000])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim((20, audio_sr / 2))

    return fig


def plot_lo(lo, sr):
    fig = plt.figure()
    ax = fig.subplots()
    t = np.arange(len(audio)) * 1 / sr
    ax.plot(t, lo)
    return fig


def plot_a0(a0, sr):
    fig = plt.figure()
    ax = fig.subplots()
    t = np.arange(len(audio)) * 1 / sr
    ax.plot(t, a0)
    return fig


def plot_aa(aa, sr):
    fig = plt.figure()
    ax = fig.subplots()
    t = np.arange(len(audio)) * 1 / sr
    ax.plot(t, aa)
    return fig


parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset_dir_path",
    metavar="dataset-dir",
    help="Path to the dataset folder containing wav files and preprocessed data",
)
parser.add_argument(
    "run_dir_path",
    metavar="run-dir",
    help="Path to TensorBoard run folder. Checkpoints will also be saved here",
)
parser.add_argument(
    "--bypass-harm", action="store_true", help="Bypass harmonics generation"
)
parser.add_argument(
    "--bypass-noise", action="store_true", help="Bypass noise generation"
)
parser.add_argument(
    "--nb-harms", type=int, default=100, help="Number of harmonics"
)
parser.add_argument(
    "--nb-noise-bands", type=int, default=64, help="Number of noise bands"
)
parser.add_argument(
    "--noise-amp", type=float, default=1e-3, help="Default noise amplitude"
)
parser.add_argument(
    "--frag-duration", type=float, default=4, help="Audio fragments duration"
)
parser.add_argument("--batch-size", type=int, default=12, help="Batch size")
parser.add_argument(
    "--nb-epochs", type=int, default=200, help="Number of epochs"
)
parser.add_argument(
    "--dtype",
    default="float",
    choices=["float", "double"],
    help="Data type for model params",
)
parser.add_argument(
    "--torch-autograd-anomaly",
    action="store_true",
    help="Enable autograd anomaly check",
)


args = parser.parse_args()
dataset_dir_path = args.dataset_dir_path
run_dir_path = args.run_dir_path
bypass_harm = args.bypass_harm
bypass_noise = args.bypass_noise
nb_harms = args.nb_harms
nb_noise_bands = args.nb_noise_bands
noise_amp = args.noise_amp
frag_duration = args.frag_duration
batch_size = args.batch_size
nb_epochs = args.nb_epochs
dtype = torch.float32 if args.dtype == "float" else torch.double
torch_autograd_anomaly = args.torch_autograd_anomaly

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torch.autograd.set_detect_anomaly(torch_autograd_anomaly)

pathlib.Path(run_dir_path).mkdir(parents=True, exist_ok=True)
checkpoint_path = os.path.join(run_dir_path, "checkpoint.pth")

writer = SummaryWriter(run_dir_path)

# load dataset
wav_dir_path = os.path.join(dataset_dir_path, "audio")
f0_dir_path = os.path.join(dataset_dir_path, "f0")
dataset = Dataset(
    wav_dir_path, f0_dir_path, fragment_duration=frag_duration, dtype=dtype
)

# # retrieve checkpoint or init new network
# checkpoint_path = os.path.join(checkpoint_dir_path, "checkpoint.pth")
# if os.path.exists(checkpoint_path):
#     print("Restoring checkpoint:", checkpoint_path)
#     training = restore_checkpoint(
#         checkpoint_path,
#         dataset,
#         nb_harms,
#         nb_noise_bands,
#     )
# else:
#     pathlib.Path(checkpoint_dir_path).mkdir(parents=True, exist_ok=True)

# init model and training
model = Decoder(
    bypass_harm=bypass_harm,
    bypass_noise=bypass_noise,
    nb_harms=nb_harms,
    nb_noise_bands=nb_noise_bands,
    default_noise_amp=noise_amp,
    audio_sr=dataset.audio_sr,
    frame_sr=dataset.frame_sr,
    dtype=dtype,
)

training = Training(model, dataset, batch_size=batch_size)

# plot ref audio and waveform
ref_audio, ref_f0, ref_f0_scaled, ref_lo = dataset.get_sample(0)
ref_audio = ref_audio.numpy()
# max 5 secs
ref_audio = ref_audio[: dataset.audio_sr * 5]
ref_f0 = ref_f0[: dataset.frame_sr * 5]
ref_f0_scaled = ref_f0_scaled[: dataset.frame_sr * 5]
ref_lo = ref_lo[: dataset.frame_sr * 5]

writer.add_audio("Groundtruth", ref_audio, 0, dataset.audio_sr)
fig = plot_wf(ref_audio, dataset.audio_sr)
writer.add_figure("Waveform/Groundtruth", fig, 0, True)
fig = plot_spectrogram(ref_audio, dataset.audio_sr)
writer.add_figure("Spectrogram/Groundtruth", fig, 0, True)
writer.flush()

best_epoch_loss = torch.finfo(dtype).max

while training.epoch < nb_epochs:
    # run training and test epoch
    training.train_epoch()
    training.test_epoch()

    # plot losses
    writer.add_scalar("Loss/Train", training.train_loss, training.epoch)
    writer.add_scalar("Loss/Test", training.test_loss, training.epoch)

    # save checkpoint if test loss is better
    if training.test_loss < best_epoch_loss:
        best_epoch_loss = training.test_loss
        save_checkpoint(training, out_path=checkpoint_path)

    # plot ref audio resynth
    resynth_audio, resynth_harm, resynth_noise = model.infer(
        ref_f0, ref_f0_scaled, ref_lo, to_numpy=True
    )
    writer.add_audio(
        "Resynth/Full", resynth_audio, training.epoch, dataset.audio_sr
    )
    writer.add_audio(
        "Resynth/Harm", resynth_harm, training.epoch, dataset.audio_sr
    )
    writer.add_audio(
        "Resynth/Noise", resynth_noise, training.epoch, dataset.audio_sr
    )

    fig = plot_wf(resynth_audio, dataset.audio_sr)
    writer.add_figure("Waveform/Resynth", fig, training.epoch, True)

    fig = plot_spectrogram(resynth_audio, dataset.audio_sr)
    writer.add_figure("Spectrogram/Resynth", fig, training.epoch, True)
    writer.flush()

writer.close()
