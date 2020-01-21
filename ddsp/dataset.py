import os
import glob

# import math

import torch
from torch.utils.data import Dataset as ParentDataset
import scipy.io.wavfile
import librosa
import numpy as np
import pandas as pd

from .loss import compute_stft

import scipy.signal

EPSILON = 10 ** (-96 / 20)


# def low_pass(signal, freq_cut, order=1):
#     b, a = scipy.signal.butter(order, freq_cut, btype="low", analog=False)
#     signal_lp = scipy.signal.filtfilt(b, a, signal, axis=signal.ndim - 1)

#     signal_lp = np.flip(signal_lp, -1)
#     signal_lp = np.flip(signal_lp, -1)
#     signal_lp = signal_lp.copy()
#     signal_lp = signal_lp.astype(signal.dtype)

#     return signal_lp


class Dataset(ParentDataset):
    """Audio dataset class, excepts an audio subfolder containing wave files
    and a f0 subfolder containing f0 csv computed with CREPE."""

    def __init__(
        self, path, audio_sr, frame_sr, fragment_duration=4, dtype=torch.float
    ):
        if not os.path.isdir(path):
            raise Exception("Could not open dataset at path: {}".format(path))

        self.path = path
        self.audio_sr = audio_sr
        self.frame_sr = frame_sr
        self.frame_length = audio_sr // frame_sr
        self.fragment_duration = fragment_duration

        self.dtype = dtype

        self.mean_lo = None
        self.std_lo = None

        self.fragments = self.compute_fragments()

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, i):
        return self.fragments[i]

    def get_audio_paths(self):
        audio_pattern = os.path.join(self.path, "audio", "*.wav")
        audio_paths = sorted(glob.glob(audio_pattern))
        return audio_paths

    def get_f0_paths(self):
        f0_pattern = os.path.join(self.path, "f0", "*.csv")
        f0_paths = sorted(glob.glob(os.path.join(f0_pattern)))
        return f0_paths

    def get_test_lo_f0(self, duration=10):
        """Returns loudness and f0 vectors for sample file to use for network evaluation."""
        audio_path = self.get_audio_paths()[0]
        lo = read_lo(audio_path, self.audio_sr, self.frame_length)
        lo = lo[: duration * self.frame_sr]  # trim to duration

        if self.mean_lo is None:
            self.mean_lo, self.stddev_lo = get_lo_stats(
                audio_paths, self.audio_sr, self.frame_length
            )
        # lo -= mean_lo  # center loudness around mean value
        # lo /= np.std(lo)  # rescale in [-1, 1]

        f0_path = self.get_f0_paths()[0]
        f0 = read_f0(f0_path)
        f0 = f0[: duration * self.frame_sr]  # trim to duration

        f0_log = np.log(f0) / np.log(self.audio_sr / 2)

        return lo, f0, f0_log

    def get_test_waveform(self, duration=10):
        """Returns waveform for sample file to use for network evaluation"""
        audio_path = self.get_audio_paths()[0]
        wf = read_waveform(audio_path, self.audio_sr)
        wf = wf[: duration * self.audio_sr]  # trim to duration

        return wf

    def compute_fragments(self):
        """Compute loudness and f0 vector to pass as input to network,
        and multiscale stfts of reference audio to use for loss computation"""
        audio_paths = self.get_audio_paths()
        f0_paths = self.get_f0_paths()

        if self.mean_lo is None:
            self.mean_lo, self.stddev_lo = get_lo_stats(
                audio_paths, self.audio_sr, self.frame_length
            )

        fragments = []

        nb_frames_per_frag = self.frame_sr * self.fragment_duration
        nb_samples_per_frag = self.audio_sr * self.fragment_duration

        for audio_path, f0_path in zip(audio_paths, f0_paths):
            wf = read_waveform(audio_path, self.audio_sr)
            lo = read_lo(
                audio_path,
                audio_sr=self.audio_sr,
                frame_length=self.frame_length,
            )
            # lo -= mean_lo  # center loudness around mean value
            # lo /= np.std(lo)  # rescale in [-1, 1]
            f0 = read_f0(f0_path)

            nb_frags = min(
                len(lo) // nb_frames_per_frag,
                len(f0) // nb_frames_per_frag,
                len(wf) // nb_samples_per_frag,
            )

            lo = lo[: nb_frags * nb_frames_per_frag]
            lo = lo.reshape(nb_frags, -1)

            f0 = f0[: nb_frags * nb_frames_per_frag]
            f0 = f0.reshape(nb_frags, -1)

            wf = wf[: nb_frags * nb_samples_per_frag]
            wf = wf.reshape(nb_frags, -1)

            f0 = torch.tensor(f0).type(self.dtype)
            lo = torch.tensor(lo).type(self.dtype)
            wf = torch.tensor(wf).type(self.dtype)

            f0_log = torch.log(f0) / np.log(self.audio_sr / 2)

            stfts = compute_stft(wf)

            for i in range(nb_frags):
                frag_f0 = f0[i]
                frag_f0_log = f0_log[i]
                frag_lo = lo[i]
                # frag_wf = wf[i]
                frag_stfts = [s[i] for s in stfts]

                frag = {
                    "f0": frag_f0,
                    "f0_log": frag_f0_log,
                    "lo": frag_lo,
                    "stfts": frag_stfts,
                }
                fragments.append(frag)

        return fragments



def read_f0(path):
    f0 = pd.read_csv(path, header=0)
    f0 = f0.to_numpy()
    f0 = f0[:-1, 1]
    f0 = f0.astype(np.float32)

    return f0


def read_lo(path, audio_sr, frame_length):
    wf = read_waveform(path, audio_sr)

    amp = librosa.feature.rms(
        wf,
        hop_length=frame_length,
        frame_length=frame_length * 2,
        center=False,
    )
    amp = amp.astype(np.float)
    lo = amp.flatten()
    return lo


def get_lo_stats(audio_paths, audio_sr, frame_length):
    lo_all = [read_lo(p, audio_sr, frame_length) for p in audio_paths]
    lo_all = np.concatenate(lo_all, axis=0)
    return np.mean(lo_all), np.abs(np.std(lo_all))


def read_waveform(path, audio_sr):
    sr, wf = scipy.io.wavfile.read(path)
    assert sr == audio_sr
    # int to float
    dtype = wf.dtype
    if np.issubdtype(dtype, np.integer):
        wf = wf.astype(np.float32) / np.iinfo(dtype).max

    return wf


if __name__ == "__main__":
    compute_cache()
