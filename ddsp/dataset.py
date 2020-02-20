import os
import glob

import torch
import scipy.io.wavfile
import numpy as np

from .model.pitch import CREPEPitch
from .model.preprocess import Preprocessor

__all__ = ["preprocess_dataset_f0", "Dataset"]


def preprocess_dataset_f0(wav_dir_path, f0_dir_path, audio_sr, frame_sr):
    pitch = CREPEPitch()

    wav_pattern = os.path.join(wav_dir_path, "*.wav")
    wav_paths = sorted(glob.glob(wav_pattern))

    for i, wav_path in enumerate(wav_paths):
        audio = read_wav(wav_path, audio_sr)
        f0 = pitch(audio, audio_sr, frame_sr)

        f0_path = os.path.join(f0_dir_path, "{:d}.pth".format(i + 1))
        torch.save(f0, f0_path)

    # store f0 sample rate
    sr_path = os.path.join(f0_dir_path, "frame_sr.pth")
    torch.save(frame_sr, sr_path)


class Dataset(torch.utils.data.Dataset):
    """Audio dataset class, excepts an audio subfolder containing wave files
    and a f0 subfolder containing f0 csv computed with CREPE."""

    def __init__(
        self,
        wav_dir_path,
        f0_dir_path,
        preprocessor=Preprocessor(),
        fragment_duration=4,
        dtype=torch.float32,
    ):
        super().__init__()

        self.wav_dir_path = wav_dir_path
        self.f0_dir_path = f0_dir_path
        self.preprocessor = preprocessor
        self.fragment_duration = fragment_duration
        self.dtype = dtype

        self.audio_sr = self._load_audio_sr()
        self.frame_sr = self._load_frame_sr()
        self.fragments = self._load_fragments()

    def _load_audio_sr(self):
        """ Loads audio sample rate by opening a wav file """
        wav_paths, _ = self.get_sample_paths()
        wav_path = wav_paths[0]
        audio_sr, _ = scipy.io.wavfile.read(wav_path)
        return audio_sr

    def _load_frame_sr(self):
        """ Loads frame sample rate for preprocessed pitch values """
        f0_sr_path = os.path.join(self.f0_dir_path, "frame_sr.pth")
        f0_frame_sr = torch.load(f0_sr_path)

        return f0_frame_sr

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, i):
        return self.fragments[i]

    def _load_fragments(self):
        wav_paths, f0_paths = self.get_sample_paths()

        fragments = []

        for i in range(len(wav_paths)):
            audio, f0, f0_scaled, lo = self.get_sample(i, wav_paths, f0_paths)

            fragments += self._split_into_fragments(audio, f0, f0_scaled, lo)

        return fragments

    def get_sample(self, i, wav_paths=None, f0_paths=None):
        """ Return unfragmented audio sample. Useful for evaluation """
        if wav_paths is None or f0_paths is None:
            wav_paths, f0_paths = self.get_sample_paths()

        wav_path = wav_paths[i]
        f0_path = f0_paths[i]

        audio = read_wav(wav_path, self.audio_sr)
        f0 = torch.load(f0_path)

        f0, f0_scaled, lo = self.preprocessor(
            audio, self.audio_sr, self.frame_sr, f0=f0
        )

        audio = torch.from_numpy(audio)

        # convert to requested type (float or double)
        audio = audio.type(self.dtype)
        f0 = f0.type(self.dtype)
        f0_scaled = f0_scaled.type(self.dtype)
        lo = lo.type(self.dtype)

        return audio, f0, f0_scaled, lo

    def get_sample_paths(self):
        wav_pattern = os.path.join(self.wav_dir_path, "*.wav")
        wav_paths = sorted(glob.glob(wav_pattern))
        assert len(wav_paths) > 0, "No wav file found in {}".format(
            self.wav_dir_path
        )

        f0_file_pattern = os.path.join(self.f0_dir_path, "[0-9]*.pth")
        f0_paths = sorted(glob.glob(f0_file_pattern))
        assert len(f0_paths) > 0, "No f0 file found in {}".format(
            self.f0_dir_path
        )

        return wav_paths, f0_paths

    def _split_into_fragments(self, audio, f0, f0_scaled, lo):
        nb_samples_per_frag = self.audio_sr * self.fragment_duration
        nb_frames_per_frag = self.frame_sr * self.fragment_duration

        nb_frags = min(
            len(audio) // nb_samples_per_frag, len(f0) // nb_frames_per_frag
        )

        audio = audio[: nb_frags * nb_samples_per_frag]
        audio = audio.reshape(nb_frags, -1)

        f0 = f0[: nb_frags * nb_frames_per_frag]
        f0 = f0.reshape(nb_frags, -1)

        f0_scaled = f0_scaled[: nb_frags * nb_frames_per_frag]
        f0_scaled = f0.reshape(nb_frags, -1)

        lo = lo[: nb_frags * nb_frames_per_frag]
        lo = lo.reshape(nb_frags, -1)

        fragments = []
        for i in range(nb_frags):
            frag_audio = audio[i]
            frag_f0 = f0[i]
            frag_f0_scaled = f0_scaled[i]
            frag_lo = lo[i]

            frag = {
                "audio": frag_audio,
                "f0": frag_f0,
                "f0_scaled": frag_f0_scaled,
                "lo": frag_lo,
            }
            fragments.append(frag)

        return fragments


def read_wav(path, audio_sr=None):
    wav_sr, audio = scipy.io.wavfile.read(path)
    assert wav_sr == audio_sr, "Inconsistent sample rate for file {}".format(
        path
    )

    # int to float
    dtype = audio.dtype
    if np.issubdtype(dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(dtype).max

    return audio
