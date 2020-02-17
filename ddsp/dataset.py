import os
import glob

import torch
import scipy.io.wavfile
import numpy as np

from .model.preprocess import Preprocessor

__all__ = ["preprocess_dataset", "Dataset"]


def preprocess_dataset(wav_dir_path, preprocess_dir_path, audio_sr, frame_sr):
    preprocessor = Preprocessor(audio_sr=audio_sr, frame_sr=frame_sr)

    wav_pattern = os.path.join(wav_dir_path, "*.wav")
    wav_paths = sorted(glob.glob(wav_pattern))

    for i, wav_path in enumerate(wav_paths):
        audio = read_wav(wav_path, audio_sr)
        f0, lo = preprocessor.preprocess(audio, audio_sr)

        preprocess_path = os.path.join(
            preprocess_dir_path, "{:d}.pth".format(i + 1)
        )
        torch.save({"f0": f0, "lo": lo}, preprocess_path)

    # store dataset sample rates
    sr_path = os.path.join(preprocess_dir_path, "sr.pth")
    torch.save({"audio_sr": audio_sr, "frame_sr": frame_sr}, sr_path)


class Dataset(torch.utils.data.Dataset):
    """Audio dataset class, excepts an audio subfolder containing wave files
    and a f0 subfolder containing f0 csv computed with CREPE."""

    def __init__(
        self,
        wav_dir_path,
        preprocess_dir_path,
        fragment_duration=2,
        dtype=torch.float,
    ):
        self.fragment_duration = fragment_duration
        self.wav_dir_path = wav_dir_path
        self.preprocess_dir_path = preprocess_dir_path
        self.dtype = dtype

        self.load_sr()
        self.load_fragments()

    def load_sr(self):
        sr_path = os.path.join(self.preprocess_dir_path, "sr.pth")
        self.audio_sr, self.frame_sr = torch.load(sr_path).values()

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, i):
        return self.fragments[i]

    def load_fragments(self):
        wav_paths, preprocess_paths = self.get_sample_paths()
        self.fragments = []

        for wav_path, preprocess_path in zip(wav_paths, preprocess_paths):
            audio = read_wav(wav_path, self.audio_sr)
            audio = torch.from_numpy(audio)

            lo, f0 = torch.load(preprocess_path).values()

            audio = audio.type(self.dtype)
            lo = lo.type(self.dtype)
            f0 = f0.type(self.dtype)

            self.fragments += self.split_into_fragments(audio, lo, f0)

    def split_into_fragments(self, audio, f0, lo):
        nb_samples_per_frag = self.audio_sr * self.fragment_duration
        nb_frames_per_frag = self.frame_sr * self.fragment_duration

        nb_frags = min(
            len(audio) // nb_samples_per_frag,
            len(lo) // nb_frames_per_frag,
            len(f0) // nb_frames_per_frag,
        )

        audio = audio[: nb_frags * nb_samples_per_frag]
        audio = audio.reshape(nb_frags, -1)

        lo = lo[: nb_frags * nb_frames_per_frag]
        lo = lo.reshape(nb_frags, -1)

        f0 = f0[: nb_frags * nb_frames_per_frag]
        f0 = f0.reshape(nb_frags, -1)

        fragments = []
        for i in range(nb_frags):
            frag_audio = audio[i]
            frag_f0 = f0[i]
            frag_lo = lo[i]

            frag = {"audio": frag_audio, "f0": frag_f0, "lo": frag_lo}
            fragments.append(frag)

        return fragments

    def get_sample(self, i):
        """ Return unfragmented audio sample. Useful for evaluation """
        wav_paths, preprocess_paths = self.get_sample_paths()

        wav_path = wav_paths[i]
        preprocess_path = preprocess_paths[i]
        audio = read_wav(wav_path, self.audio_sr)
        audio = torch.from_numpy(audio)

        lo, f0 = torch.load(preprocess_path).values()
        # TODO why aren't they the same size?
        length = min(lo.size()[-1], f0.size()[-1])
        lo = lo[:length]
        f0 = f0[:length]

        audio = audio.type(self.dtype)
        lo = lo.type(self.dtype)
        f0 = f0.type(self.dtype)

        return audio, f0, lo

    def get_sample_paths(self):
        wav_pattern = os.path.join(self.wav_dir_path, "*.wav")
        wav_paths = sorted(glob.glob(wav_pattern))
        assert len(wav_paths) > 0

        preprocess_file_pattern = os.path.join(
            self.preprocess_dir_path, "*.pth"
        )
        preprocess_paths = sorted(glob.glob(preprocess_file_pattern))
        assert len(preprocess_paths) > 0

        return wav_paths, preprocess_paths


def read_wav(path, audio_sr):
    wav_sr, audio = scipy.io.wavfile.read(path)
    assert wav_sr == audio_sr, "Inconsistent wav sample rate"

    # int to float
    dtype = audio.dtype
    if np.issubdtype(dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(dtype).max

    return audio
