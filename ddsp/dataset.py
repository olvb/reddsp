import os
import glob

import torch
import scipy.io.wavfile
import numpy as np

from .model.preprocess import Preprocessor

__all__ = ["preprocess_dataset", "Dataset"]


def preprocess_dataset(
    wav_dir_path, pitch_dir_path, loudness_dir_path, audio_sr, frame_sr
):
    preprocessor = Preprocessor(audio_sr=audio_sr, frame_sr=frame_sr)

    wav_pattern = os.path.join(wav_dir_path, "*.wav")
    wav_paths = sorted(glob.glob(wav_pattern))

    for i, wav_path in enumerate(wav_paths):
        audio = read_wav(wav_path, audio_sr)
        f0, lo = preprocessor.preprocess(audio, audio_sr)

        pitch_path = os.path.join(pitch_dir_path, "{:d}.pth".format(i + 1))
        torch.save(f0, pitch_path)

        loudness_path = os.path.join(
            loudness_dir_path, "{:d}.pth".format(i + 1)
        )
        torch.save(lo, loudness_path)

    # store dataset sample rates
    for dir_path in [pitch_dir_path, loudness_dir_path]:
        sr_path = os.path.join(dir_path, "frame_sr.pth")
        torch.save(frame_sr, sr_path)


class Dataset(torch.utils.data.Dataset):
    """Audio dataset class, excepts an audio subfolder containing wave files
    and a f0 subfolder containing f0 csv computed with CREPE."""

    def __init__(
        self,
        wav_dir_path,
        pitch_dir_path,
        loudness_dir_path,
        fragment_duration=2,
        dtype=torch.float,
    ):
        self.fragment_duration = fragment_duration
        self.wav_dir_path = wav_dir_path
        self.pitch_dir_path = pitch_dir_path
        self.loudness_dir_path = loudness_dir_path
        self.dtype = dtype

        self.load_audio_sr()
        self.load_frame_sr()
        self.load_fragments()

    def load_audio_sr(self):
        """ Loads audio sample rate by opening a wav file """
        wav_paths, _, _ = self.get_sample_paths()
        wav_path = wav_paths[0]
        self.audio_sr, _ = scipy.io.wavfile.read(wav_path)

    def load_frame_sr(self):
        """ Loads frame sample rate for preprocessed pitch and loudness values """
        pitch_sr_path = os.path.join(self.pitch_dir_path, "frame_sr.pth")
        pitch_frame_sr = torch.load(pitch_sr_path)

        loudness_sr_path = os.path.join(self.loudness_dir_path, "frame_sr.pth")
        loudness_frame_sr = torch.load(loudness_sr_path)

        assert (
            pitch_frame_sr == loudness_frame_sr
        ), "Pitch and loudness data not sampled at same sample rate"

        self.frame_sr = pitch_frame_sr

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, i):
        return self.fragments[i]

    def load_fragments(self):
        wav_paths, pitch_paths, loudness_paths = self.get_sample_paths()
        all_paths = zip(wav_paths, pitch_paths, loudness_paths)
        self.fragments = []

        for wav_path, pitch_path, loudness_path in all_paths:
            audio = read_wav(wav_path, self.audio_sr)
            audio = torch.from_numpy(audio)

            f0 = torch.load(pitch_path)
            lo = torch.load(loudness_path)

            # convert to requested type (float or double)
            audio = audio.type(self.dtype)
            f0 = f0.type(self.dtype)
            lo = lo.type(self.dtype)

            self.fragments += self.split_into_fragments(audio, f0, lo)

    def split_into_fragments(self, audio, f0, lo):
        nb_samples_per_frag = self.audio_sr * self.fragment_duration
        nb_frames_per_frag = self.frame_sr * self.fragment_duration

        nb_frags = min(
            len(audio) // nb_samples_per_frag,
            len(f0) // nb_frames_per_frag,
            len(lo) // nb_frames_per_frag,
        )

        audio = audio[: nb_frags * nb_samples_per_frag]
        audio = audio.reshape(nb_frags, -1)

        f0 = f0[: nb_frags * nb_frames_per_frag]
        f0 = f0.reshape(nb_frags, -1)

        lo = lo[: nb_frags * nb_frames_per_frag]
        lo = lo.reshape(nb_frags, -1)

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
        wav_paths, pitch_paths, loudness_paths = self.get_sample_paths()

        wav_path = wav_paths[i]
        pitch_path = pitch_paths[i]
        loudness_path = loudness_paths[i]
        audio = read_wav(wav_path, self.audio_sr)
        audio = torch.from_numpy(audio)

        f0 = torch.load(pitch_path)
        lo = torch.load(loudness_path)
        # TODO why aren't they the same size?
        length = min(f0.size()[-1], lo.size()[-1])
        f0 = f0[:length]
        lo = lo[:length]

        audio = audio.type(self.dtype)
        f0 = f0.type(self.dtype)
        lo = lo.type(self.dtype)

        return audio, f0, lo

    def get_sample_paths(self):
        wav_pattern = os.path.join(self.wav_dir_path, "*.wav")
        wav_paths = sorted(glob.glob(wav_pattern))
        assert len(wav_paths) > 0, "No wav file found"

        pitch_file_pattern = os.path.join(self.pitch_dir_path, "[0-9]*.pth")
        pitch_paths = sorted(glob.glob(pitch_file_pattern))
        assert len(pitch_paths) > 0, "No pitch file found"

        loudness_file_pattern = os.path.join(
            self.loudness_dir_path, "[0-9]*.pth"
        )
        loudness_paths = sorted(glob.glob(loudness_file_pattern))
        assert len(loudness_paths) > 0, "No loudness file found"

        return wav_paths, pitch_paths, loudness_paths


def read_wav(path, audio_sr=None):
    wav_sr, audio = scipy.io.wavfile.read(path)
    assert wav_sr == audio_sr, "Inconsistent wav sample rate"

    # int to float
    dtype = audio.dtype
    if np.issubdtype(dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(dtype).max

    return audio
