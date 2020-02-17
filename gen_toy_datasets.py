#!/usr/bin/env python3

import os
import argparse
import pathlib

import scipy.io.wavfile

from ddsp.tools.toy_datasets import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "out_dir_path",
    metavar="out-dir",
    help="Path to the folder where to write datasets wav files",
)
parser.add_argument(
    "--audio-sr", type=int, default=16000, help="Audio sample rate"
)

args = parser.parse_args()
out_dir_path = args.out_dir_path
audio_sr = args.audio_sr


def write_wavs(audios, out_dir_path, dataset_name):
    audio_dir_path = os.path.join(out_dir_path, dataset_name, "audio")
    pathlib.Path(audio_dir_path).mkdir(parents=True, exist_ok=True)
    for i, audio in enumerate(audios):
        filename = "{}_{:d}.wav".format(dataset_name, i)
        audio_path = os.path.join(audio_dir_path, filename)
        scipy.io.wavfile.write(audio_path, audio_sr, audio)


audios = gen_harm_dataset(nb_harms=1, audio_sr=audio_sr)
write_wavs(audios, out_dir_path, "pure")

audios = gen_harm_dataset(nb_harms=10, dyn_profile=True, audio_sr=audio_sr)
write_wavs(audios, out_dir_path, "harm_dyn")

audios = gen_harm_dataset(nb_harms=10, dyn_profile=False, audio_sr=audio_sr)
write_wavs(audios, out_dir_path, "harm_static")

audios = gen_noise_dataset(
    nb_noise_bands=20, dyn_profile=True, default_amp=1e-1, audio_sr=audio_sr
)
write_wavs(audios, out_dir_path, "noise_dyn")

audios = gen_noise_dataset(
    nb_noise_bands=20, dyn_profile=False, default_amp=1e-1, audio_sr=audio_sr
)
write_wavs(audios, out_dir_path, "noise_static")

audios = gen_harm_and_noise_dataset(
    nb_harms=10, nb_noise_bands=20, audio_sr=audio_sr
)
write_wavs(audios, out_dir_path, "harm_and_noise")

audios = gen_harm_then_noise_dataset(
    nb_harms=10, nb_noise_bands=20, audio_sr=audio_sr
)
write_wavs(audios, out_dir_path, "harm_then_noise")

audios = gen_decay_harm_dataset(nb_harms=10, decay_amp=True, audio_sr=audio_sr)
write_wavs(audios, out_dir_path, "harm_decay")
