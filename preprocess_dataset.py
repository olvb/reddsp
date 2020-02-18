#!/usr/bin/env python3

import os
import argparse
import pathlib

from ddsp.dataset import preprocess_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset_dir_path",
    metavar="dataset-dir",
    help="Path to the dataset folder containing audio folder with wav files",
)
parser.add_argument(
    "--audio-sr", type=int, default=16000, help="Audio sample rate"
)
parser.add_argument(
    "--frame-sr", type=int, default=100, help="Frame sample rate"
)

args = parser.parse_args()
dataset_dir_path = args.dataset_dir_path
audio_sr = args.audio_sr
frame_sr = args.frame_sr

wav_dir_path = os.path.join(dataset_dir_path, "audio")
pitch_dir_path = os.path.join(dataset_dir_path, "pitch")
pathlib.Path(pitch_dir_path).mkdir(parents=True, exist_ok=True)
loudness_dir_path = os.path.join(dataset_dir_path, "loudness")
pathlib.Path(loudness_dir_path).mkdir(parents=True, exist_ok=True)

preprocess_dataset(
    wav_dir_path,
    pitch_dir_path,
    loudness_dir_path,
    audio_sr=audio_sr,
    frame_sr=frame_sr,
)
