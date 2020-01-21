import sys
import os

import torch
import scipy.io.wavfile

from ddsp.dataset import Dataset
from ddsp.generation import generate
from ddsp.training import restore_checkpoint

NB_EPOCHS = 100
NB_HARMS = 64
NB_NOISE_BANDS = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

AUDIO_SR = 16000
FRAME_SR = 100
FRAME_LENGTH = AUDIO_SR // FRAME_SR

DURATION = 10
OUT_PATH = "."


if __name__ == "__main__":
    dataset_path = sys.argv[1]
    checkpoint_dir_path = sys.argv[2]

    # restore checkpoint
    checkpoint_path = os.path.join(checkpoint_dir_path, "checkpoint.pth")

    print("Using dataset at:", dataset_path)
    print("Using checkpoint at:", checkpoint_path)
    # load dataset
    dataset = Dataset(dataset_path, audio_sr=AUDIO_SR, frame_sr=FRAME_SR)
    training = restore_checkpoint(
        checkpoint_path,
        dataset,
        NB_HARMS,
        NB_NOISE_BANDS,
        dtype=DTYPE,
        device=DEVICE,
    )
    net = training.net

    test_lo, test_f0, test_f0log = dataset.get_test_lo_f0(DURATION)
    harm_wf, noise_wf, synth_wf = generate(
        net, test_lo, test_f0, frame_length=FRAME_LENGTH, audio_sr=AUDIO_SR
    )

    harm_path = os.path.join(OUT_PATH, "harm.wav")
    scipy.io.wavfile.write(harm_path, AUDIO_SR, harm_wf)
    noise_path = os.path.join(OUT_PATH, "noise.wav")
    scipy.io.wavfile.write(noise_path, AUDIO_SR, noise_wf)
    synth_path = os.path.join(OUT_PATH, "synth.wav")
    scipy.io.wavfile.write(synth_path, AUDIO_SR, synth_wf)

    truth_wf = dataset.get_test_waveform(DURATION)
    truth_path = os.path.join(OUT_PATH, "truth.wav")
    scipy.io.wavfile.write(truth_path, AUDIO_SR, truth_wf)

