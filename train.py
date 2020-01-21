import sys
import pathlib
import os
import csv
import time

import torch
import scipy.io.wavfile

from ddsp.network import DDSPNetwork
from ddsp.dataset import Dataset
from ddsp.generation import generate
from ddsp.training import Training, save_checkpoint, restore_checkpoint

NB_EPOCHS = 200
NB_HARMS = 64
NB_NOISE_BANDS = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

AUDIO_SR = 16000
FRAME_SR = 100

TEST_AUDIO_DURATION = 10
CHECKPOINT_PERIOD = 10

# torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    dataset_path = sys.argv[1]
    checkpoint_dir_path = sys.argv[2]

    print("Using dataset:", dataset_path)
    print("Using device:", str(DEVICE))
    print("Using type:", str(DTYPE))

    # load dataset
    dataset = Dataset(dataset_path, audio_sr=AUDIO_SR, frame_sr=FRAME_SR)

    # retrieve checkpoint or init new network
    checkpoint_path = os.path.join(checkpoint_dir_path, "checkpoint.pth")

    if os.path.exists(checkpoint_path):
        print("Restoring checkpoint:", checkpoint_path)
        training = restore_checkpoint(
            checkpoint_path,
            dataset,
            NB_HARMS,
            NB_NOISE_BANDS,
            dtype=DTYPE,
            device=DEVICE,
        )
    else:
        pathlib.Path(checkpoint_dir_path).mkdir(parents=True, exist_ok=True)

        net = DDSPNetwork(NB_HARMS, NB_NOISE_BANDS, dtype=DTYPE, device=DEVICE)
        training = Training(net, dataset)

    test_lo, test_f0, test_f0_log = dataset.get_test_lo_f0(TEST_AUDIO_DURATION)

    # generate and save reference audio
    truth_wf = dataset.get_test_waveform(TEST_AUDIO_DURATION)
    truth_path = os.path.join(checkpoint_dir_path, "truth.wav")
    scipy.io.wavfile.write(truth_path, AUDIO_SR, truth_wf)

    loss_curve = []
    loss_path = os.path.join(checkpoint_dir_path, "loss.csv")
    best_loss = 100000

    while training.cur_epoch < NB_EPOCHS:
        print("Epoch {}/{}".format(training.cur_epoch + 1, NB_EPOCHS))

        start_time = time.time()
        epoch_loss = training.run_epoch()
        end_time = time.time()
        epoch_time = end_time - start_time

        print(
            "Epoch Loss: {:.3f} (computed in {:.2f} secs)".format(
                epoch_loss, epoch_time
            )
        )

        loss_curve.append(epoch_loss)
        with open(loss_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_NONE)
            writer.writerow(loss_curve)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print("Saving checkpoint")
            save_checkpoint(training, out_path=checkpoint_path)

            harm_wf, noise_wf, synth_wf = generate(
                training.net,
                test_lo,
                test_f0,
                harm_synth=training.harm_synth,
                noise_synth=training.noise_synth,
            )

            harm_path = os.path.join(checkpoint_dir_path, "harm.wav")
            scipy.io.wavfile.write(harm_path, AUDIO_SR, harm_wf)
            noise_path = os.path.join(checkpoint_dir_path, "noise.wav")
            scipy.io.wavfile.write(noise_path, AUDIO_SR, noise_wf)
            synth_path = os.path.join(checkpoint_dir_path, "synth.wav")
            scipy.io.wavfile.write(synth_path, AUDIO_SR, synth_wf)

