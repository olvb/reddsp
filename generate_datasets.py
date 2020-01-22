import csv
import pathlib
import os

import numpy as np
import torch
import scipy.io.wavfile

from ddsp.harmonic import HarmonicSynth
from ddsp.noise import NoiseSynth

AUDIO_SR = 16000
FRAME_SR = 100
FRAME_LENGTH = AUDIO_SR // FRAME_SR
DATASET_FOLDER_PATH = "datasets"

def gen_harm_dataset(
    out_path, nb_files=5, duration=30, nb_harms=10, dyn_profile=True
):
    """Generates a dataset of harmonic sounds at random amplitudes, fading in and out every second.."""

    audio_path = os.path.join(out_path, "audio")
    pathlib.Path(audio_path).mkdir(parents=True, exist_ok=True)
    f0_path = os.path.join(out_path, "f0")
    pathlib.Path(f0_path).mkdir(parents=True, exist_ok=True)

    base_name = os.path.basename(os.path.normpath(out_path))

    harm_synth = HarmonicSynth(
        nb_harms, FRAME_LENGTH, AUDIO_SR, dtype=torch.double
    )

    for i in range(nb_files):
        f0, a0, aa = gen_harm_frames(duration, nb_harms, dyn_profile)

        # write f0 files
        filename = "{}_{}.f0.csv".format(base_name, i)
        path = os.path.join(f0_path, filename)
        write_crepe_like_csv(f0, duration, path)

        f0 = torch.from_numpy(f0).unsqueeze(0)
        a0 = torch.tensor(a0).unsqueeze(0)
        aa = torch.tensor(aa).unsqueeze(0)

        # generate and write audio
        wf = harm_synth.synthesize(f0, a0, aa)
        wf = wf[0].numpy().astype(np.float32)

        filename = "{}_{}.wav".format(base_name, i)
        path = os.path.join(audio_path, filename)
        scipy.io.wavfile.write(path, AUDIO_SR, wf)


def gen_noise_dataset(
    out_path,
    nb_files=5,
    duration=30,
    nb_noise_bands=20,
    default_amp=1e-1,
    dyn_profile=True,
):
    """Generates a dataset of noise at random amplitudes, fading in and out every second."""

    audio_path = os.path.join(out_path, "audio")
    pathlib.Path(audio_path).mkdir(parents=True, exist_ok=True)
    f0_path = os.path.join(out_path, "f0")
    pathlib.Path(f0_path).mkdir(parents=True, exist_ok=True)

    base_name = os.path.basename(os.path.normpath(out_path))

    noise_synth = NoiseSynth(
        nb_noise_bands, FRAME_LENGTH, default_amp, dtype=torch.double
    )

    # iterate over very file
    for i in range(nb_files):
        h0, hh = gen_noise_frames(duration, nb_noise_bands, dyn_profile)

        h0 = torch.from_numpy(h0).unsqueeze(0)
        hh = torch.from_numpy(hh).unsqueeze(0)

        # generate and write audio
        wf = noise_synth.synthesize(h0, hh)
        wf = wf[0].numpy().astype(np.float32)

        filename = "{}_{}.wav".format(base_name, i)
        path = os.path.join(audio_path, filename)
        scipy.io.wavfile.write(path, AUDIO_SR, wf)

        # generate and write fake f0s
        nb_frames = wf.shape[0] // FRAME_SR
        f0 = np.ones(nb_frames) * 200

        filename = "{}_{}.f0.csv".format(base_name, i)
        path = os.path.join(f0_path, filename)
        write_crepe_like_csv(f0, duration, path)


def gen_harm_and_noise_dataset(
    out_path,
    nb_files=5,
    duration=30,
    nb_harms=10,
    nb_noise_bands=20,
    default_noise_amp=1e-1,
):
    """Generates a dataset of harmonic sounds plus noise at random amplitudes,
    fading in and out every second."""

    audio_path = os.path.join(out_path, "audio")
    pathlib.Path(audio_path).mkdir(parents=True, exist_ok=True)
    f0_path = os.path.join(out_path, "f0")
    pathlib.Path(f0_path).mkdir(parents=True, exist_ok=True)

    base_name = os.path.basename(os.path.normpath(out_path))

    harm_synth = HarmonicSynth(
        nb_harms, FRAME_LENGTH, AUDIO_SR, dtype=torch.double
    )
    noise_synth = NoiseSynth(
        nb_noise_bands, FRAME_LENGTH, default_noise_amp, dtype=torch.double
    )

    for i in range(nb_files):
        amps = np.random.rand(duration)
        f0, a0, aa = gen_harm_frames(
            duration, nb_harms, dyn_profile=False, amps=amps
        )
        h0, hh = gen_noise_frames(
            duration, nb_noise_bands, dyn_profile=False, amps=amps
        )

        # write f0 files
        filename = "{}_{}.f0.csv".format(base_name, i)
        path = os.path.join(f0_path, filename)
        write_crepe_like_csv(f0, duration, path)

        f0 = torch.from_numpy(f0).unsqueeze(0)
        a0 = torch.tensor(a0).unsqueeze(0)
        aa = torch.tensor(aa).unsqueeze(0)
        h0 = torch.from_numpy(h0).unsqueeze(0)
        hh = torch.from_numpy(hh).unsqueeze(0)

        # generate and write audio
        harm_wf = harm_synth.synthesize(f0, a0, aa)
        noise_wf = noise_synth.synthesize(h0, hh)
        wf = harm_wf + noise_wf
        wf = wf[0].numpy().astype(np.float32)

        filename = "{}_{}.wav".format(base_name, i)
        path = os.path.join(audio_path, filename)
        scipy.io.wavfile.write(path, AUDIO_SR, wf)


def gen_harm_then_noise_dataset(
    out_path,
    nb_files=5,
    duration=30,
    nb_harms=10,
    nb_noise_bands=20,
    default_noise_amp=1e-1,
):
    """Generates a dataset of harmonic sounds at random amplitudes,
    fading in and out every second, with noise between harmonic parts."""

    audio_path = os.path.join(out_path, "audio")
    pathlib.Path(audio_path).mkdir(parents=True, exist_ok=True)
    f0_path = os.path.join(out_path, "f0")
    pathlib.Path(f0_path).mkdir(parents=True, exist_ok=True)

    base_name = os.path.basename(os.path.normpath(out_path))

    harm_synth = HarmonicSynth(
        nb_harms, FRAME_LENGTH, AUDIO_SR, dtype=torch.double
    )
    noise_synth = NoiseSynth(
        nb_noise_bands, FRAME_LENGTH, default_noise_amp, dtype=torch.double
    )

    for i in range(nb_files):
        amps = np.random.rand(duration)
        f0, a0, aa = gen_harm_frames(
            duration, nb_harms, dyn_profile=False, amps=amps
        )
        h0, hh = gen_noise_frames(
            duration, nb_noise_bands, dyn_profile=False, amps=amps
        )

        # write f0 files
        filename = "{}_{}.f0.csv".format(base_name, i)
        path = os.path.join(f0_path, filename)
        write_crepe_like_csv(f0, duration, path)

        f0 = torch.from_numpy(f0).unsqueeze(0)
        a0 = torch.tensor(a0).unsqueeze(0)
        aa = torch.tensor(aa).unsqueeze(0)
        h0 = torch.from_numpy(h0).unsqueeze(0)
        hh = torch.from_numpy(hh).unsqueeze(0)

        # generate and write audio
        harm_wf = harm_synth.synthesize(f0, a0, aa)
        noise_wf = noise_synth.synthesize(h0, hh)
        noise_wf = torch.roll(noise_wf, AUDIO_SR // 2)

        wf = harm_wf + noise_wf
        wf = wf[0].numpy().astype(np.float32)

        filename = "{}_{}.wav".format(base_name, i)
        path = os.path.join(audio_path, filename)
        scipy.io.wavfile.write(path, AUDIO_SR, wf)


def gen_decay_harm_dataset(
    out_path, nb_files=5, duration=30, nb_harms=10, decay_amp=False
):
    """Generates a dataset of harmonic sounds at random amplitudes, fading in and out every second, with a  higher frequencies having a faster decay."""

    audio_path = os.path.join(out_path, "audio")
    pathlib.Path(audio_path).mkdir(parents=True, exist_ok=True)
    f0_path = os.path.join(out_path, "f0")
    pathlib.Path(f0_path).mkdir(parents=True, exist_ok=True)

    base_name = os.path.basename(os.path.normpath(out_path))

    harm_synth = HarmonicSynth(
        nb_harms, FRAME_LENGTH, AUDIO_SR, dtype=torch.double
    )
    harm_synth.max_normalize = True

    for i in range(nb_files):
        f0, a0, aa = gen_decay_harm_values(duration, nb_harms, decay_amp)

        # write f0 files
        filename = "{}_{}.f0.csv".format(base_name, i)
        path = os.path.join(f0_path, filename)
        write_crepe_like_csv(f0, duration, path)

        f0 = torch.from_numpy(f0).unsqueeze(0)
        a0 = torch.tensor(a0).unsqueeze(0)
        aa = torch.tensor(aa).unsqueeze(0)

        # generate and write audio
        wf = harm_synth.synthesize(f0, a0, aa)
        wf = wf[0].numpy().astype(np.float32)

        filename = "{}_{}.wav".format(base_name, i)
        path = os.path.join(audio_path, filename)
        scipy.io.wavfile.write(path, AUDIO_SR, wf)


def gen_harm_frames(duration, nb_harms, dyn_profile, amps=None):
    # random frequencies
    freqs = 200 * (1 + 3 * np.random.rand(duration))
    # random amplitudes
    if amps is None:
        amps = np.random.rand(duration)
    # fading in and out
    win = np.hanning(FRAME_SR)

    f0_all = np.zeros(0)
    a0_all = np.zeros(0)
    aa_all = np.zeros((0, nb_harms))

    aa_static = np.hamming(nb_harms * 2)[nb_harms:] ** 3
    aa_static[0] = 1

    # iterate over very second
    for j in range(duration):
        f0 = np.ones(FRAME_SR) * freqs[j]
        f0_all = np.concatenate((f0_all, f0))

        a0 = np.ones(FRAME_SR) * amps[j]
        a0 *= win
        a0_all = np.concatenate((a0_all, a0))

        if dyn_profile:
            # harmonic profile linked with frequency
            if freqs[j] > 400:
                aa = np.ones(nb_harms // 3)
                aa = np.concatenate((aa, np.zeros(nb_harms - nb_harms // 3)))
            else:
                aa = np.ones(nb_harms)
            aa[0] = 1
        else:
            aa = aa_static

        aa = np.tile(aa, (FRAME_SR, 1))
        aa_all = np.concatenate((aa_all, aa))

    aa_all = aa_all.T

    return f0_all, a0_all, aa_all


def gen_decay_harm_values(duration, nb_harms, decay_amp):
    # random frequencies
    freqs = 200 * (1 + 3 * np.random.rand(duration))
    # random amplitudes
    amps = np.random.rand(duration)

    amps /= 4  # avoid saturation

    # fading in and out
    win = np.hanning(FRAME_SR // 2)
    win = np.concatenate(
        (win[: FRAME_SR // 4], np.ones(FRAME_SR // 2), win[FRAME_SR // 4 :])
    )
    assert len(win) == FRAME_SR

    # decaying harmonic profil
    time_decay = np.hamming(FRAME_SR * 2)[FRAME_SR:]
    aa_decay = np.zeros((FRAME_SR, nb_harms))
    for i in range(nb_harms):
        aa_decay[:, i] = np.hamming(FRAME_SR * 2)[FRAME_SR:] ** (i * 2)
    aa_decay /= np.sum(aa_decay[0, :])

    f0_all = np.zeros(0)
    a0_all = np.zeros(0)
    aa_all = np.zeros((0, nb_harms))

    # iterate over very second
    for j in range(duration):
        f0 = np.ones(FRAME_SR) * freqs[j]
        f0_all = np.concatenate((f0_all, f0))

        if decay_amp:
            a0 = np.hanning(FRAME_SR * 2)[FRAME_SR:] * amps[j]
        else:
            a0 = np.ones(FRAME_SR) * amps[j]

        a0 *= win
        a0_all = np.concatenate((a0_all, a0))
        aa_all = np.concatenate((aa_all, aa_decay))

    aa_all = aa_all.T

    # print(aa_all.shape)
    # import matplotlib.pyplot as plt
    # plt.plot(aa_all[0,:])
    # plt.show()
    # exit()

    return f0_all, a0_all, aa_all


def gen_noise_frames(duration, nb_noise_bands, dyn_profile, amps=None):
    # random amplitudes
    if amps is None:
        amps = np.random.rand(duration)
    # spectrum correlated with amplitude
    end_hh_all = amps.copy()
    # fading in and out
    win = np.hanning(FRAME_SR)

    h0_all = np.zeros(0)
    hh_all = np.zeros((0, nb_noise_bands))

    hh_static = np.hamming(nb_noise_bands) ** 3

    # iterate over very second
    for j in range(duration):
        h0 = np.ones(FRAME_SR) * amps[j]
        h0 *= win
        h0_all = np.concatenate((h0_all, h0))

        if dyn_profile:
            # frequency profile linked with amplitude
            if amps[j] < 0.5:
                # hh = np.hamming(nb_noise_bands * 2)[nb_noise_bands:] ** 3
                hh = np.ones(nb_noise_bands // 3)
                hh = np.concatenate(
                    (hh, np.zeros(nb_noise_bands - nb_noise_bands // 3))
                )
            else:
                hh = np.ones(nb_noise_bands)
        else:
            hh = hh_static

        hh = np.tile(hh, (FRAME_SR, 1))
        hh_all = np.concatenate((hh_all, hh))

    hh_all = hh_all.T

    return h0_all, hh_all


def write_crepe_like_csv(f0s, duration, out_path):
    """Write a f0 csv file as if generated by CREPE"""

    times = np.linspace(0, duration, len(f0s))
    confidences = np.ones_like(f0s)
    rows = np.vstack((times, f0s, confidences)).T

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_NONE)
        writer.writerow(["time", "frequency", "confidence"])
        writer.writerows(rows)


if __name__ == "__main__":
    gen_harm_dataset(nb_harms=1, out_path=os.path.join("datasets", "pure"))

    gen_harm_dataset(
        nb_harms=10, dyn_profile=True, out_path=os.path.join("datasets", "harm_dyn")
    )

    gen_harm_dataset(
        nb_harms=10, dyn_profile=False, out_path=os.path.join("datasets", "harm_static")
    )

    gen_noise_dataset(
        nb_noise_bands=20,
        dyn_profile=True,
        default_amp=1e-1,
        out_path=os.path.join("datasets", "noise_dyn")
    )

    gen_noise_dataset(
        nb_noise_bands=20,
        dyn_profile=False,
        default_amp=1e-1,
        out_path=os.path.join("datasets", "noise_static")
    )

    gen_harm_and_noise_dataset(
        nb_harms=10,
        nb_noise_bands=20,
        out_path=os.path.join("datasets", "harm_and_noise")
    )

    gen_harm_then_noise_dataset(
        nb_harms=10,
        nb_noise_bands=20,
        out_path=os.path.join("datasets", "harm_or_noise")
    )

    gen_decay_harm_dataset(
        nb_harms=10, decay_amp=True, out_path=os.path.join("datasets", "harm_decay")
    )

