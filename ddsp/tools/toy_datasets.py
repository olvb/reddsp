import numpy as np
import torch

from ddsp.model.harmonic import HarmSynth
from ddsp.model.noise import NoiseSynth

__all__ = [
    "gen_harm_dataset",
    "gen_noise_dataset",
    "gen_harm_and_noise_dataset",
    "gen_harm_then_noise_dataset",
    "gen_decay_harm_dataset",
]

FRAME_SR = 100


def gen_harm_dataset(
    nb_files=5, duration=30, nb_harms=10, dyn_profile=True, audio_sr=16000
):
    """Generates a dataset of harmonic sounds at random amplitudes, fading in and out every second.."""

    frame_length = audio_sr // FRAME_SR

    harm_synth = HarmSynth(nb_harms, frame_length, audio_sr)

    audios = []

    for _ in range(nb_files):
        f0, a0, aa = gen_harm_frames(duration, nb_harms, dyn_profile)

        f0 = torch.from_numpy(f0).unsqueeze(0)
        a0 = torch.from_numpy(a0).unsqueeze(0)
        aa = torch.from_numpy(aa).unsqueeze(0)

        audio = harm_synth.synthesize(f0, a0, aa)
        audio = audio[0].numpy().astype(np.float32)

        audios.append(audio)

    return audios


def gen_noise_dataset(
    nb_files=5,
    duration=30,
    nb_noise_bands=20,
    default_amp=1e-1,
    dyn_profile=True,
    audio_sr=16000,
):
    """Generates a dataset of noise at random amplitudes, fading in and out every second."""

    frame_length = audio_sr // FRAME_SR

    noise_synth = NoiseSynth(nb_noise_bands, frame_length, default_amp)

    audios = []

    for _ in range(nb_files):
        h0, hh = gen_noise_frames(duration, nb_noise_bands, dyn_profile)

        h0 = torch.from_numpy(h0).unsqueeze(0)
        hh = torch.from_numpy(hh).unsqueeze(0)

        audio = noise_synth.synthesize(h0, hh)
        audio = audio[0].numpy().astype(np.float32)
        audios.append(audio)

    return audios


def gen_harm_and_noise_dataset(
    nb_files=5,
    duration=30,
    nb_harms=10,
    nb_noise_bands=20,
    default_noise_amp=1e-1,
    audio_sr=16000,
):
    """Generates a dataset of harmonic sounds plus noise at random amplitudes,
    fading in and out every second."""

    frame_length = audio_sr // FRAME_SR

    harm_synth = HarmSynth(nb_harms, frame_length, audio_sr)
    noise_synth = NoiseSynth(nb_noise_bands, frame_length, default_noise_amp)

    audios = []

    for _ in range(nb_files):
        amps = np.random.rand(duration)
        f0, a0, aa = gen_harm_frames(
            duration, nb_harms, dyn_profile=False, amps=amps
        )
        h0, hh = gen_noise_frames(
            duration, nb_noise_bands, dyn_profile=False, amps=amps
        )

        f0 = torch.from_numpy(f0).unsqueeze(0)
        a0 = torch.from_numpy(a0).unsqueeze(0)
        aa = torch.from_numpy(aa).unsqueeze(0)
        h0 = torch.from_numpy(h0).unsqueeze(0)
        hh = torch.from_numpy(hh).unsqueeze(0)

        harm_audio = harm_synth.synthesize(f0, a0, aa)
        noise_audio = noise_synth.synthesize(h0, hh)
        audio = harm_audio + noise_audio
        audio = audio[0].numpy().astype(np.float32)
        audios.append(audio)

    return audios


def gen_harm_then_noise_dataset(
    nb_files=5,
    duration=30,
    nb_harms=10,
    nb_noise_bands=20,
    default_noise_amp=1e-1,
    audio_sr=16000,
):
    """Generates a dataset of harmonic sounds at random amplitudes,
    fading in and out every second, with noise between harmonic parts."""

    frame_length = audio_sr // FRAME_SR

    harm_synth = HarmSynth(nb_harms, frame_length, audio_sr)
    noise_synth = NoiseSynth(nb_noise_bands, frame_length, default_noise_amp)

    audios = []

    for i in range(nb_files):
        amps = np.random.rand(duration)
        f0, a0, aa = gen_harm_frames(
            duration, nb_harms, dyn_profile=False, amps=amps
        )
        h0, hh = gen_noise_frames(
            duration, nb_noise_bands, dyn_profile=False, amps=amps
        )

        f0 = torch.from_numpy(f0).unsqueeze(0)
        a0 = torch.from_numpy(a0).unsqueeze(0)
        aa = torch.from_numpy(aa).unsqueeze(0)
        h0 = torch.from_numpy(h0).unsqueeze(0)
        hh = torch.from_numpy(hh).unsqueeze(0)

        harm_audio = harm_synth.synthesize(f0, a0, aa)
        noise_audio = noise_synth.synthesize(h0, hh)
        noise_audio = torch.roll(noise_audio, audio_sr // 2)

        audio = harm_audio + noise_audio
        audio = audio[0].numpy().astype(np.float32)
        audios.append(audio)

    return audios


def gen_decay_harm_dataset(
    nb_files=5, duration=30, nb_harms=10, decay_amp=False, audio_sr=16000
):
    """Generates a dataset of harmonic sounds at random amplitudes, fading in and out every second, with a  higher frequencies having a faster decay."""

    frame_length = audio_sr // FRAME_SR

    harm_synth = HarmSynth(nb_harms, frame_length, audio_sr)
    harm_synth.max_normalize = True

    audios = []

    for i in range(nb_files):
        f0, a0, aa = gen_decay_harm_values(duration, nb_harms, decay_amp)

        f0 = torch.from_numpy(f0).unsqueeze(0)
        a0 = torch.tensor(a0).unsqueeze(0)
        aa = torch.tensor(aa).unsqueeze(0)

        audio = harm_synth.synthesize(f0, a0, aa)
        audio = audio[0].numpy().astype(np.float32)
        audios.append(audio)

    return audios


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
