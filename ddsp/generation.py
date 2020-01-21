import torch
import numpy as np
import scipy.io.wavfile

from .harmonic import HarmonicSynth
from .noise import NoiseSynth
from .network import DDSPNetwork

__all__ = ["generate"]


def generate(
    net,
    lo,
    f0,
    harm_synth=None,
    noise_synth=None,
    frame_length=None,
    audio_sr=None,
):
    """Generates audio with trained network using lo and f0 vectors."""

    if harm_synth is None:
        assert frame_length is not None
        assert audio_sr is not None
        harm_synth = HarmonicSynth(
            net.nb_harms,
            dtype=net.dtype,
            device=net.device,
            frame_length=frame_length,
            audio_sr=audio_sr,
        )

    if noise_synth is None:
        assert frame_length is not None
        assert audio_sr is not None
        noise_synth = NoiseSynth(
            net.nb_noise_bands,
            dtype=net.dtype,
            device=net.device,
            frame_length=frame_length,
        )

    lo = torch.tensor(lo).unsqueeze(0).unsqueeze(-1).type(net.dtype)
    f0 = torch.tensor(f0).unsqueeze(0).unsqueeze(-1).type(net.dtype)

    x = {"f0": f0, "lo": lo}

    with torch.no_grad():
        a0, aa, h0, hh = net.forward(f0, lo)

    f0 = f0.squeeze(-1)

    harm_wf = harm_synth.synthetize(f0, a0, aa)
    noise_wf = noise_synth.synthetize(h0, hh)
    synth_wf = harm_wf + noise_wf

    harm_wf = harm_wf.numpy().reshape(-1).astype(np.float32)
    noise_wf = noise_wf.numpy().reshape(-1).astype(np.float32)
    synth_wf = synth_wf.numpy().reshape(-1).astype(np.float32)

    return harm_wf, noise_wf, synth_wf

