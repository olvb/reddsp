import torch
import torch.nn
import numpy as np

from .decoder_net import DecoderNet
from .harm_synth import HarmSynth
from .noise_synth import NoiseSynth


__all__ = ["Decoder"]


class Decoder(torch.nn.Module):
    def __init__(
        self,
        bypass_harm=False,
        bypass_noise=False,
        nb_harms=64,
        nb_noise_bands=64,
        default_noise_amp=1e-3,
        audio_sr=16000,
        frame_sr=160,
        dtype=torch.float32,
    ):
        super(Decoder, self).__init__()

        assert not bypass_harm or not bypass_noise

        self.dtype = dtype
        self.add_module(
            "decoder_net", DecoderNet(nb_harms, nb_noise_bands).type(dtype)
        )

        self.bypass_harm = bypass_harm
        self.bypass_noise = bypass_noise

        frame_length = audio_sr / frame_sr
        assert frame_length.is_integer()
        frame_length = int(frame_length)

        if not bypass_harm:
            self.add_module(
                "harm_synth",
                HarmSynth(nb_harms, frame_length, audio_sr).type(dtype),
            )
        if not bypass_noise:
            self.add_module(
                "noise_synth",
                NoiseSynth(
                    nb_noise_bands, frame_length, default_amp=default_noise_amp
                ).type(dtype),
            )

    def forward(self, f0, lo):
        f0 = f0.unsqueeze(-1)
        lo = lo.unsqueeze(-1)

        a0, aa, h0, hh = self.decoder_net.forward(f0, lo)

        f0 = f0.squeeze(-1)  # TODO explain what is going on here

        if not self.bypass_harm:
            harm_audio = self.harm_synth.forward(f0, a0, aa)
        else:
            harm_audio = None
        if not self.bypass_noise:
            noise_audio = self.noise_synth.forward(h0, hh)
        else:
            noise_audio = None

        # sum harm+noise
        if noise_audio is None:
            resynth_audio = harm_audio
        elif harm_audio is None:
            resynth_audio = noise_audio
        else:
            resynth_audio = noise_audio + harm_audio

        return resynth_audio, harm_audio, noise_audio

    def infer(self, f0, lo, to_numpy=False):
        """Generates audio with trained network using f0 and lo vectors."""

        # lo = torch.tensor(lo)
        # f0 = torch.tensor(f0)

        # cast to proper dtype
        f0 = f0.type(self.dtype)
        lo = lo.type(self.dtype)

        # add batch dim
        f0 = f0.unsqueeze(0)
        lo = lo.unsqueeze(0)

        with torch.no_grad():
            resynth_audio, harm_audio, noise_audio = self.forward(f0, lo)

        if harm_audio is None:
            harm_audio = torch.zeros_like(resynth_audio)
        if noise_audio is None:
            noise_audio = torch.zeros_like(resynth_audio)

        # remove batch dim
        resynth_audio = resynth_audio.reshape(-1)
        harm_audio = harm_audio.reshape(-1)
        noise_audio = noise_audio.reshape(-1)

        if to_numpy:
            resynth_audio = resynth_audio.numpy().astype(np.float32)
            harm_audio = harm_audio.numpy().astype(np.float32)
            noise_audio = noise_audio.numpy().astype(np.float32)

        return resynth_audio, harm_audio, noise_audio
