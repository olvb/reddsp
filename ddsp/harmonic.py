import torch
import torch.nn.functional as F
import numpy as np

__all__ = ["HarmonicSynth"]


class HarmonicSynth:
    def __init__(
        self,
        nb_harms,
        frame_length,
        audio_sr,
        dtype=torch.float,
        device=torch.device("cpu"),
    ):
        self.nb_harms = nb_harms
        self.frame_length = frame_length
        self.audio_sr = audio_sr
        self.max_normalize = False

        self.device = device
        self.dtype = dtype

    def synthesize(self, f0, a0, aa):
        """Synthesize the harmonic path of the reconsructed signal
        by modulating oscillators with amplitude and spectral profiles output by network."""

        assert len(f0.size()) == 2
        assert len(a0.size()) == 2
        assert len(aa.size()) == 3

        # Batch dim
        assert f0.size()[0] == a0.size()[0]
        assert f0.size()[0] == aa.size()[0]
        # Frame dim
        assert f0.size()[1] == a0.size()[1]
        assert f0.size()[1] == aa.size()[2]

        # Initialisation of lengths
        nb_bounds = f0.size()[1]
        signal_length = nb_bounds * self.frame_length

        # Init harmonic ranks
        nb_harms = aa.size()[1]
        harm_ranks = torch.arange(self.nb_harms, device=self.device) + 1

        # Multiply f0s by harmonic ranks to get all freqs
        ff = f0.unsqueeze(1) * harm_ranks.unsqueeze(0).unsqueeze(-1)
        # Prevent aliasing
        max_f = self.audio_sr / 2.0
        aa[ff >= max_f] = 0.0

        # Normalize harmonic distribution
        if self.max_normalize:
            aa_max, _ = torch.max(aa, dim=1)
            aa = aa / aa_max.unsqueeze(1)
        else:
            aa_sum = torch.sum(aa, dim=1)
            aa = aa / aa_sum.unsqueeze(1)

        # apply global amplitude
        aa = aa * a0.unsqueeze(1)

        # interpolate aa to audio fs
        aa = F.interpolate(
            aa, size=signal_length, mode="linear", align_corners=True
        )

        # Interpolate f0s over time
        f0 = f0.unsqueeze(1)
        f0 = F.interpolate(
            f0, size=signal_length, mode="linear", align_corners=True
        )
        f0 = f0.squeeze(1)

        # Phase accumulation over time
        phases = 2 * np.pi * f0 / self.audio_sr
        phases %= 2 * np.pi
        phases_acc = phases

        # DO NOT DO THIS: phases_acc = torch.cumsum(phases, dim=-1) % (2 * np.pi)
        # but rather % 2pi every 512 samples to avoid precision issues
        cursor = 0
        stride = 512 + 1
        while cursor < signal_length:
            phases_acc[..., cursor] %= 2.0 * np.pi
            phases_acc[..., cursor : cursor + stride] = torch.cumsum(
                phases_acc[..., cursor : cursor + stride], dim=1
            )
            cursor += stride - 1
        phases_acc %= 2.0 * np.pi

        # multiply by harmonic ranks to get phase for all harmonics
        phases_acc = phases_acc.unsqueeze(-1) * harm_ranks
        phases_acc = phases_acc.transpose(1, 2)

        harm_wf = aa * torch.sin(phases_acc)

        # Sum over harmonics
        wf = torch.sum(harm_wf, dim=1)

        return wf
