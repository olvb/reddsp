import torch
import torch.nn.functional as F

__all__ = ["synthesize_noise"]


class NoiseSynth:
    def __init__(
        self,
        nb_bands,
        frame_length,
        default_amp=1e-3,
        dtype=torch.float,
        device=torch.device("cpu"),
    ):
        self.nb_bands = nb_bands
        self.default_amp = default_amp
        self.frame_length = frame_length

        self.device = device
        self.dtype = dtype

    def synthesize(self, h0, hh):
        """Synthesize the noise part of the reconsructed signal by filtering noise
        with filter coefficients output by network"""

        assert len(h0.size()) == 2
        assert len(hh.size()) == 3
        assert h0.size()[0] == hh.size()[0]  # batch dim
        assert h0.size()[1] == hh.size()[2]  # frame dim

        nb_batchs, _, nb_frames = hh.size()

        hh = hh.transpose(1, 2)
        # Init data shared for all synthesis operation
        spectral_zeros = torch.zeros(hh.shape, dtype=self.dtype, device=self.device)
        hh = torch.stack((hh, spectral_zeros), dim=-1)
        ir = torch.irfft(hh, 1, onesided=True)

        # to linear phase
        ir = torch.roll(ir, ir.size()[-1] // 2, -1)

        # apply hann window
        win = torch.hann_window(
            ir.shape[-1], periodic=False, device=self.device, dtype=self.dtype
        )
        win = win.reshape(1, 1, -1)
        ir = ir * win

        # pad filter in time so it has the same length as a frame
        padding = (self.frame_length - ir.size()[-1]) + self.frame_length - 1
        padding_left = padding // 2
        padding_right = padding - padding_left
        ir = F.pad(ir, (padding_left, padding_right))

        # init noise
        noise = (
            torch.rand(
                nb_frames,
                self.frame_length,
                device=self.device,
                dtype=self.dtype,
            )
            * 2
            - 1
        ) * self.default_amp

        noise = torch.conv1d(ir, noise.unsqueeze(1), groups=hh.shape[1])
        noise = noise * h0.unsqueeze(-1)

        noise = noise.reshape(nb_batchs, -1)

        return noise
