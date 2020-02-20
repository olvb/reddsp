import torch
import torch.nn
import torch.nn.functional as F

__all__ = ["NoiseSynth"]


class NoiseSynth(torch.nn.Module):
    def __init__(self, nb_bands, frame_length, default_amp=1e-3):
        super(NoiseSynth, self).__init__()

        self.nb_bands = nb_bands
        self.default_amp = default_amp
        self.frame_length = frame_length

    def forward(self, h0, hh):
        """Synthesize the noise part of the reconsructed signal by filtering noise
        with filter coefficients output by network"""

        # assert len(h0.size()) == 2
        # assert len(hh.size()) == 3
        # assert h0.size()[0] == hh.size()[0]  # batch dim
        # assert h0.size()[1] == hh.size()[2]  # frame dim

        nb_batchs, _, nb_frames = hh.size()

        hh = hh.transpose(1, 2)

        # spectral to time
        spectral_zeros = torch.zeros(
            hh.shape, dtype=hh.dtype, device=hh.device
        )
        hh = torch.stack((hh, spectral_zeros), dim=-1)
        ir = torch.irfft(hh, 1, onesided=True)

        # to linear phase
        ir = torch.roll(ir, ir.size()[-1] // 2, -1)

        # apply hann window
        win = torch.hann_window(
            ir.shape[-1], periodic=False, device=ir.device, dtype=ir.dtype
        )
        win = win.reshape(1, 1, -1)
        ir = ir * win

        # pad filter in time so it has the same length as a frame
        padding = self.frame_length - ir.size()[-1]
        padding_left = padding // 2
        padding_right = padding - padding_left
        ir = F.pad(ir, (padding_left, padding_right))

        # init noise
        noise = torch.rand(
            nb_frames, self.frame_length, device=ir.device, dtype=ir.dtype
        )
        noise = (noise * 2 - 1) * self.default_amp

        # filter noise
        noise = conv1d_fft(noise.unsqueeze(0), ir)
        noise = noise * h0.unsqueeze(-1)

        noise = noise.reshape(nb_batchs, -1)

        return noise

    def synthesize(self, h0, hh):
        return self.forward(h0, hh)


# https://github.com/pyro-ppl/pyro/blob/dev/pyro/ops/tensor_utils.py
def conv1d_fft(signal, kernel):
    assert signal.size(-1) == kernel.size(-1)
    f_signal = torch.rfft(
        torch.cat([signal, torch.zeros_like(signal)], dim=-1), 1
    )
    f_kernel = torch.rfft(
        torch.cat([kernel, torch.zeros_like(kernel)], dim=-1), 1
    )
    f_result = complex_mul(f_signal, f_kernel)
    result = torch.irfft(f_result, 1)[..., : signal.size(-1)]
    return result


# https://github.com/pyro-ppl/pyro/blob/dev/pyro/ops/tensor_utils.py
def complex_mul(a, b):
    a_real, a_imag = a.unbind(-1)
    b_real, b_imag = b.unbind(-1)
    result = torch.stack(
        [a_real * b_real - a_imag * b_imag, a_real * b_imag + a_imag * b_real],
        dim=-1,
    )
    return result
