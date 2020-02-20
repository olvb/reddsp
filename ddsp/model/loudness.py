import torch
import librosa
import numpy as np

__all__ = ["RMSLoudness", "PerceptualLoudness", "ScalePerceptualLoudness"]


class RMSLoudness:
    def __init__(self):
        pass

    def __call__(self, audio, audio_sr, frame_sr):
        frame_length = audio_sr / frame_sr
        assert frame_length.is_integer()
        frame_length = int(frame_length)

        lo = librosa.feature.rms(
            audio,
            hop_length=frame_length,
            frame_length=frame_length * 2,
            center=True,
        )
        lo = lo.flatten()
        lo = lo.astype(np.float)
        lo = torch.from_numpy(lo)

        return lo


class PerceptualLoudness:
    # upper_lo/lower_lo: values (in dB) used for loudness scaling and centering
    # 21dB: loudness for a 1.0 amplitude white noise
    # cf https://github.com/magenta/ddsp/blob/master/ddsp/spectral_ops.py
    def __init__(self, fft_size=2048, upper_lo=21, lo_range=90):
        self.fft_size = fft_size
        self.lo_range = lo_range
        self.lo_center = upper_lo - self.lo_range / 2

    def __call__(self, audio, audio_sr, frame_sr):
        frame_length = audio_sr / frame_sr
        assert frame_length.is_integer()
        frame_length = int(frame_length)

        stft = librosa.stft(
            audio,
            n_fft=self.fft_size,
            hop_length=frame_length,
            window="rect",  # window must be rect!
            center=True,
            pad_mode="reflect",
        )

        # frequencies weighting
        stft = np.abs(stft) ** 2
        freqs = librosa.fft_frequencies(sr=audio_sr, n_fft=self.fft_size)
        # remove DC
        stft = stft[1:, :]
        freqs = freqs[1:]

        stft_weighted = librosa.core.perceptual_weighting(stft, freqs)

        # average over freqs
        lo = np.mean(stft_weighted, axis=0)

        lo = torch.from_numpy(lo)

        return lo


class ScalePerceptualLoudness:
    # upper_lo/lower_lo: values (in dB) used for loudness scaling and centering
    # 21dB: loudness for a 1.0 amplitude white noise
    # cf https://github.com/magenta/ddsp/blob/master/ddsp/spectral_ops.py
    def __init__(self, upper_lo=21, lo_range=90):
        self.lo_range = lo_range
        self.lo_center = upper_lo - self.lo_range / 2.0

    def __call__(self, lo):
        lo = (lo - self.lo_center) / self.lo_range

        return lo
