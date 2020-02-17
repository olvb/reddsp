import torch
import librosa
import numpy as np


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
            center=False,
        )

        lo = lo.flatten()
        lo = lo.astype(np.float)
        lo = torch.from_numpy(lo)
        return lo
