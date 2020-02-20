from .loudness import *
from .pitch import *


class Preprocessor:
    def __init__(self):
        self.pitch = CREPEPitch()
        self.scale_pitch = ScalePitch()
        self.loudness = PerceptualLoudness()
        self.scale_loudness = ScalePerceptualLoudness()
        # self.loudness = RMSLoudness()

    def __call__(self, audio, audio_sr, frame_sr, f0=None):
        if f0 is None:
            f0 = self.pitch(audio, audio_sr, frame_sr)
        f0_scaled = self.scale_pitch(f0)

        lo = self.loudness(audio, audio_sr, frame_sr)
        # lo = self.scale_loudness(lo)

        return f0, f0_scaled, lo
