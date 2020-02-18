from .loudness import RMSLoudness
from .pitch import CREPEPitch


class Preprocessor:
    def __init__(
        self,
        audio_sr=16000,
        frame_sr=100,
        pitch_transform=CREPEPitch(),
        loudness_transform=RMSLoudness(),
    ):
        self.frame_sr = frame_sr
        self.audio_sr = audio_sr

        self.pitch_transform = pitch_transform
        self.loudness_transform = loudness_transform

    def preprocess(self, audio, audio_sr):
        assert audio_sr == self.audio_sr, "Inconsistent audio sample rate"

        f0 = self.pitch_transform(audio, audio_sr, self.frame_sr)
        lo = self.loudness_transform(audio, audio_sr, self.frame_sr)

        return f0, lo
