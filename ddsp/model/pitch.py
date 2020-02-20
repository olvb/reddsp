import torch
import crepe

__all__ = ["CREPEPitch", "ScalePitch"]


class CREPEPitch:
    def __init__(self, model_capacity="full", smooth="true"):
        self.model_capacity = model_capacity
        self.smooth = smooth

    def __call__(self, audio, audio_sr, frame_sr):
        step_size = 1 / frame_sr * 1e3  # ms
        assert step_size.is_integer()
        step_size = int(step_size)

        _, f0, _, _ = crepe.predict(
            audio,
            audio_sr,
            step_size=step_size,
            center=True,
            model_capacity=self.model_capacity,
            viterbi=self.smooth,
        )

        f0 = torch.from_numpy(f0)

        return f0


class ScalePitch:
    def __init__(self, f0_lower=10, f0_upper=20000):
        self.f0_scale = f0_upper - f0_lower
        self.f0_center = self.f0_scale / 2.0

    def __call__(self, f0):
        f0 = torch.log(f0)
        f0 = (f0 - self.f0_center) / self.f0_scale

        return f0
