import torch
import crepe


class CREPEPitch:
    def __init__(self, model_capacity="full", smooth="true"):
        self.model_capacity = model_capacity
        self.smooth = smooth

    def __call__(self, audio, audio_sr, frame_sr):
        step_size = 1 / frame_sr * 1e3  # ms

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
