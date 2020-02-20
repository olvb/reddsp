import torch


class MultiScaleSTFT:
    def __init__(
        self,
        fft_sizes=[64, 128, 256, 512, 1024, 2048],
        overlap=0.75,
        normalized=True,
    ):
        self.fft_sizes = fft_sizes
        self.normalized = normalized
        self.overlap = overlap

    def __call__(self, audio):
        """Computes multiscale stfts to use for  multiscale spectral loss."""
        stft_all = []

        for fft_size in self.fft_sizes:
            hop_length = int(fft_size * (1 - self.overlap))
            stft = torch.stft(
                audio,
                n_fft=fft_size,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                normalized=self.normalized,
                onesided=True,
            )
            stft = torch.sum(stft ** 2, dim=-1)
            # TODO check this
            # stft = stft[:, 1, :]  # remove DC
            stft_all.append(stft)

        return stft_all
