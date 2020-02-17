## Redo DDSP

ReDDSP is a PyTorch reimplementation of Google's [Differentiable DSP][1]. For now it only features the simpler version of the network plugged to an additive + noise synthesizer (no z-encoder, no reverb). It is still WIP but already operational.

Usage:
1. Import a wav dataset or generate toy datasets.
2. Preprocess chosen dataset. Loudness and f0 data will be added in subfolder.
3. Launch training with paths to dataset, checkpoint folder and TensorBoard run folder.
4. Monitor training and results with TensorBoard.
```shell
./preprocess_dataset.
./gen_toy_datasets.py datasets/py datasets/harm_decay/
./train_model.py datasets/harm_decay/ checkpoints/harm_static_checkpt.pth runs/harm_decay
tensorboard --logdir=runs
```

NB: The code in this repository is based on collective work by fellow [ATIAM][2] students as part of [Philippe Esling][3]'s course on machine learning for music.

[1]: https://openreview.net/pdf?id=B1x1ma4tDr
[2]: http://www.atiam.ircam.fr/en/
[3]: https://esling.github.io/
