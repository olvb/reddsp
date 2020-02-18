## Redo DDSP

ReDDSP is a PyTorch reimplementation of Google's [Differentiable DSP][1]. For now it only features the simpler version of the network plugged to an additive + noise synthesizer (no z-encoder, no reverb). It is still WIP but already operational.

Usage:
1. Import a wav dataset or generate toy datasets
2. Preprocess chosen dataset. Pitch and loudness data will be added in subfolders
3. Launch training with paths to dataset and to TensorBoard run folder. Checkpoints will also be saved in the run folder
4. Monitor training and results with TensorBoard
```shell
./gen_toy_datasets.py datasets/
./preprocess_dataset.py datasets/harm_decay/
./train_model.py datasets/harm_decay/ runs/harm_decay
tensorboard --logdir=runs
```

NB: The code in this repository is based on collective work by fellow [ATIAM][2] students as part of [Philippe Esling][3]'s course on machine learning for music.

[1]: https://openreview.net/pdf?id=B1x1ma4tDr
[2]: http://www.atiam.ircam.fr/en/
[3]: https://esling.github.io/
