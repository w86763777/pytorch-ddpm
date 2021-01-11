# Denoising Diffusion Probabilistic Models

Unofficial PyTorch implementation of Denoising Diffusion Probabilistic Models [1].

This implementation follows the most of details in official TensorFlow
implementation [2]. I use PyTorch coding style to port [2] to PyTorch and hope
that anyone who is familiar with PyTorch can easily understand every
implementation details.

## TODO
- Datasets
    - [x] Support CIFAR10
    - [ ] Support LSUN
    - [ ] Support CelebA-HQ
- Featurex
    - [ ] Gradient accumulation
    - [x] Multi-GPU training
- Reproducing Experiment
    - [ ] CIFAR10

## Requirements
- Python 3.6
- Packages
    Upgrade pip for installing latest tensorboard
    ```
    pip install -U pip setuptools
    pip install -r requirements.txt
    ```
- Download precalculated statistic for dataset:
    - [CIFAR10 train 50k](https://drive.google.com/file/d/1j-oW5Hq-IvmoZZcHWJ-ipCE5XwalZLyJ/view?usp=sharing)
    
    Create folder `stats` and put downloaded `.npz` files in it
    ```
    stats
    └── cifar10_train.npz
    ```

## Train From Scratch
- Take CIFAR10 for example:
    ```
    python main.py --train \
        --flagfile ./config/CIFAR10.txt
    ```
- [Optional] Overwrite arguments
    ```
    python main.py --train \
        --flagfile ./config/CIFAR10.txt \
        --batch_size 64 \
        --logdir ./path/to/logdir
    ```
- [Optional] Select GPU IDs
    ```
    CUDA_VISIBLE_DEVICES=1 python main.py --train \
        --flagfile ./config/CIFAR10.txt
    ```
- [Optional] Multi-GPU training
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train \
        --flagfile ./config/CIFAR10.txt \
        --parallel
    ```

## Evaluate
- A `flagfile.txt` is autosaved to your log directory. The default logdir for `config/CIFAR10.txt` is `./logs/DDPM_CIFAR10_EPS`
- Start evaluation
    ```
    python main.py --eval \
        --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt
    ```
- [Optional] Multi-GPU evaluation
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --eval \
        --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt \
        --parallel
    ```


## Reproducing Experiment

Work in progress.

## Reference

[1] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

[2] [Official TensorFlow implementation](https://github.com/hojonathanho/diffusion)