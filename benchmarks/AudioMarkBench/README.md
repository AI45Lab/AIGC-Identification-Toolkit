# AudioMarkBench

Welcome to **AudioMarkBench**: Benchmarking Robustness of Audio Watermarking

## Overview

AudioMarkBench is designed to provide a comprehensive evaluation of the robustness of various audio watermarking techniques against adversarial attacks. This repository includes datasets and code implementations that allow researchers to benchmark the performance of different watermarking methods under various attack scenarios.

## Features

- **Dataset Access**: Obtain access to our curated audio datasets used for benchmarking.
- **Attack Implementations**: Explore and utilize pre-implemented black-box adversarial attack methods.
- **Robustness Evaluation**: Benchmark the resilience of audio watermarking techniques under different conditions.

## Datasets

To download the audio datasets used in this benchmark, please visit the following link:

[Download Audio Datasets](https://drive.google.com/drive/folders/1037mBf4LoGq0CDxe6hYx5fNNv56AY_9e?usp=sharing)

## Attack Implementations

This repository includes implementations of various black-box adversarial attacks that can be used to test the robustness of audio watermarking methods:

- **HopSkipJumpAttack**: A query-efficient attack based on geometric progression. For more details, check out the official implementation [here](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/hop_skip_jump.py).

- **Square Attack**: A score-based black-box attack that queries a target model with random square-shaped perturbations. For more details, see the implementation [here](https://github.com/max-andr/square-attack/blob/master/attack.py).

## Citation

If you find AudioMarkBench helpful in your research, please consider citing:

```bibtex
@article{liu2024audiomarkbenchbenchmarkingrobustnessaudio,
  title={AudioMarkBench: Benchmarking Robustness of Audio Watermarking},
  author={Hongbin Liu and Moyang Guo and Zhengyuan Jiang and Lun Wang and Neil Zhenqiang Gong},
  journal={arXiv preprint arXiv:2406.06979},
  year={2024}
}
