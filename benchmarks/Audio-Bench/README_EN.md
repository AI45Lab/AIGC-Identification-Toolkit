# Audio-Bench: Audio Watermarking Robustness Evaluation Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: AudioMark](https://img.shields.io/badge/Dataset-AudioMark-green.svg)](https://github.com/moyangkuo/AudioMarkBench)

> Evaluate the robustness of audio watermarking algorithms (AudioSeal) under various audio attacks, based on AudioMark dataset (106 audio samples × 45 attack configurations).

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install torch torchaudio librosa julius soundfile audiomentations pydub

# Visualization dependencies
pip install matplotlib numpy scipy pyyaml tqdm
```

### 2. Download Dataset

Download the dataset from Google Drive: [AudioMark Dataset](https://drive.google.com/drive/folders/1037mBf4LoGq0CDxe6hYx5fNNv56AY_9e). After downloading, place the files in `benchmarks/Audio-Bench/dataset/audiomark/`.



### 3. Run Evaluation


```bash
python benchmarks/Audio-Bench/run_benchmark.py
```

**Results Output**: `benchmarks/Audio-Bench/results/audioseal_robustness/`

---

## 📊 Evaluation Pipeline

### Supported Attack Types (8 types × 5 strengths = 40 configurations)

| Attack Type | Strength Parameters | Description |
|---------|---------|------|
| **Gaussian Noise** | [40, 30, 20, 10, 5] | Gaussian noise (SNR in dB, lower = stronger attack) |
| **Background Noise** | [40, 30, 20, 10, 5] | Background noise (SNR in dB) |
| **Quantization** | [4, 8, 16, 32, 64] | Quantization (bit levels, lower = stronger attack) |
| **Lowpass Filter** | [0.1, 0.2, 0.3, 0.4, 0.5] | Low-pass filtering (cutoff frequency ratio) |
| **Highpass Filter** | [0.1, 0.2, 0.3, 0.4, 0.5] | High-pass filtering (cutoff frequency ratio) |
| **Smooth** | [6, 10, 14, 18, 22] | Smoothing (moving average window size) |
| **Echo** | [(0.1,0.1), ..., (0.5,0.5)] | Echo (delay in seconds, volume) |
| **MP3 Compression** | [64, 96, 128, 192, 256] | MP3 compression (bitrate kbps, lower = stronger attack) |

### Evaluation Metrics

#### Quality Metrics (Original Audio vs Watermarked Audio)
- **SNR (Signal-to-Noise Ratio)**: Signal-to-noise ratio in dB, higher is better (typically >40dB indicates high quality)

#### Robustness Metrics (By Attack Type and Strength)
- **TPR (prob)**: True Positive Rate based on detection probability, detection probability > `tau_prob`(0.15) is considered successful detection
- **TPR (BA)**: True Positive Rate based on bit accuracy, bit accuracy > `tau_ba`(0.1) is considered successful detection
- **Bit Accuracy**: Bit accuracy, ratio of correctly extracted watermark bits to total bits (0-1)
- **Average Confidence**: Average confidence when detection succeeds (0-1)


---

## 📈 Visualization Analysis


```bash
python benchmarks/Audio-Bench/utils/plot_radar.py \
  benchmarks/Audio-Bench/results/audioseal_robustness/metrics.json
```

| TPR (Detection Probability) | Avg Confidence | Bit Accuracy |
| --- | --- | --- |
| ![TPR prob](results/audioseal_robustness/audioseal_tpr_prob_radar.png) | ![TPR BA](results/audioseal_robustness/audioseal_avg_confidence_radar.png) | ![Bit Accuracy](results/audioseal_robustness/audioseal_bit_accuracy_radar.png) |

Each chart displays **5 curves** corresponding to 5 attack strength levels (from weak to strong).



---

## 🏆 Acknowledgements

This project is based on the following open source works:

- **[AudioMarkBench](https://github.com/moyangkuo/AudioMarkBench)** - Audio attack implementation and evaluation framework
- **[AudioSeal](https://github.com/facebookresearch/audioseal)** - Meta AI's audio watermarking algorithm
- **[Image-Bench](../Image-Bench/)** - Evaluation pipeline architecture design reference

---
