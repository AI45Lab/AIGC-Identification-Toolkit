# Video-Bench: Video Watermarking Robustness Evaluation Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: VideoMarkBench](https://img.shields.io/badge/Dataset-VideoMarkBench-green.svg)](https://www.kaggle.com/datasets/zhengyuanjiang/videomarkbench/data)

> Evaluate the robustness of video watermarking algorithms (VideoSeal) under various video attacks, based on VideoMarkBench dataset,.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio

# Evaluation and visualization dependencies
pip install pytorch-msssim lpips scipy pyyaml tqdm matplotlib numpy
```

### 2. Download Dataset

Download VideoMarkBench dataset from Kaggle: [VideoMarkBench Dataset](https://www.kaggle.com/datasets/zhengyuanjiang/videomarkbench/data), then extract to `benchmarks/Video-Bench/dataset/VideoMarkBench/`



### 3. Run Evaluation

```bash
python benchmarks/Video-Bench/run_benchmark.py
```

**Results Output**: `benchmarks/Video-Bench/results/videoseal_robustness/`

---

## 📊 Evaluation Pipeline

### Supported Attack Types (7 attacks × multiple strength levels)

#### Frame-Level Perturbations

Applied frame-by-frame to test spatial domain robustness:

| Attack Type | Strength Parameters | Description |
|---------|---------|------|
| **Gaussian Noise** | [0.01, 0.05, 0.10, 0.15, 0.20] | Gaussian noise (standard deviation σ, larger = stronger attack) |
| **Gaussian Blur** | [0.1, 0.5, 1.0, 1.5] | Gaussian blur (kernel standard deviation σ, larger = stronger) |
| **JPEG Compression** | [90, 80, 60, 40, 20] | JPEG quality factor (smaller = stronger attack) |
| **Crop** | [0.98, 0.96, 0.94, 0.92, 0.90] | Crop then resize (retention ratio, smaller = stronger) |

#### Video-Level Perturbations

Leveraging temporal characteristics to test temporal domain robustness:

| Attack Type | Strength Parameters | Description |
|---------|---------|------|
| **Frame Average** | [1, 2, 3, 4, 5] | Frame averaging (window size N, 1=no change, larger = stronger) |
| **Frame Swap** | [0.00, 0.05, 0.10, 0.15, 0.20] | Random swap adjacent frames (probability p, larger = stronger) |
| **Frame Remove** | [0.00, 0.05, 0.10, 0.15, 0.20] | Random frame removal (probability p, larger = stronger) |



### Evaluation Metrics

#### Quality Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**: Peak signal-to-noise ratio in dB, higher is better (>40dB is high quality)
- **SSIM (Structural Similarity Index)**: Structural similarity, 0-1 range, closer to 1 is better (>0.98 is high quality)
- **tLP (temporal LPIPS)**: Temporal consistency, perceptual difference between adjacent frames, lower is better (<0.01 is excellent)

#### Robustness Metrics
- **FNR (False Negative Rate)**: False negative rate, proportion of videos where watermark was not detected (0-1, lower is better)
- **Bit Accuracy**: Bit accuracy, ratio of correctly extracted watermark bits to total bits (0-1, >0.9 is excellent)
- **Average Confidence**: Average detection confidence, detection signal strength (0-1, higher is better)

---

## 📈 Visualization Analysis



```bash
python benchmarks/Video-Bench/utils/plot_radar.py \
  benchmarks/Video-Bench/results/videoseal_robustness/metrics.json
```

| FNR | Bit Accuracy | Avg Confidence |
| --- | --- | --- |
| ![FNR](results/videoseal_robustness/videoseal_fnr_radar.png) | ![Bit Accuracy](results/videoseal_robustness/videoseal_bit_accuracy_radar.png) | ![Avg Confidence](results/videoseal_robustness/videoseal_avg_confidence_radar.png) |

Each chart displays **5 curves** corresponding to 5 attack strength levels (from weak to strong).

---


## 🏆 Acknowledgements

This project is based on the following open source works:

- **[VideoMarkBench](https://www.kaggle.com/datasets/zhengyuanjiang/videomarkbench/data)** - Video attack implementation and evaluation framework
- **[VideoSeal](https://github.com/facebookresearch/videosse)** - Meta Research's video watermarking algorithm


---

