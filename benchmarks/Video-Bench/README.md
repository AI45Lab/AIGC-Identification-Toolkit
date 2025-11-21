# Video-Bench: 视频水印鲁棒性评估基准

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: VideoMarkBench](https://img.shields.io/badge/Dataset-VideoMarkBench-green.svg)](https://www.kaggle.com/datasets/zhengyuanjiang/videomarkbench/data)

> 评估视频水印算法（VideoSeal）在多种视频攻击下的鲁棒性，基于 VideoMarkBench 数据集。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 核心依赖
pip install torch torchvision torchaudio

# 评估和可视化依赖
pip install pytorch-msssim lpips scipy pyyaml tqdm matplotlib numpy
```

### 2. 下载数据集

从 Kaggle 下载 VideoMarkBench 数据集：[VideoMarkBench Dataset](https://www.kaggle.com/datasets/zhengyuanjiang/videomarkbench/data)，下载后解压到 `benchmarks/Video-Bench/dataset/VideoMarkBench/`



### 3. 运行评估

```bash
python benchmarks/Video-Bench/run_benchmark.py
```

**结果输出**：`benchmarks/Video-Bench/results/videoseal_robustness/`

---

## 📊 评估流程

### 支持的攻击类型（7种攻击 × 多个强度级别）

#### 图像级扰动（Frame-Level Perturbations）

逐帧应用，测试空间域鲁棒性：

| 攻击类型 | 强度参数 | 说明 |
|---------|---------|------|
| **Gaussian Noise** | [0.01, 0.05, 0.10, 0.15, 0.20] | 高斯噪声（标准差 σ，越大攻击越强） |
| **Gaussian Blur** | [0.1, 0.5, 1.0, 1.5] | 高斯模糊（核标准差 σ，越大越强） |
| **JPEG Compression** | [90, 80, 60, 40, 20] | JPEG质量因子（越小攻击越强） |
| **Crop** | [0.98, 0.96, 0.94, 0.92, 0.90] | 裁剪后resize（保留比例，越小越强） |

#### 视频级扰动（Video-Level Perturbations）

利用时间特性，测试时间域鲁棒性：

| 攻击类型 | 强度参数 | 说明 |
|---------|---------|------|
| **Frame Average** | [1, 2, 3, 4, 5] | 帧平均（窗口大小 N，1=无变化，越大越强） |
| **Frame Swap** | [0.00, 0.05, 0.10, 0.15, 0.20] | 随机交换相邻帧（概率 p，越大越强） |
| **Frame Remove** | [0.00, 0.05, 0.10, 0.15, 0.20] | 随机删除帧（概率 p，越大越强） |


### 评估指标

#### 质量指标
- **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比，单位 dB，越高越好（>40dB为高质量）
- **SSIM (Structural Similarity Index)**: 结构相似性，0-1范围，越接近1越好（>0.98为高质量）
- **tLP (temporal LPIPS)**: 时间一致性，相邻帧感知差异，越低越好（<0.01为优秀）

#### 鲁棒性指标
- **FNR (False Negative Rate)**: 漏报率，未检测到水印的视频比例（0-1，越低越好）
- **Bit Accuracy**: 比特准确率，正确提取的水印比特占总比特数的比例（0-1，>0.9为优秀）
- **Average Confidence**: 平均检测置信度，检测信号强度（0-1，越高越好）

---

## 📈 可视化分析



```bash
python benchmarks/Video-Bench/utils/plot_radar.py \
  benchmarks/Video-Bench/results/videoseal_robustness/metrics.json
```

| FNR | Bit Accuracy | Avg Confidence |
| --- | --- | --- |
| ![FNR](results/videoseal_robustness/videoseal_fnr_radar.png) | ![Bit Accuracy](results/videoseal_robustness/videoseal_bit_accuracy_radar.png) | ![Avg Confidence](results/videoseal_robustness/videoseal_avg_confidence_radar.png) |

每张图显示 **5条曲线**，对应 5个攻击强度级别（从弱到强）。

---


## 🏆 致谢

本项目基于以下开源工作：

- **[VideoMarkBench](https://www.kaggle.com/datasets/zhengyuanjiang/videomarkbench/data)** - 视频攻击实现和评估框架
- **[VideoSeal](https://github.com/facebookresearch/videosse)** - Meta Research 的视频水印算法


---
