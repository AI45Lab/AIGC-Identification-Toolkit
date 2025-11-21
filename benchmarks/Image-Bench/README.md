# Image-Bench: 图像水印鲁棒性评估基准

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: W-Bench](https://img.shields.io/badge/Dataset-W--Bench-green.svg)](https://huggingface.co/datasets/Shilin-LU/W-Bench)

> 评估图像水印算法在传统失真攻击下的鲁棒性，基于W-Bench DISTORTION_1K数据集（1000张图像 × 25种攻击配置）。


---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 必需依赖
pip install pillow numpy torch tqdm pyyaml

# 质量指标计算
pip install scikit-image lpips
```

### 2. 下载数据集


```bash
huggingface-cli download Shilin-LU/W-Bench \
  --repo-type=dataset \
  --local-dir dataset/W-Bench \
  --include "DISTORTION_1K/**"
```

**数据集大小**: ~9GB
**验证下载**: `ls dataset/W-Bench/DISTORTION_1K/image/ | wc -l` 应输出 1000

### 3. 运行评估

```bash
python benchmarks/Image-Bench/run_benchmark.py
```

**结果输出**: `results/videoseal_distortion/metrics.json`

---

## 📊 评估流程

### 支持的攻击类型

| 攻击类型 | 强度参数 | 说明 |
|---------|---------|------|
| **Brightness** | [1.2, 1.4, 1.6, 1.8, 2.0] | 亮度增强（倍数） |
| **Contrast** | [0.2, 0.4, 0.6, 0.8, 1.0] | 对比度降低（倍数） |
| **Blurring** | [1, 3, 5, 7, 9] | 高斯模糊（核大小） |
| **Noise** | [0.01, 0.03, 0.05, 0.07, 0.1] | 高斯噪声（标准差） |
| **JPEG Compression** | [95, 90, 80, 70, 60] | JPEG质量 |

共计 **5种攻击 × 5个强度 = 25种配置**

### 评估指标

#### 质量指标（原图 vs 水印图）
- **PSNR** (Peak Signal-to-Noise Ratio): 峰值信噪比，越高越好（通常>40dB表示高质量）
- **SSIM** (Structural Similarity Index): 结构相似性，范围0-1，越高越好
- **LPIPS** (Learned Perceptual Similarity): 感知相似性，越低越好

#### 鲁棒性指标（按攻击类型）
- **TPR (True Positive Rate)**: 真阳性率，水印正确检测的比例（0-1）
- **Bit Accuracy**: 位准确率，正确提取的水印比特数占总比特数的比例（0-1）
- **Average Confidence**: 检测成功时的平均置信度

---

## 📈 可视化分析

生成雷达图以可视化水印鲁棒性：

```bash
python benchmarks/Image-Bench/utils/plot_radar.py \
    benchmarks/Image-Bench/results/videoseal_distortion/metrics.json
```


|  |  |  |
| --- | --- | --- |
| ![VideoSeal Avg Confidence Radar](results/videoseal_distortion/videoseal_avg_confidence_radar.png) | ![VideoSeal Bit Accuracy Radar](results/videoseal_distortion/videoseal_bit_accuracy_radar.png) | ![VideoSeal TPR Radar](results/videoseal_distortion/videoseal_tpr_radar.png) |






---

## 🏆 致谢

本项目基于以下开源工作：

- **[VINE](https://github.com/Shilin-LU/VINE)** - W-Bench数据集和失真攻击实现
- **[VideoSeal](https://github.com/facebookresearch/videoseal)** - 视频/图像水印算法




