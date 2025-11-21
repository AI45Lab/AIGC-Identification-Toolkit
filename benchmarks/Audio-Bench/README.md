# Audio-Bench: 音频水印鲁棒性评估基准

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: AudioMark](https://img.shields.io/badge/Dataset-AudioMark-green.svg)](https://github.com/moyangkuo/AudioMarkBench)

> 评估音频水印算法（AudioSeal）在多种音频攻击下的鲁棒性，基于 AudioMark 数据集（106个音频样本 × 45种攻击配置）。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 核心依赖
pip install torch torchaudio librosa julius soundfile audiomentations pydub

# 可视化依赖
pip install matplotlib numpy scipy pyyaml tqdm
```

### 2. 下载数据集

下载数据集（Google Drive）：[AudioMark 数据集](https://drive.google.com/drive/folders/1037mBf4LoGq0CDxe6hYx5fNNv56AY_9e)。下载后将文件放到 `benchmarks/Audio-Bench/dataset/audiomark/`。



### 3. 运行评估


```bash
python benchmarks/Audio-Bench/run_benchmark.py
```

**结果输出**：`benchmarks/Audio-Bench/results/audioseal_robustness/`

---

## 📊 评估流程

### 支持的攻击类型（9种 × 5个强度 = 45种配置）

| 攻击类型 | 强度参数 | 说明 |
|---------|---------|------|
| **Gaussian Noise** | [40, 30, 20, 10, 5] | 高斯噪声（SNR in dB，越低攻击越强） |
| **Background Noise** | [40, 30, 20, 10, 5] | 背景噪声（SNR in dB） |
| **Time Stretch** | [0.8, 0.9, 1.0, 1.1, 1.2] | 时间拉伸（速度因子，1.0=原速） |
| **Quantization** | [4, 8, 16, 32, 64] | 量化（比特级数，越低攻击越强） |
| **Lowpass Filter** | [0.1, 0.2, 0.3, 0.4, 0.5] | 低通滤波（截止频率比例） |
| **Highpass Filter** | [0.1, 0.2, 0.3, 0.4, 0.5] | 高通滤波（截止频率比例） |
| **Smooth** | [6, 10, 14, 18, 22] | 平滑（移动平均窗口大小） |
| **Echo** | [(0.1,0.1), ..., (0.5,0.5)] | 回声（延迟秒数, 音量） |
| **MP3 Compression** | [64, 96, 128, 192, 256] | MP3压缩（比特率 kbps，越低攻击越强） |

### 评估指标

#### 质量指标（原音频 vs 水印音频）
- **SNR (Signal-to-Noise Ratio)**: 信噪比，单位 dB，越高越好（通常>40dB表示高质量）

#### 鲁棒性指标（按攻击类型和强度）
- **TPR (prob)**: 基于检测概率的真阳性率，检测概率 > `tau_prob`(0.15) 即判定为检测成功
- **Bit Accuracy**: 位准确率，正确提取的水印比特数占总比特数的比例（0-1）
- **Average Confidence**: 检测成功时的平均置信度（0-1）


---

## 📈 可视化分析


```bash
python benchmarks/Audio-Bench/utils/plot_radar.py \
  benchmarks/Audio-Bench/results/audioseal_robustness/metrics.json
```

| TPR (Detection Probability) | Avg Confidence | Bit Accuracy |
| --- | --- | --- |
| ![TPR prob](results/audioseal_robustness/audioseal_tpr_prob_radar.png) | ![TPR BA](results/audioseal_robustness/audioseal_avg_confidence_radar.png) | ![Bit Accuracy](results/audioseal_robustness/audioseal_bit_accuracy_radar.png) |

每张图显示 **5条曲线**，对应 5个攻击强度级别（从弱到强）。



---

## 🏆 致谢

本项目基于以下开源工作：

- **[AudioMarkBench](https://github.com/moyangkuo/AudioMarkBench)** - 音频攻击实现和评估框架
- **[AudioSeal](https://github.com/facebookresearch/audioseal)** - Meta AI 的音频水印算法
- **[Image-Bench](../Image-Bench/)** - 评估流程架构设计参考

---

