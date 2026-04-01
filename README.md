# UniMark

<div align="center">
  <!-- 项目logo占位符 - 需要logo图片 -->
  <!-- <a href="https://github.com/your-repo-link">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->


  <h3 align="center">UniMark: AIGC Watermark & Identification Toolkit</h3>
  
  <p align="center">
    一站式开源水印标识开发套件，支持文本、图像、音频和视频内容的显式标识，隐式标识和隐水印功能
    <br />
    <a href="#使用方法"><strong>快速开始 »</strong></a>
    <br />
  </p>

</div>

<div align="center">
  <a href="./README_EN.md">English</a> | 简体中文
</div>

---

## 关于项目

<!-- 项目截图占位符 - 需要网页界面截图 -->
<!-- [![产品截图][product-screenshot]](https://example.com) -->

本项目提供一站式开源水印标识开发套件。支持文本、图像、音频和视频四大模态，具备显式水印标识和隐水印功能，覆盖GB 45438-2025《网络安全技术 人工智能生成合成内容标识方法》标准规定的标识范围。


> **应用案例**：本项目作为实训平台，支撑了**上海市人工智能安全专项课程 2025 短训班**的教学与实战演练，帮助学员深度掌握AIGC水印嵌入与检测技术。

### 为什么选择我们？

- **全面覆盖**：支持GB 45438-2025标准要求的标识方法
- **多模态支持**：统一处理文本、图像、音频和视频内容的水印添加与检测
- **双模式操作**：既支持AI内容生成，也支持现有文件处理

### 构建技术

* [![Python][Python.org]][Python-url]
[![PyTorch][PyTorch.org]][PyTorch-url]
[![Flask][Flask.palletsprojects.com]][Flask-url] [![Transformers][Transformers-badge]][Transformers-url] [![Diffusers][Diffusers-badge]][Diffusers-url]

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

---

## 📑 目录

- [效果展示](#效果展示)
- [开始使用](#开始使用)
  - [安装](#安装)
- [使用方法](#使用方法)
- [Benchmarks](#benchmarks)
  - [Image-Bench](#image-bench)
  - [Audio-Bench](#audio-bench)
  - [Video-Bench](#video-bench)

---

## 🎬 效果展示

### 图像标识效果

<table width="100%">
  <tr>
    <th align="center" width="33%">原始图像</th>
    <th align="center" width="33%">隐水印图像</th>
    <th align="center" width="34%">显式标识图像</th>
  </tr>
  <tr>
    <td align="center"><img src="docs/demo/original_image.png" width="100%"/></td>
    <td align="center"><img src="docs/demo/watermarked_image.png" width="100%"/></td>
    <td align="center"><img src="docs/demo/visible_marked_image.png" width="100%"/></td>
  </tr>
</table>

### 视频标识效果

<table width="100%">
  <tr>
    <th align="center" width="33%">原始视频</th>
    <th align="center" width="33%">隐水印视频</th>
    <th align="center" width="34%">显式标识视频</th>
  </tr>
  <tr>
    <td align="center"><img src="docs/demo/original_video.gif" width="100%"/></td>
    <td align="center"><img src="docs/demo/watermarked_video.gif" width="100%"/></td>
    <td align="center"><img src="docs/demo/visible_marked_video.gif" width="100%"/></td>
  </tr>
</table>

---

## 开始使用

### 安装

#### 🔧 传统安装

1. 克隆仓库

   ```bash
   git clone --recurse-submodules https://github.com/MillionMillionLi/AIGC-Watermark-Toolkit.git
   cd AIGC-Watermark-Toolkit
   ```

2. 安装核心依赖

   ```bash
   pip install -r requirements.txt
   ```

3. 安装系统依赖

   ```bash
   sudo apt install ffmpeg
   ```

4. （可选）下载 AI 生成模型

   仅当需要使用 AI 生成内容并添加水印功能时才需要此步骤。如果只处理已有文件（上传模式添加水印），可跳过此步骤。

   ```bash
   # 图像生成 + 水印（Stable Diffusion 2.1）
   python scripts/download_sd_model.py

   # 视频生成 + 水印（Wan2.1）
   python scripts/download_wan_model.py

   # 文本生成 + 水印（PostMark + Mistral）
   python scripts/download_postmark_deps.py

   # 音频生成 + 水印（Bark）
   python scripts/download_bark_model.py
   pip install git+https://github.com/suno-ai/bark.git
   ```

5. 配置环境

   ```bash
   export TRANSFORMERS_OFFLINE=1
   export HF_HUB_OFFLINE=1
   export HF_ENDPOINT=https://hf-mirror.com
   ```
#### 🐳 Docker 安装

**前置要求**：需要 NVIDIA GPU 和 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

1. 拉取镜像
   ```bash
   docker pull millionmillionli/aigc-watermark-toolkit:latest
   ```

2. 运行容器
   ```bash
   docker run -d --name aigc-watermark-toolkit \
     --gpus all \
     -v /path/to/your/.cache/huggingface:/cache/huggingface \ ## 将这一行的路径改为你的实际模型缓存路径
     -v $(pwd):/app \
     -e HF_HOME=/cache/huggingface \
     -e PYTHONPATH=/app \
     -e CUDA_VISIBLE_DEVICES=0 \
     --restart unless-stopped \
     millionmillionli/aigc-watermark-toolkit:latest \
     tail -f /dev/null
   ```

   (可选)准备AI模型,仅当需要使用 AI 生成内容并添加水印功能时才需要此步骤。
 
   **需要下载的模型**：
   - 图像生成：Stable Diffusion 2.1 (`stabilityai/stable-diffusion-2-1-base`)
   - 视频生成：Wan2.1 (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`)
   - 文本生成：Mistral 7B + PostMark词嵌入 (`mistralai/Mistral-7B-Instruct-v0.2`)
   - 音频生成：Bark (`suno/bark`)
3. 进入容器
   ```bash
   docker exec -it aigc-watermark-toolkit bash
   ```
**进阶使用**：如需完整配置（benchmark挂载、只读配置等），可使用 `docker-compose up -d`或查看`docker-compose.yml` 文件。





## 使用方法


`WatermarkTool`是推荐的主要入口点，提供统一的接口支持所有模态的水印和显式标识操作。

#### 初始化

```python
from src.unified.watermark_tool import WatermarkTool

# 使用默认配置初始化
tool = WatermarkTool()

# 使用自定义配置初始化
tool = WatermarkTool(config_path="path/to/config.yaml")
```

#### 核心方法

##### embed() - 嵌入水印或显式标识

```python
def embed(self,
          content: Union[str, Path],
          message: str,
          modality: str,
          operation: str = 'watermark',
          **kwargs) -> Union[str, PIL.Image, torch.Tensor, Path]:
    """
    嵌入水印或添加显式标识

    Args:
        content: 输入内容
            - 文本模态: 提示文本(AI生成)或文本文件路径(上传模式)
            - 图像模态: 提示文本(AI生成)或图像文件路径(上传模式)
            - 音频模态: 提示文本(AI生成)或音频文件路径(上传模式)
            - 视频模态: 提示文本(AI生成)或视频文件路径(上传模式)
        message: 要嵌入的水印信息或显式标识文本
        modality: 模态类型 ('text', 'image', 'audio', 'video')
        operation: 操作类型 ('watermark' 或 'visible_mark')
        **kwargs: 模态特定参数

    Returns:
        处理后的内容（格式根据模态而定）
    """
```

**使用示例：**

```python
# 隐式水印（默认operation='watermark'）
img_wm = tool.embed("a cat under the sun", "img_msg", 'image')


# 上传文件模式
img_wm = tool.embed("", "file_msg", 'image', image_input="/path/to/image.jpg")


# 显式标识
marked_img = tool.embed("/path/to/image.jpg", "AI标识", 'image',
                       operation='visible_mark', position='bottom_right')
```

##### extract() - 提取水印或检测显式标识

```python
def extract(self,
           content: Union[str, PIL.Image, torch.Tensor, Path],
           modality: str,
           operation: str = 'watermark',
           **kwargs) -> Dict[str, Any]:
    """
    提取水印或检测显式标识

    Args:
        content: 待检测的内容
        modality: 模态类型
        operation: 操作类型 ('watermark' 或 'visible_mark')
        **kwargs: 检测参数

    Returns:
        检测结果字典:
        {
            'detected': bool,      # 是否检测到水印/标识
            'message': str,        # 提取的消息内容
            'confidence': float,   # 置信度 (0.0-1.0)
            'metadata': dict       # 额外的元数据信息
        }
    """
```

**使用示例：**

```python
# 提取隐式水印
img_result = tool.extract(watermarked_image, 'image')


# 检测显式标识
mark_result = tool.extract(marked_content, 'text', operation='visible_mark')
```

##  Benchmarks

评估各模态水印算法的性能表现，提供标准化的测试数据集、攻击方式和评估指标，帮助用户选择最适合应用场景的算法。


---

### Image-Bench

评估图像水印算法对传统失真攻击（亮度、对比度、模糊、噪声、JPEG压缩）的鲁棒性。

**核心特性**:
-  **数据集**: W-Bench DISTORTION_1K（1000张图像）

-  **评估指标**：: PSNR, SSIM, LPIPS, TPR，Bit accuracy

**快速使用**:
```bash
python benchmarks/Image-Bench/run_benchmark.py
```

**使用自定义数据集**:
1. 准备图像数据：将PNG图像放入自定义目录（如 `benchmarks/Image-Bench/dataset/my_dataset/`）
2. 修改配置 `configs/videoseal_distortion.yaml`：
   ```yaml
   dataset:
     path: benchmarks/Image-Bench/dataset/my_dataset
   ```
**评估指标**：
| 指标类别 | 指标 | 判定阈值 | 指标说明 |
|----------|------|----------|----------|
| **质量** | PSNR | ≥ 35.0 dB | Peak Signal-to-Noise Ratio（峰值信噪比），越高越好 |
| **质量** | SSIM | ≥ 0.95 | Structural Similarity Index（结构相似度），越接近 1 越好 |
| **质量** | LPIPS | ≤ 0.015 | Learned Perceptual Similarity（感知相似度），越低越好 |
| **鲁棒性** | TPR | ≥ 0.80 | True Positive Rate（检测成功率），越高表示鲁棒性越强 |
| **鲁棒性** | Bit Accuracy | ≥ 0.85 | 水印比特准确率，决定解码结果与原始水印的接近程度 |

**结果分析**：
<table>
  <tr>
    <th>TPR</th>
    <th>Bit Accuracy</th>
    <th>质量评估指标</th>
  </tr>
  <tr>
    <td><img src="benchmarks/Image-Bench/results/videoseal_distortion/videoseal_tpr_radar.png" alt="VideoSeal TPR Radar" /></td>
    <td><img src="benchmarks/Image-Bench/results/videoseal_distortion/videoseal_bit_accuracy_radar.png" alt="VideoSeal Bit Accuracy Radar" /></td>
    <td style="vertical-align: top; height: 100%;">
      <table>
        <tr><th>指标</th><th>数值</th><th style="white-space: nowrap;">达到阈值</th></tr>
        <tr><td><strong>PSNR</strong></td><td>45.52 dB</td><td>✅</td></tr>
        <tr><td><strong>SSIM</strong></td><td>0.9953</td><td>✅</td></tr>
        <tr><td><strong>LPIPS</strong></td><td>0.0025</td><td>✅</td></tr>
      </table>
    </td>
  </tr>
</table>

**详细文档**: [benchmarks/Image-Bench/README.md](benchmarks/Image-Bench/README.md)

---

### Audio-Bench

评估音频水印算法（AudioSeal）对多种音频攻击的鲁棒性，覆盖噪声、滤波、压缩等常见干扰。

**核心特性**:
- 📊 **数据集**: [AudioMark Dataset](https://drive.google.com/drive/folders/1037mBf4LoGq0CDxe6hYx5fNNv56AY_9e)
- 🔧 **攻击类型**: 高斯噪声、背景噪声、量化、滤波、平滑、回声、MP3压缩

**快速使用**:
```bash
python benchmarks/Audio-Bench/run_benchmark.py
```

**使用自定义数据集**:
1. 准备音频数据：将音频文件（支持WAV/MP3/FLAC/M4A）放入自定义目录
2. 修改配置 `configs/audioseal_robustness.yaml`：
   ```yaml
   dataset:
     path: benchmarks/Audio-Bench/dataset/my_audio_dataset
   ```
**评估指标**：
| 指标类别 | 指标 | 判定阈值 | 指标说明 |
|----------|------|----------|----------|
| **质量** | SNR | ≥ 20.0 dB | Signal-to-Noise Ratio，原音频 vs 水印音频，越高越好 |
| **鲁棒性** | TPR (Detection Probability) | ≥ 0.80 | 以检测概率判定的真阳性率 |
| **鲁棒性** | Bit Accuracy | ≥ 0.875 | 图案水印比特正确率，越高越好 |

**结果分析**：
<table>
  <tr>
    <th>TPR (Detection Probability)</th>
    <th>Bit Accuracy</th>
    <th>质量评估指标</th>
  </tr>
  <tr>
    <td><img src="benchmarks/Audio-Bench/results/audioseal_robustness/audioseal_tpr_prob_radar.png" alt="AudioSeal TPR Probability Radar" /></td>
    <td><img src="benchmarks/Audio-Bench/results/audioseal_robustness/audioseal_bit_accuracy_radar.png" alt="AudioSeal Bit Accuracy Radar" /></td>
    <td style="vertical-align: top; height: 100%;">
      <table>
        <tr><th>指标</th><th>数值</th><th style="white-space: nowrap;">达到阈值</th></tr>
        <tr><td><strong>SNR</strong></td><td>23</td><td>✅</td></tr>
      </table>
    </td>
  </tr>
</table>


**详细文档**: [benchmarks/Audio-Bench/README.md](benchmarks/Audio-Bench/README.md)

---

### Video-Bench

评估视频水印算法（VideoSeal）在图像级和视频级扰动下的鲁棒性，严格遵循VideoMarkBench论文方法。

**核心特性**:
- 📊 **数据集**: [VideoMarkBench Dataset](https://www.kaggle.com/datasets/zhengyuanjiang/videomarkbench/data)
- 🔧 **攻击类型**: 高斯噪声、模糊、JPEG压缩、裁剪、帧平均、帧交换、帧删除

**快速使用**:
```bash
python benchmarks/Video-Bench/run_benchmark.py
```

**使用自定义数据集**:
1. 准备视频数据：将视频文件（支持MP4/AVI/MOV/MKV）放入自定义目录，支持子目录
2. 修改配置 `configs/videoseal_robustness.yaml`：
   ```yaml
   dataset:
     path: benchmarks/Video-Bench/dataset/my_video_dataset
   ```

**评估指标**：

| 指标类别 | 指标 | 判定阈值 | 指标说明 |
|----------|------|----------|----------|
| **质量** | PSNR | ≥ 35.0 dB | Peak Signal-to-Noise Ratio，越高越好 |
| **质量** | SSIM | ≥ 0.95 | Structural Similarity Index，越接近 1 越好 |
| **质量** | tLP | ≤ 0.20 | Temporal LPIPS，衡量跨帧感知一致性，越低越好 |
| **鲁棒性** | FNR | ≤ 0.01 | False Negative Rate，漏检率，越低表示鲁棒性越强 |
| **鲁棒性** | Bit Accuracy | ≥ 0.85 | 解码比特准确率，越高越好 |

**结果分析**：
<table>
  <tr>
    <th>FNR</th>
    <th>Bit Accuracy</th>
    <th>质量评估指标</th>
  </tr>
  <tr>
    <td><img src="benchmarks/Video-Bench/results/videoseal_robustness/videoseal_fnr_radar.png" alt="VideoSeal FNR Radar" /></td>
    <td><img src="benchmarks/Video-Bench/results/videoseal_robustness/videoseal_bit_accuracy_radar.png" alt="VideoSeal Bit Accuracy Radar" /></td>
    <td style="vertical-align: top; height: 100%;">
      <table>
        <tr><th>指标</th><th>数值</th><th style="white-space: nowrap;">达到阈值</th></tr>
        <tr><td><strong>PSNR</strong></td><td>40.59</td><td>✅</td></tr>
        <tr><td><strong>SSIM</strong></td><td>0.97</td><td>✅</td></tr>
        <tr><td><strong>tLP</strong></td><td>0.001</td><td>✅</td></tr>
      </table>
    </td>
  </tr>
</table>

**详细文档**: [benchmarks/Video-Bench/README.md](benchmarks/Video-Bench/README.md)

---

### Text-Bench（规划中）



<p align="right">(<a href="#readme-top">返回顶部</a>)</p>


## 致谢

本项目基于以下优秀的开源工作构建：

### 水印算法

* [Meta AudioSeal](https://github.com/facebookresearch/audioseal) - 音频水印算法
* [VideoSeal](https://github.com/facebookresearch/videoseal) - 视频/图像水印技术
* [PostMark](https://github.com/your-postmark-repo) - 文本后处理水印算法
* [CredID](https://github.com/your-credid-repo) - 多方文本水印框架
* [PRC-Watermark](https://github.com/rmin2000/PRC-Watermark) - 图像水印算法

### AI 生成模型

* [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) - 文本生成图像模型
* [Wan2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) - 文本生成视频模型
* [Bark](https://github.com/suno-ai/bark) - 文本转语音合成模型

### 评估与基准测试

* [VINE](https://github.com/Shilin-LU/VINE) - W-Bench 数据集和图像失真攻击实现
* [AudioMarkBench](https://github.com/mileskuo42/AudioMarkBench) - 音频水印评估框架



<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 引用

如果您觉得本项目对您的研究有所帮助，请考虑引用我们的论文：

```bibtex
@article{li2025unimark,
  title={UniMark: Artificial Intelligence Generated Content Identification Toolkit},
  author={Meilin Li and Ji He and Yu Yi and Jia Xu and Shanzhe Lei and Yan Teng and Yingchun Wang and Xuhong Wang},
  journal={arXiv preprint arXiv:2512.12324},
  year={2025}
}
```

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

<!-- MARKDOWN链接和图像 -->

[Python.org]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://python.org/
[PyTorch.org]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[Flask.palletsprojects.com]: https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/
[Transformers-badge]: https://img.shields.io/badge/🤗%20Transformers-FFD700?style=for-the-badge
[Transformers-url]: https://huggingface.co/transformers/
[Diffusers-badge]: https://img.shields.io/badge/🧨%20Diffusers-FF6B6B?style=for-the-badge
[Diffusers-url]: https://huggingface.co/docs/diffusers/