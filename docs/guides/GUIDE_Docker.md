# 🐳 Docker安装和使用指南（零基础版）

> 📌 **适合人群**：完全不了解Docker的研究者
> 🎯 **目标**：5步运行起AIGC水印Benchmark系统
> ⏱️ **预计时间**：首次安装约30分钟，后续使用仅需1分钟

---

## 📚 目录

1. [为什么使用Docker？](#1-为什么使用docker)
2. [安装Docker](#2-安装docker)
3. [安装NVIDIA Docker（GPU支持）](#3-安装nvidia-docker-gpu支持)
4. [理解项目的目录和模型缓存](#4-理解项目的目录和模型缓存)
5. [快速开始（5步到运行）](#5-快速开始5步到运行)
6. [常用命令速查表](#6-常用命令速查表)
7. [使用场景详解](#7-使用场景详解)
8. [开发者模式](#8-开发者模式)
9. [故障排查FAQ](#9-故障排查faq)
10. [性能优化建议](#10-性能优化建议)

---

## 1. 为什么使用Docker？

### 🤔 Docker是什么？

**简单类比**：Docker就像一个"虚拟的计算机"，它可以：
- 📦 **打包**：把你的代码和所有依赖（Python、CUDA、ffmpeg等）打包在一起
- 🚀 **一致性**：在任何机器上运行结果都一样（避免"在我电脑上能跑"的尴尬）
- 🔒 **隔离**：不会污染你的系统环境，删除容器即可完全清理

**技术术语**：
- **镜像（Image）**：类似"虚拟机快照"，包含完整的运行环境
- **容器（Container）**：镜像的运行实例，类似"启动虚拟机"
- **Volume**：主机和容器之间共享的文件夹

### ✅ 对比传统安装方式

| 特性 | 传统安装（conda/pip） | Docker安装 |
|-----|----------------------|-----------|
| 环境配置 | 需要手动安装Python、CUDA、ffmpeg、libsndfile等 | 一键构建，全自动 |
| 依赖冲突 | 可能与现有环境冲突（PyTorch版本、CUDA版本等） | 完全隔离，互不干扰 |
| 可复现性 | 不同机器可能结果不同（系统库版本差异） | 完全一致 |
| 清理 | 卸载复杂，可能留下残留文件 | 删除容器即可 |
| 分享 | 需要详细的安装文档，其他人可能装不上 | 一个命令复现环境 |

### 🎯 本项目使用Docker的好处

1. **GPU支持开箱即用**：自动配置CUDA 11.8、cuDNN 8环境
2. **避免依赖地狱**：不用担心PyTorch、ffmpeg、libsndfile版本冲突
3. **共享模型缓存**：复用你已有的Hugging Face模型（不重复下载）
4. **方便分享**：其他研究者可以一键复现你的实验
5. **多项目隔离**：同时运行多个不同版本的水印项目

### 💡 Docker vs Conda

| 工具 | 隔离级别 | 适用场景 |
|-----|---------|---------|
| Conda | Python环境隔离 | 单机开发，Python包管理 |
| Docker | 系统级隔离（包含OS、系统库） | 部署、分享、多环境管理 |

**结论**：Docker和Conda不冲突，是互补工具。本项目推荐Docker。

---

## 2. 安装Docker

### 🐧 Linux (Ubuntu/Debian)

#### 步骤1: 卸载旧版本（如果有）

```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
```

#### 步骤2: 安装依赖

```bash
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```

#### 步骤3: 添加Docker官方GPG密钥

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

#### 步骤4: 设置稳定版仓库

```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

#### 步骤5: 安装Docker Engine

```bash
sudo apt-get update
sudo apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-compose-plugin
```

#### 步骤6: 验证安装

```bash
sudo docker run hello-world
```

**预期输出**：
```
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

#### 步骤7: 配置非root用户（可选但推荐）

```bash
# 添加当前用户到docker组
sudo usermod -aG docker $USER

# 重新登录使配置生效
# 或者运行：newgrp docker

# 验证无需sudo运行
docker run hello-world
```

danhzuyi
---

## 3. 安装NVIDIA Docker（GPU支持）

### 🎯 为什么需要？

本项目的水印算法（VideoSeal、AudioSeal、PostMark）都需要GPU加速。NVIDIA Docker（nvidia-docker2）让容器能够访问主机的GPU。

### ⚠️ 前提条件

- ✅ 已安装NVIDIA驱动（运行`nvidia-smi`能看到GPU信息）
- ✅ 已安装Docker Engine（上一步完成）
- ✅ GPU型号：GTX 10系列及以上，或专业卡（Tesla、Quadro等）

### 📦 安装步骤（Linux）

#### 步骤1: 添加NVIDIA Docker仓库

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

#### 步骤2: 安装nvidia-docker2

```bash
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

#### 步骤3: 重启Docker服务

```bash
sudo systemctl restart docker
```

#### 步骤4: 验证GPU访问

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**预期输出**：应该能看到你的GPU信息（型号、显存、驱动版本等）

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   45C    P0    50W / 250W |      0MiB / 11264MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```


---

## 4. 理解项目的目录和模型缓存

### 📁 项目目录结构

```
/fs-computility/wangxuhong/limeilin/
│
├── .cache/huggingface/          # 👈 你的AI模型存储位置（实际路径）
│   ├── hub/                     # 👈 Hugging Face Hub模型缓存
│   │   ├── models--stabilityai--stable-diffusion-2-1-base/
│   │   │   └── snapshots/<hash>/  # 实际模型文件
│   │   ├── models--mistralai--Mistral-7B-Instruct-v0.2/
│   │   │   └── snapshots/<hash>/
│   │   └── ...
│   └── transformers/            # 👈 Transformers缓存
│
└── AIGC-Watermark-Toolkit/ # 👈 项目根目录
    ├── config/                  # 👈 配置文件
    │   └── default_config.yaml  # 统一配置文件
    ├── src/                     # 👈 源代码
    │   ├── unified/             # 统一引擎
    │   ├── text_watermark/      # 文本水印（PostMark）
    │   ├── image_watermark/     # 图像水印（VideoSeal）
    │   ├── audio_watermark/     # 音频水印（AudioSeal）
    │   ├── video_watermark/     # 视频水印（VideoSeal）
    │   └── utils/               # 工具（PathManager）
    ├── tests/                   # 👈 测试脚本
    ├── benchmarks/              # 👈 Benchmark套件
    │   ├── Image-Bench/
    │   │   ├── dataset/         # 👈 W-Bench数据集（需自行下载）
    │   │   │   └── W-Bench/DISTORTION_1K/
    │   │   └── results/         # 👈 评估结果（Docker自动创建）
    │   ├── VINE/                # Git子模块（视频评估）
    │   └── AudioMarkBench/      # Git子模块（音频评估）
    ├── outputs/                 # 👈 AI生成内容输出（Docker自动创建）
    ├── models/                  # 👈 占位符目录（空的，不使用）
    ├── Dockerfile               # 👈 Docker镜像构建文件
    ├── docker-compose.yml       # 👈 Docker Compose配置
    └── .dockerignore            # 👈 Docker构建排除规则
```

### 🔑 核心理解：模型缓存挂载

#### 问题：为什么项目的`models/`目录是空的？

**答案**：因为你的模型实际存储在 `/gotoyourpath/.cache/huggingface`

#### 解决方案：通过Volume挂载

在 `docker-compose.yml` 中，我们将**你的实际模型缓存路径**挂载到容器：

```yaml
volumes:
  # 主机路径 → 容器路径
  - /gotoyourpath/.cache/huggingface:/cache/huggingface
```

然后通过环境变量告诉项目使用这个路径：

```yaml
environment:
  - HF_HOME=/cache/huggingface
  - HF_HUB_CACHE=/cache/huggingface/hub
```

#### 工作原理

1. **主机**：模型存储在 `/fs-computility/wangxuhong/limeilin/.cache/huggingface`
2. **容器**：通过`HF_HOME=/cache/huggingface`访问模型
3. **项目代码**：`src/utils/path_manager.py`读取`HF_HOME`环境变量
4. **结果**：项目自动找到模型，无需重复下载

#### 好处

- ✅ **不重复下载**：直接使用你已有的Stable Diffusion、Mistral等模型
- ✅ **跨项目共享**：多个Docker容器可以共享同一份模型
- ✅ **节省空间**：不会在项目的 `models/` 目录重复存储
- ✅ **灵活配置**：修改主机路径即可切换不同的模型缓存

---

## 5. 快速开始（5步到运行）

### 步骤1: 克隆项目

```bash
cd /fs-computility/wangxuhong/limeilin/
git clone --recurse-submodules https://github.com/MillionMillionLi/AIGC-Watermark-Toolkit.git
cd AIGC-Watermark-Toolkit
```

> 💡 `--recurse-submodules` 会自动克隆VINE和AudioMarkBench子模块

**如果忘记加 `--recurse-submodules`**：

```bash
git submodule update --init --recursive
```

### 步骤2: 准备数据集（可选，仅运行benchmark需要）

```bash
# 创建数据集目录
mkdir -p benchmarks/Image-Bench/dataset

# 下载W-Bench数据集
# 参考：benchmarks/Image-Bench/README.md
# 数据集链接：[根据README中的链接下载]

# 解压到正确位置
# 最终结构：benchmarks/Image-Bench/dataset/W-Bench/DISTORTION_1K/image/
```

> ⚠️ 如果跳过此步骤，可以运行其他测试，但无法运行Image-Bench评估

### 步骤3: 构建Docker镜像

```bash
# 首次构建需要10-15分钟（下载基础镜像和安装依赖）
docker-compose build

# 查看构建的镜像
docker images | grep aigc
```

**预期输出**：

```
REPOSITORY                  TAG       IMAGE ID       CREATED          SIZE
aigc-watermark-toolkit      latest    abc123def456   2 minutes ago    5.2GB
```

**构建过程说明**：
- 下载PyTorch基础镜像（~2GB）
- 安装系统依赖（ffmpeg、libsndfile等）
- 安装Python依赖（requirements.txt）
- 初始化Git子模块（VINE、AudioMarkBench）
- 总镜像大小：约5-6GB

> 💡 **加速技巧**：使用BuildKit加速构建
> ```bash
> DOCKER_BUILDKIT=1 docker-compose build
> ```

### 步骤4: 验证GPU访问

```bash
# 测试GPU是否可用
docker-compose run --rm toolkit nvidia-smi
```

**预期输出**：应该能看到GPU信息（型号、显存、温度等）

**如果失败**：检查nvidia-docker2是否安装（参考第3章）

### 步骤5: 运行测试

```bash
# 方案A: 运行快速验证测试（推荐首次使用）
docker-compose run --rm toolkit python -m pytest tests/ -v -k "not slow" -x

# 方案B: 运行单个测试（验证PostMark文本水印）
docker-compose run --rm toolkit python -m pytest tests/test_mistral_postmark.py -v

# 方案C: 交互式探索
docker-compose run --rm toolkit bash
# 进入容器后可以手动运行命令
```

**成功标志**：
```
============================= test session starts ==============================
...
tests/test_mistral_postmark.py::test_postmark_watermark PASSED         [100%]

============================== 1 passed in 10.23s ===============================
```

---

## 6. 常用命令速查表

### 🚀 启动和交互

```bash
# 1. 进入交互式bash环境（探索容器）
docker-compose run --rm toolkit bash

# 2. 在容器内查看模型缓存
docker-compose run --rm toolkit ls -lh /cache/huggingface/hub

# 3. 在容器内验证Python导入
docker-compose run --rm toolkit python -c "from src.unified.watermark_tool import WatermarkTool; print('导入成功！')"

# 4. 查看环境变量
docker-compose run --rm toolkit env | grep HF

# 5. 查看GPU状态
docker-compose run --rm toolkit nvidia-smi

# 6. 查看CUDA是否可用
docker-compose run --rm toolkit python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 🧪 运行测试

```bash
# 1. 运行所有测试
docker-compose run --rm toolkit python -m pytest tests/ -v

# 2. 运行PostMark文本水印测试
docker-compose run --rm toolkit python -m pytest tests/test_mistral_postmark.py -v

# 3. 运行Stable Diffusion + VideoSeal图像测试
docker-compose run --rm toolkit python -m pytest tests/test_sd_videoseal.py -v

# 4. 运行Bark + AudioSeal音频测试
docker-compose run --rm toolkit python -m pytest tests/test_bark_audioseal.py -v

# 5. 运行Wan2.1 + VideoSeal视频测试
docker-compose run --rm toolkit python -m pytest tests/test_wan_videoseal.py -v

# 6. 运行测试并显示详细输出
docker-compose run --rm toolkit python -m pytest tests/ -v -s

# 7. 运行测试并在失败时停止
docker-compose run --rm toolkit python -m pytest tests/ -v -x
```

### 📊 运行Benchmark

```bash
# 1. 运行Image-Bench快速测试（10张图像）
docker-compose run --rm toolkit python benchmarks/Image-Bench/run_benchmark.py --max-images 10

# 2. 运行Image-Bench完整评估（1000张图像，约需30分钟-1小时）
docker-compose run --rm toolkit python benchmarks/Image-Bench/run_benchmark.py

# 3. 使用自定义配置运行
docker-compose run --rm toolkit python benchmarks/Image-Bench/run_benchmark.py \
  --config benchmarks/Image-Bench/configs/videoseal_distortion.yaml

# 4. 查看评估结果
cat benchmarks/Image-Bench/results/videoseal_distortion/metrics.json
```

### 💻 开发和调试

```bash
# 1. 修改代码后立即测试（无需重新构建）
vim src/unified/watermark_tool.py
docker-compose run --rm toolkit python -c "from src.unified.watermark_tool import WatermarkTool; ..."

# 2. 使用自定义配置文件
vim config/default_config.yaml
docker-compose run --rm toolkit python benchmarks/Image-Bench/run_benchmark.py

# 3. 在容器内安装额外的包（临时，重启容器后失效）
docker-compose run --rm toolkit pip install ipython
docker-compose run --rm toolkit ipython

# 4. 运行自定义Python脚本
docker-compose run --rm toolkit python << EOF
from src.unified.watermark_tool import WatermarkTool
tool = WatermarkTool()
print("✅ 工具初始化成功！")
EOF
```

### 🔧 容器管理

```bash
# 1. 查看运行中的容器
docker ps

# 2. 停止所有容器
docker-compose down

# 3. 删除容器和网络（不删除镜像）
docker-compose down --volumes

# 4. 清理所有未使用的容器和镜像（释放磁盘空间）
docker system prune -a

# 5. 查看Docker磁盘占用
docker system df

# 6. 重新构建镜像（修改requirements.txt后）
docker-compose build --no-cache

# 7. 仅重建依赖层（快速）
docker-compose build
```


---

## 7. 使用场景详解

### 场景1: 快速验证工具是否正常工作

**目标**：确认环境配置正确，所有模块可导入

```bash
docker-compose run --rm toolkit bash -c "
python << EOF
from src.unified.watermark_tool import WatermarkTool
from src.text_watermark.postmark_watermark import PostMarkWatermark
from src.image_watermark.videoseal_image_watermark import VideoSealImageWatermark
from src.audio_watermark.audio_watermark import AudioWatermark
from src.video_watermark.video_watermark import VideoWatermark
import torch

print('✅ 所有模块导入成功！')
print(f'✅ CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU型号: {torch.cuda.get_device_name(0)}')
    print(f'✅ CUDA版本: {torch.version.cuda}')
EOF
"
```

### 场景2: 运行图像水印Benchmark（VideoSeal评估）

**目标**：评估VideoSeal在W-Bench DISTORTION_1K上的性能

```bash
# 步骤1: 确认数据集已准备
ls benchmarks/Image-Bench/dataset/W-Bench/DISTORTION_1K/

# 步骤2: 运行快速测试（10张图像，验证流程）
docker-compose run --rm toolkit \
  python benchmarks/Image-Bench/run_benchmark.py \
  --max-images 10 \
  --config benchmarks/Image-Bench/configs/videoseal_distortion.yaml

# 步骤3: 运行完整评估（1000张图像，约需30分钟-1小时）
docker-compose run --rm toolkit \
  python benchmarks/Image-Bench/run_benchmark.py \
  --config benchmarks/Image-Bench/configs/videoseal_distortion.yaml

# 步骤4: 查看结果
cat benchmarks/Image-Bench/results/videoseal_distortion/metrics.json | jq '.'
```

**预期输出**：JSON格式的评估结果，包含PSNR、SSIM、LPIPS、检测率等指标



## 8. 开发者模式

### 💡 核心理解：代码热更新

由于 `src/`、`tests/`、`config/` 目录通过volume挂载，你在**主机上**修改代码后，**容器内**会立即生效，无需重新构建镜像！

### 🔥 开发工作流

```bash
# 步骤1: 在主机上编辑代码（使用你喜欢的编辑器）
vim src/unified/watermark_tool.py
# 或者使用VS Code、PyCharm等

# 步骤2: 立即在Docker中测试
docker-compose run --rm toolkit python << 'EOF'
from src.unified.watermark_tool import WatermarkTool
tool = WatermarkTool()
print("✅ 修改已生效！")
EOF

# 步骤3: 运行测试验证
docker-compose run --rm toolkit python -m pytest tests/test_unified_engine.py -v

# 步骤4: 满意后提交代码
git add src/unified/watermark_tool.py
git commit -m "feat: add new feature"
git push
```

### ❓ 何时需要重新构建镜像？

#### ❌ **不需要**重新构建的情况

- ✅ 修改 `src/` 中的Python代码
- ✅ 修改 `config/` 中的配置文件
- ✅ 修改 `tests/` 中的测试脚本
- ✅ 添加新的Python文件（如`src/new_module.py`）

**原因**：这些目录通过volume挂载，容器直接访问主机文件

#### ✅ **需要**重新构建的情况

- ⚠️ 修改 `requirements.txt`（添加/删除Python包）
- ⚠️ 修改 `Dockerfile`（改变系统依赖或构建步骤）
- ⚠️ 修改 `setup.py`（改变项目安装配置）

**重新构建命令**：

```bash
docker-compose build
```

**快速重建（利用缓存）**：

```bash
docker-compose build --pull
```

