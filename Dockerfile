# ====================================
# AIGC-Identification-Toolkit Docker配置
# 多模态水印Benchmark系统
# ====================================

# =====================================
# 阶段1: 基础环境
# =====================================

# FROM opencompass-cn-beijing.cr.volces.com/dockerhub/pytorch/pytorch:2.4.0-cuda11.8-cudnn8-runtime AS base
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime AS base
# 设置工作目录
# 所有后续操作都在 /app 目录下进行
WORKDIR /app

# =====================================
# 安装系统依赖
# =====================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsndfile1 \
    libsndfile1-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*
# =====================================
# 阶段2: Python依赖安装
# =====================================
FROM base AS dependencies

COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =====================================
# 阶段3: 项目代码构建
# =====================================
FROM dependencies AS builder

# 复制项目代码
COPY . .

# 生产镜像不需要 14GB benchmark 子模块，跳过下载以减小镜像大小
# 开发者可以在本地手动初始化子模块: git submodule update --init --recursive
# RUN git submodule update --init --recursive || echo "Git submodules initialization skipped"

# 安装项目本身
RUN pip install --no-cache-dir -e .

# =====================================
# 阶段4: 最终运行环境
# =====================================
FROM builder AS runtime

# ====================================
# 创建非root用户（安全最佳实践）
# ====================================
RUN useradd -m -u 1000 -s /bin/bash aigc && \
    chown -R aigc:aigc /app

# 切换到非root用户
# 后续所有命令以aigc用户身份运行
USER aigc

# ====================================
# 设置环境变量
# ====================================
# PYTHONUNBUFFERED=1: 实时输出print内容，不缓冲
ENV PYTHONUNBUFFERED=1

# PYTHONPATH=/app: 添加项目根目录到Python搜索路径，确保 `from src.xxx import` 能正常工作
ENV PYTHONPATH=/app

# ====================================
# 默认命令
# ====================================
# 启动bash交互式环境，方便用户探索
# 用户可以通过 docker-compose run toolkit <command> 覆盖
CMD ["bash"]
