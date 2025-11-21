# 环境配置指南

本指南说明如何配置AIGC Identification Toolkit的运行环境，包括模型缓存路径、离线模式和跨平台支持。

## 目录

- [快速开始](#快速开始)
- [环境变量配置](#环境变量配置)
- [路径管理说明](#路径管理说明)
- [离线模式配置](#离线模式配置)
- [跨平台支持](#跨平台支持)
- [常见问题](#常见问题)

## 快速开始

### 默认配置（无需任何设置）

项目开箱即用，无需配置任何环境变量。默认情况下，模型和缓存文件将保存到：

- **Linux/macOS**: `~/.cache/huggingface/`
- **Windows**: `%LOCALAPPDATA%\huggingface\` 或 `C:\Users\<用户名>\AppData\Local\huggingface\`

### 自定义配置（可选）

如果需要自定义路径，可以通过环境变量或配置文件进行设置。

## 环境变量配置

### 方式1：使用.env文件（推荐）

1. 复制环境变量模板：
   ```bash
   cp .env.example .env
   ```

2. 编辑`.env`文件，设置您的自定义路径：
   ```bash
   # 编辑.env文件
   HF_HOME=/your/custom/path/huggingface
   HF_ENDPOINT=https://hf-mirror.com  # 可选：使用镜像
   ```

3. 项目会自动读取`.env`文件（如果使用支持的工具如`python-dotenv`）

### 方式2：Shell环境变量

**临时设置（当前会话）**：
```bash
# Linux/macOS
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub

# Windows PowerShell
$env:HF_HOME="$env:USERPROFILE\.cache\huggingface"
$env:HF_HUB_CACHE="$env:USERPROFILE\.cache\huggingface\hub"
```

**永久设置**：

- **Linux/macOS**: 添加到`~/.bashrc`或`~/.zshrc`
  ```bash
  echo 'export HF_HOME=/fs-computility/wangxuhong/limeilin/.cache/huggingface' >> ~/.bashrc
  source ~/.bashrc
  ```

- **Windows**: 系统设置 → 环境变量 → 新建用户变量

### 核心环境变量说明

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `HF_HOME` | HuggingFace缓存根目录 | `~/.cache/huggingface` |
| `HF_HUB_CACHE` | HF Hub模型缓存目录 | `$HF_HOME/hub` |
| `TRANSFORMERS_CACHE` | Transformers缓存目录 | `$HF_HOME/transformers` |
| `BARK_CACHE_DIR` | Bark TTS模型缓存 | `~/.cache/bark` |
| `HF_ENDPOINT` | HF镜像站点URL | `https://huggingface.co` |

## 路径管理说明

### 路径解析优先级

项目使用智能路径管理系统，按以下优先级解析路径：

1. **环境变量**（最高优先级）
   - 用户通过`HF_HOME`等环境变量明确指定的路径

2. **配置文件**
   - `config/default_config.yaml`中的`cache_dir`配置

3. **智能默认值**（跨平台）
   - Linux/macOS: `~/.cache/huggingface/`
   - Windows: `%LOCALAPPDATA%\huggingface\`

### 路径管理器（PathManager）

项目提供了统一的路径管理器（`src/utils/path_manager.py`），负责所有路径解析：

```python
from src.utils.path_manager import path_manager

# 获取HuggingFace Hub缓存目录
hub_dir = path_manager.get_hf_hub_dir()

# 获取Bark缓存目录
bark_dir = path_manager.get_bark_cache_dir()

# 查找模型
model_path = path_manager.find_model_in_hub("stabilityai/stable-diffusion-2-1-base")
```

## 离线模式配置

### 启用离线模式

如果已经下载了所有需要的模型，可以启用离线模式避免网络访问：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export DIFFUSERS_OFFLINE=1
```

### 离线模式工作流程

1. **首次下载模型**（需要网络）：
   ```bash
   # 下载文本模型
   python scripts/download_postmark_deps.py

   # 下载图像模型
   python scripts/download_sd_model.py

   # 下载视频模型
   python scripts/download_wan_model.py

   # 下载音频模型
   python scripts/download_bark_model.py
   ```

2. **启用离线模式**：
   ```bash
   export TRANSFORMERS_OFFLINE=1
   export HF_HUB_OFFLINE=1
   ```

3. **正常使用**：
   ```python
   from src.unified.watermark_tool import WatermarkTool

   tool = WatermarkTool()  # 会自动使用本地缓存
   ```

### 验证离线模式

```bash
# 断网或设置代理为无效地址
export http_proxy=http://0.0.0.0:0
export https_proxy=http://0.0.0.0:0

# 运行测试（应该仍能正常工作）
python tests/test_text_watermark_dual_algorithm.py
```

## 跨平台支持

### 路径兼容性

项目使用`pathlib.Path`确保跨平台兼容：

```python
from pathlib import Path

# ✅ 正确：跨平台路径
cache_dir = Path.home() / '.cache' / 'huggingface'

# ❌ 错误：硬编码Linux路径
cache_dir = "/home/user/.cache/huggingface"
```

### 平台特定配置

#### Linux
```bash
# 推荐使用XDG标准
export XDG_CACHE_HOME=~/.cache
export HF_HOME=$XDG_CACHE_HOME/huggingface
```

#### macOS
```bash
# 与Linux相同
export HF_HOME=~/.cache/huggingface
```

#### Windows
```powershell
# PowerShell
$env:HF_HOME="$env:USERPROFILE\.cache\huggingface"

# CMD
set HF_HOME=%USERPROFILE%\.cache\huggingface
```

## 使用HuggingFace镜像

### 中国大陆用户推荐配置

```bash
# 设置镜像站点
export HF_ENDPOINT=https://hf-mirror.com

# 验证配置
python -c "import os; print(os.getenv('HF_ENDPOINT'))"
```

### 测试镜像连接

```bash
# 下载一个小模型测试
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print('✅ 镜像连接成功')
"
```

## 常见问题

### Q1: 如何查看当前缓存路径？

```python
from src.utils.path_manager import path_manager

print("HF_HOME:", path_manager.get_hf_home())
print("HF_HUB_CACHE:", path_manager.get_hf_hub_dir())
print("BARK_CACHE:", path_manager.get_bark_cache_dir())
```

### Q2: 如何清理缓存？

```bash
# 查看缓存大小
du -sh ~/.cache/huggingface

# 删除特定模型
rm -rf ~/.cache/huggingface/hub/models--<model-name>

# 清空全部缓存（谨慎！）
rm -rf ~/.cache/huggingface
```

### Q3: 离线模式下提示模型未找到怎么办？

1. 检查模型是否已下载：
   ```bash
   ls -la ~/.cache/huggingface/hub/
   ```

2. 检查环境变量是否正确：
   ```bash
   echo $HF_HOME
   echo $HF_HUB_CACHE
   ```

3. 重新下载模型：
   ```bash
   # 临时禁用离线模式
   unset TRANSFORMERS_OFFLINE
   unset HF_HUB_OFFLINE

   # 运行下载脚本
   python scripts/download_sd_model.py
   ```

### Q4: Windows上路径包含中文或空格怎么办？

尽量避免在路径中使用中文或空格。如果必须使用，请确保路径用引号括起来：

```powershell
$env:HF_HOME="C:\Users\用户名\我的文档\模型缓存"
```

推荐使用纯英文路径：
```powershell
$env:HF_HOME="C:\Models\huggingface"
```

### Q5: 如何迁移已有的模型缓存？

```bash
# 方法1：移动缓存并设置环境变量
mv ~/.cache/huggingface /new/path/huggingface
export HF_HOME=/new/path/huggingface

# 方法2：创建符号链接
ln -s /new/path/huggingface ~/.cache/huggingface
```

## 最佳实践

1. **开发环境**：使用默认路径，无需配置
2. **生产环境**：通过环境变量配置路径，确保可控
3. **离线部署**：提前下载所有模型，启用离线模式
4. **多用户环境**：每个用户使用独立的`HF_HOME`
5. **CI/CD流水线**：在Docker中设置`HF_HOME`到固定路径

## 示例配置

### 开发环境（默认）

不需要任何配置，直接使用：
```bash
python tests/test_text_watermark_dual_algorithm.py
```

### 生产环境（自定义路径）

```bash
# .env文件
HF_HOME=/data/models/huggingface
HF_ENDPOINT=https://hf-mirror.com
TRANSFORMERS_OFFLINE=1
```

### 离线部署（完全离线）

```bash
# 1. 下载所有模型（联网环境）
./scripts/download_all_models.sh

# 2. 打包缓存目录
tar -czf models.tar.gz ~/.cache/huggingface

# 3. 在离线机器上解压并配置
tar -xzf models.tar.gz -C ~
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

## 技术支持

如遇问题，请提供以下信息：

```bash
# 系统信息
uname -a  # Linux/macOS
systeminfo | findstr /C:"OS Name"  # Windows

# Python环境
python --version
pip list | grep -E "torch|transformers|diffusers"

# 环境变量
env | grep -E "HF_|TRANSFORMERS_|XDG_CACHE"

# 路径信息
python -c "from src.utils.path_manager import path_manager; \
print('HF_HUB:', path_manager.get_hf_hub_dir()); \
print('Exists:', path_manager.get_hf_hub_dir().exists())"
```

在GitHub Issues中提交问题时附上以上信息可以帮助我们更快定位问题。
