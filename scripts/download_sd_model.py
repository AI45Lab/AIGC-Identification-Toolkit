"""
下载Stable Diffusion 2.1模型到HuggingFace缓存目录
"""
import os
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline

print("=" * 60)
print("开始下载Stable Diffusion 2.1模型")
print("=" * 60)

# 确定缓存目录
cache_dir = os.environ.get('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
cache_hub_dir = os.path.join(cache_dir, 'hub')

# 确认环境变量
print(f"\nHF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'Not set')}")
print(f"缓存目录: {cache_hub_dir}")

print("\n开始下载模型（大小约5.2GB，使用镜像预计10-30分钟）...")
print("下载进度会显示在下方：\n")

try:
    # 下载模型（会自动创建目录结构）
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        cache_dir=None,  # 使用环境变量中的HF_HOME
    )

    print("\n" + "=" * 60)
    print("✅ 模型下载成功！")
    print("=" * 60)

    # 检查下载位置
    if os.path.exists(cache_hub_dir):
        print(f"\n✅ 缓存目录已创建: {cache_hub_dir}")
        # 列出下载的模型
        import subprocess
        result = subprocess.run(
            ["find", cache_hub_dir, "-name", "*stable-diffusion*", "-type", "d", "-maxdepth", "2"],
            capture_output=True, text=True
        )
        if result.stdout:
            print(f"✅ 找到模型目录:\n{result.stdout}")
    else:
        print(f"⚠️  缓存目录未找到: {cache_hub_dir}")

    print("\n下一步: 运行测试脚本验证图像水印功能")

except Exception as e:
    print("\n" + "=" * 60)
    print(f"❌ 下载失败: {e}")
    print("=" * 60)
    print("\n可能的原因:")
    print("1. 网络连接问题 - 请检查网络或代理设置")
    print("2. 磁盘空间不足 - 需要至少8GB空闲空间")
    print("3. 环境变量未设置 - 请确认HF_HOME等变量已设置")
    print("\n请将错误信息提供给我以获取进一步帮助")
    raise
