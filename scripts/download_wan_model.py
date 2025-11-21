"""
下载Wan2.1-T2V-1.3B视频生成模型
"""
import os
import torch
from pathlib import Path

print("=" * 60)
print("开始下载Wan2.1-T2V-1.3B视频生成模型")
print("=" * 60)

# 确定缓存目录
cache_dir = os.environ.get('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
cache_hub_dir = os.path.join(cache_dir, 'hub')

# 确认环境变量
print(f"\nHF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'Not set')}")
print(f"缓存目录: {cache_hub_dir}")

print("\n模型信息:")
print("  名称: Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
print("  大小: ~8.19GB VRAM")
print("  用途: 文本生成视频（比HunyuanVideo轻量）")
print("  分辨率: 832x480 (480p), 支持81帧")
print("  优势: 更小的模型大小，更低的显存需求")

print("\n开始下载模型（使用镜像预计20-40分钟）...")
print("下载进度会显示在下方：\n")

try:
    from diffusers import DiffusionPipeline

    # 下载模型（trust_remote_code=True是Wan2.1必需的）
    pipe = DiffusionPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,  # 重要：Wan2.1需要trust_remote_code
    )

    print("\n" + "=" * 60)
    print("✅ Wan2.1模型下载成功！")
    print("=" * 60)

    # 检查下载位置
    if os.path.exists(cache_hub_dir):
        print(f"\n✅ 缓存目录已创建: {cache_hub_dir}")
        # 列出下载的模型
        import subprocess
        result = subprocess.run(
            ["find", cache_hub_dir, "-name", "*Wan*", "-type", "d", "-maxdepth", "2"],
            capture_output=True, text=True
        )
        if result.stdout:
            print(f"✅ 找到模型目录:\n{result.stdout}")
    else:
        print(f"⚠️  缓存目录未找到: {cache_hub_dir}")

    print("\n✅ 模型已准备就绪，可以开始使用！")
    print("\n下一步:")
    print("  1. 创建Wan2.1生成器类（wan_video_generator.py）")
    print("  2. 修改video_watermark.py使用新的生成器")
    print("  3. 更新配置文件（config/default_config.yaml）")

except Exception as e:
    print("\n" + "=" * 60)
    print(f"❌ 下载失败: {e}")
    print("=" * 60)
    print("\n可能的原因:")
    print("1. 网络连接问题 - 请检查网络或代理设置")
    print("2. 磁盘空间不足 - 需要至少10GB空闲空间")
    print("3. 环境变量未设置 - 请确认HF_HOME等变量已设置")
    print("4. diffusers版本太老 - 请运行: pip install -U diffusers")
    print("\n请将错误信息提供给我以获取进一步帮助")
    raise
