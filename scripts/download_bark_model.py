#!/usr/bin/env python3
"""
Bark模型下载和修复脚本
解决PyTorch 2.6的weights_only兼容性问题
"""

import os
import sys
import warnings
from pathlib import Path

# 设置环境变量（使用跨平台路径）
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = str(Path.home() / '.cache' / 'huggingface')
if 'XDG_CACHE_HOME' not in os.environ:
    os.environ['XDG_CACHE_HOME'] = str(Path.home() / '.cache')

# 临时修复torch.load的weights_only问题
import torch
original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    """
    修复Bark模型加载的torch.load兼容性问题
    PyTorch 2.6默认weights_only=True，但Bark旧模型需要weights_only=False
    """
    # 如果没有明确指定weights_only，设置为False以兼容旧模型
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
        # 添加安全白名单
        if hasattr(torch.serialization, 'add_safe_globals'):
            try:
                # 允许numpy.core.multiarray.scalar
                import numpy as np
                torch.serialization.add_safe_globals([np.core.multiarray.scalar])
            except:
                pass

    return original_torch_load(f, *args, **kwargs)

# 应用补丁
torch.load = patched_torch_load
print("✅ 已应用torch.load兼容性补丁")

# 现在导入Bark
try:
    from bark import generate_audio, SAMPLE_RATE, preload_models
    import scipy.io.wavfile
    print("✅ Bark模块导入成功")
except ImportError as e:
    print(f"❌ Bark模块导入失败: {e}")
    print("请先安装Bark: pip install git+https://github.com/suno-ai/bark.git")
    sys.exit(1)

def download_bark_models():
    """下载Bark模型文件"""
    print("\n" + "="*60)
    print("开始下载Bark模型...")
    print("="*60)

    # 检查缓存目录
    cache_dir = os.path.join(os.environ['XDG_CACHE_HOME'], 'suno', 'bark_v0')
    print(f"缓存目录: {cache_dir}")

    # 预加载所有模型（会自动下载缺失的文件）
    try:
        print("\n正在预加载Bark模型（首次运行需要下载约5GB文件）...")
        print("这可能需要几分钟，请耐心等待...\n")

        preload_models()

        print("\n✅ 模型预加载成功！")

        # 列出下载的文件
        print(f"\n已下载的模型文件:")
        if os.path.exists(cache_dir):
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    if file.endswith('.pt'):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        print(f"  ✓ {file} ({file_size:.1f} MB)")

        return True

    except Exception as e:
        print(f"\n❌ 模型预加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bark_generation():
    """测试Bark音频生成"""
    print("\n" + "="*60)
    print("测试Bark音频生成...")
    print("="*60)

    test_texts = [
        "Hello, this is a test.",
        "你好，这是一个测试。"
    ]

    for i, text in enumerate(test_texts):
        try:
            print(f"\n测试 {i+1}: '{text}'")

            # 生成音频
            audio_array = generate_audio(text, text_temp=0.7, waveform_temp=0.7)

            # 保存测试文件
            output_file = f"test_bark_{i+1}.wav"
            scipy.io.wavfile.write(output_file, SAMPLE_RATE, audio_array)

            print(f"  ✓ 生成成功，已保存到: {output_file}")
            print(f"  ✓ 音频长度: {len(audio_array)/SAMPLE_RATE:.2f} 秒")

        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n✅ 所有测试通过！")
    return True

def verify_model_files():
    """验证Bark模型文件完整性"""
    print("\n" + "="*60)
    print("验证Bark模型文件完整性...")
    print("="*60)

    cache_dir = os.path.join(os.environ['XDG_CACHE_HOME'], 'suno', 'bark_v0')

    # 必需的模型文件
    required_files = [
        'text.pt', 'text_2.pt',
        'coarse.pt', 'coarse_2.pt',
        'fine.pt', 'fine_2.pt'
    ]

    found_files = []
    missing_files = []

    if os.path.exists(cache_dir):
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                if file.endswith('.pt'):
                    found_files.append(file)

    # 检查缺失的文件（至少需要text, coarse, fine的其中一个版本）
    has_text = any(f in found_files for f in ['text.pt', 'text_2.pt'])
    has_coarse = any(f in found_files for f in ['coarse.pt', 'coarse_2.pt'])
    has_fine = any(f in found_files for f in ['fine.pt', 'fine_2.pt'])

    print(f"\n找到的模型文件: {len(found_files)} 个")
    for file in sorted(found_files):
        print(f"  ✓ {file}")

    if has_text and has_coarse and has_fine:
        print("\n✅ 模型文件完整！")
        return True
    else:
        print("\n⚠️  模型文件不完整:")
        if not has_text:
            print("  ✗ 缺少文本模型 (text.pt 或 text_2.pt)")
        if not has_coarse:
            print("  ✗ 缺少粗粒度音频模型 (coarse.pt 或 coarse_2.pt)")
        if not has_fine:
            print("  ✗ 缺少细粒度音频模型 (fine.pt 或 fine_2.pt)")
        print("\n需要重新下载模型...")
        return False

def main():
    """主函数"""
    print("="*60)
    print("Bark模型下载和修复工具")
    print("="*60)

    # 步骤1: 验证现有模型
    is_complete = verify_model_files()

    # 步骤2: 下载/更新模型
    if not is_complete:
        print("\n开始下载缺失的模型文件...")
        if not download_bark_models():
            print("\n❌ 模型下载失败，请检查网络连接和磁盘空间")
            sys.exit(1)
    else:
        print("\n模型文件已存在，跳过下载")

    # 步骤3: 测试音频生成
    print("\n开始测试音频生成功能...")
    if not test_bark_generation():
        print("\n❌ 音频生成测试失败")
        sys.exit(1)

    # 成功
    print("\n" + "="*60)
    print("✅ Bark模型安装和测试完成！")
    print("="*60)
    print("\n现在可以使用音频水印模块了：")
    print("  python tests/test_audio_watermark.py")
    print("  或者")
    print("  from src.audio_watermark import create_audio_watermark")

if __name__ == "__main__":
    main()
