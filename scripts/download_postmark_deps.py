#!/usr/bin/env python3
"""
PostMark HuggingFace模型下载脚本
自动下载所有需要的模型（利用HuggingFace默认缓存机制）
支持使用国内镜像站点加速下载
"""

import os
import sys
from pathlib import Path

# 设置HuggingFace镜像（国内加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print("✅ 已设置HuggingFace镜像: https://hf-mirror.com")

# HuggingFace默认缓存目录（无需显式指定，自动使用）
CACHE_DIR = Path.home() / ".cache/huggingface/hub"
print(f"✅ 模型将自动下载到HuggingFace默认缓存目录")
print(f"   预期路径: {CACHE_DIR}")

print("=" * 60)
print("PostMark HuggingFace模型下载脚本")
print("=" * 60)
print(f"缓存目录: {CACHE_DIR}")
print("=" * 60)

def check_gpu():
    """检查GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠️  未检测到GPU，将使用CPU（下载速度较慢）")
            return False
    except ImportError:
        print("❌ PyTorch未安装，请先安装: pip install torch")
        sys.exit(1)

def download_bert_tokenizer():
    """下载BERT tokenizer（Nomic依赖）"""
    print("\n[1/3] 下载 BERT Tokenizer...")
    print("-" * 60)
    print("说明: NomicEmbed需要bert-base-uncased tokenizer")
    try:
        from transformers import AutoTokenizer
        # 不指定cache_dir，使用HuggingFace默认路径
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        print("✅ bert-base-uncased tokenizer 下载成功")
        model_path = CACHE_DIR / "models--bert-base-uncased"
        if model_path.exists():
            print(f"   存储路径: {model_path}")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def download_nomic_embedder():
    """下载Nomic embedding模型"""
    print("\n[2/3] 下载 Nomic Embedding 模型...")
    print("-" * 60)
    print("模型: nomic-ai/nomic-embed-text-v1")
    print("大小: ~550MB")
    print("用途: 生成水印词嵌入")

    try:
        from transformers import AutoModel
        print("开始下载... (可能需要几分钟)")
        # 不指定cache_dir，使用HuggingFace默认路径
        model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1",
            trust_remote_code=True
        )
        print("✅ nomic-ai/nomic-embed-text-v1 下载成功")
        model_path = CACHE_DIR / "models--nomic-ai--nomic-embed-text-v1"
        if model_path.exists():
            print(f"   存储路径: {model_path}")

        # 释放内存
        del model
        import gc
        gc.collect()
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def download_llama3_inserter():
    """下载Llama-3-8B-Instruct模型"""
    print("\n[3/3] 下载 Llama-3-8B-Instruct 模型...")
    print("-" * 60)
    print("模型: meta-llama/Meta-Llama-3-8B-Instruct")
    print("大小: ~16GB")
    print("用途: 水印词插入")
    print("⚠️  注意: 需要在HuggingFace接受Llama 3使用协议")
    print("   访问: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

        print("\n下载 tokenizer...")
        # 不指定cache_dir，使用HuggingFace默认路径
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer 下载成功")

        print("\n下载 模型... (这可能需要10-30分钟，取决于网速)")
        print("提示: 如果下载中断，重新运行此脚本会自动续传")

        # 不指定cache_dir，使用HuggingFace默认路径
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=None,  # 不加载到GPU，仅下载
            low_cpu_mem_usage=True
        )
        print("✅ meta-llama/Meta-Llama-3-8B-Instruct 下载成功")
        model_path = CACHE_DIR / "models--meta-llama--Meta-Llama-3-8B-Instruct"
        if model_path.exists():
            print(f"   存储路径: {model_path}")

        # 释放内存
        del model, tokenizer
        import gc
        gc.collect()
        return True

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "gated" in error_msg.lower():
            print("\n❌ 权限错误: 无法访问Llama-3模型")
            print("\n解决方案:")
            print("1. 访问: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
            print("2. 点击 'Access repository' 接受使用协议")
            print("3. 设置HuggingFace token:")
            print("   a) 访问: https://huggingface.co/settings/tokens")
            print("   b) 创建或复制token")
            print("   c) 运行: huggingface-cli login")
            print("   d) 粘贴token")
            print("\n或者使用替代模型 Mistral-7B-Instruct (见下方提示)")
        else:
            print(f"❌ 下载失败: {e}")
        return False

def download_mistral_alternative():
    """下载Mistral-7B-Instruct作为替代方案"""
    print("\n[替代方案] 下载 Mistral-7B-Instruct 模型...")
    print("-" * 60)
    print("模型: mistralai/Mistral-7B-Instruct-v0.2")
    print("大小: ~14GB")
    print("说明: 如果无法获取Llama-3访问权限，可使用此模型")

    response = input("\n是否下载Mistral-7B-Instruct? (y/n): ").strip().lower()
    if response != 'y':
        print("跳过Mistral下载")
        return False

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "mistralai/Mistral-7B-Instruct-v0.2"

        print("\n下载 tokenizer...")
        # 不指定cache_dir，使用HuggingFace默认路径
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer 下载成功")

        print("\n下载 模型...")
        # 不指定cache_dir，使用HuggingFace默认路径
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=None,
            low_cpu_mem_usage=True
        )
        print("✅ mistralai/Mistral-7B-Instruct-v0.2 下载成功")
        model_path = CACHE_DIR / "models--mistralai--Mistral-7B-Instruct-v0.2"
        if model_path.exists():
            print(f"   存储路径: {model_path}")

        del model, tokenizer
        import gc
        gc.collect()
        return True

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def verify_downloads():
    """验证所有模型是否下载成功"""
    print("\n" + "=" * 60)
    print("验证下载结果")
    print("=" * 60)

    models_to_check = {
        "bert-base-uncased": "models--bert-base-uncased",
        "nomic-ai/nomic-embed-text-v1": "models--nomic-ai--nomic-embed-text-v1",
        "meta-llama/Meta-Llama-3-8B-Instruct": "models--meta-llama--Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2": "models--mistralai--Mistral-7B-Instruct-v0.2",
    }

    results = {}
    for model_name, cache_name in models_to_check.items():
        model_path = Path(CACHE_DIR) / cache_name
        if model_path.exists():
            # 计算目录大小
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            size_gb = total_size / (1024**3)
            print(f"✅ {model_name}")
            print(f"   路径: {model_path}")
            print(f"   大小: {size_gb:.2f} GB")
            results[model_name] = True
        else:
            print(f"❌ {model_name} - 未找到")
            results[model_name] = False

    return results

def main():
    """主函数"""
    print("\n开始下载流程...\n")

    # 检查GPU
    has_gpu = check_gpu()

    # 检查必需的Python包
    print("\n检查Python依赖...")
    try:
        import torch
        import transformers
        print(f"✅ torch {torch.__version__}")
        print(f"✅ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请先安装: pip install torch transformers")
        sys.exit(1)

    # 下载模型
    success = []

    # 1. BERT tokenizer
    if download_bert_tokenizer():
        success.append("BERT")

    # 2. Nomic embedder
    if download_nomic_embedder():
        success.append("Nomic")

    # 3. Llama-3 inserter
    llama_success = download_llama3_inserter()
    if llama_success:
        success.append("Llama-3")
    else:
        # 如果Llama-3失败，提供Mistral替代方案
        if download_mistral_alternative():
            success.append("Mistral")

    # 验证下载
    print("\n" + "=" * 60)
    results = verify_downloads()

    # 总结
    print("\n" + "=" * 60)
    print("下载总结")
    print("=" * 60)
    print(f"成功下载: {len(success)} 个模型")
    for model in success:
        print(f"  ✅ {model}")

    # 检查是否有inserter模型
    has_inserter = results.get("meta-llama/Meta-Llama-3-8B-Instruct", False) or \
                   results.get("mistralai/Mistral-7B-Instruct-v0.2", False)

    if results.get("bert-base-uncased") and \
       results.get("nomic-ai/nomic-embed-text-v1") and \
       has_inserter:
        print("\n🎉 所有必需模型下载完成！")
        print(f"\n模型存储位置: {CACHE_DIR}")
        print("\n下一步:")
        print("1. 从Google Drive下载 paragram_xxl.pkl 和 nomic_embs.pkl")
        print("2. 运行PostMark封装代码")
    else:
        print("\n⚠️  部分模型下载失败，请检查错误信息")
        if not has_inserter:
            print("\n注意: 水印插入器模型未成功下载")
            print("      请解决Llama-3访问权限问题或下载Mistral替代")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断下载")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    # Manual download instructions
    print("\n" + "=" * 70)
    print("[2/2] Manual Download Required - PKL Files")
    print("=" * 70)
    print("\nPlease download the following files from Google Drive:\n")
    print("📦 Required files:")
    print("  1. paragram_xxl.pkl (~1-2GB)")
    print("  2. filtered_data_100k_unique_250w_sentbound_nomic_embs.pkl (~1GB)\n")
    print("🔗 Google Drive link:")
    print("  https://drive.google.com/drive/folders/1Rdpqbtvy2s91ZrcgqDy6CrTCb9dZBQBf\n")
    print("📁 Place downloaded files in:")
    print(f"  {os.path.abspath('src/text_watermark/PostMark')}")
    print("\n" + "=" * 70)