"""
任务A：批量嵌入水印
遍历 training_dataset，对每个文件嵌入水印并保存原始文件副本
"""

import sys
from pathlib import Path
import shutil

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.unified.watermark_tool import WatermarkTool

# 配置
DATASET_DIR = Path('training_dataset')
RESULT_DIR = Path('class/result')
ORIGINAL_DIR = RESULT_DIR / 'original'
WATERMARKED_DIR = RESULT_DIR / 'watermarked'

# 水印消息
MESSAGES = {
    'image': 'Img_01',
    'video': 'Vid_01',
    'audio': 'Aud_01'
}

# 支持的文件格式
FORMATS = {
    'images': ['.jpg', '.jpeg', '.png'],
    'video': ['.mp4', '.avi'],
    'audio': ['.wav', '.mp3']
}

# 模态映射
MODALITY_MAP = {
    'images': 'image',
    'video': 'video',
    'audio': 'audio'
}


def create_directories():
    """创建输出目录"""
    for modality_dir in FORMATS.keys():
        (ORIGINAL_DIR / modality_dir).mkdir(parents=True, exist_ok=True)
        (WATERMARKED_DIR / modality_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ 输出目录已创建: {RESULT_DIR}")


def get_all_files(modality_dir, extensions):
    """获取所有文件（递归遍历子目录）"""
    input_dir = DATASET_DIR / modality_dir
    if not input_dir.exists():
        print(f"⚠️  目录不存在: {input_dir}")
        return []

    files = [f for f in input_dir.rglob('*') if f.is_file() and f.suffix.lower() in extensions]
    return sorted(files)


def copy_to_original(src_file, modality_dir):
    """复制到 original 目录（保持子目录结构）"""
    input_base = DATASET_DIR / modality_dir
    relative_path = src_file.relative_to(input_base)
    dst_file = ORIGINAL_DIR / modality_dir / relative_path
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)
    return relative_path


def embed_watermark(tool, src_file, message, modality, output_path):
    """嵌入水印（统一接口）"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if modality == 'image':
        # 图像水印
        result = tool.embed(str(src_file), message, 'image', image_input=str(src_file))
        if hasattr(result, 'save'):
            result.save(output_path)
            return True

    elif modality == 'video':
        # 视频水印
        result = tool.embed("placeholder", message, 'video', video_input=str(src_file))
        if result and isinstance(result, str) and Path(result).exists():
            if Path(result) != output_path:
                shutil.move(result, output_path)
            return True
        elif output_path.exists():
            return True

    elif modality == 'audio':
        # 音频水印
        tool.embed("placeholder", message, 'audio',
                  audio_input=str(src_file), output_path=str(output_path))
        return output_path.exists()

    return False


def process_file(tool, file_path, modality_dir, stats):
    """处理单个文件"""
    stats['total'] += 1
    modality = MODALITY_MAP[modality_dir]
    message = MESSAGES[modality]

    # 计算相对路径
    input_base = DATASET_DIR / modality_dir
    relative_path = file_path.relative_to(input_base)

    print(f"\n[{stats['total']}] {relative_path}")

    try:
        # 1. 复制原始文件
        copy_to_original(file_path, modality_dir)

        # 2. 嵌入水印
        output_path = WATERMARKED_DIR / modality_dir / relative_path
        success = embed_watermark(tool, file_path, message, modality, output_path)

        if success:
            print(f"    ✓ 成功")
            stats['success'] += 1
        else:
            print(f"    ✗ 失败")
            stats['failed'] += 1

    except Exception as e:
        print(f"    ✗ 错误: {e}")
        stats['failed'] += 1


def main():
    """主函数"""
    print("="*60)
    print("任务A：批量嵌入水印")
    print("="*60)

    # 初始化
    create_directories()
    tool = WatermarkTool()
    stats = {'total': 0, 'success': 0, 'failed': 0}

    # 处理所有模态
    for modality_dir, extensions in FORMATS.items():
        print(f"\n{'='*60}")
        print(f"处理 {modality_dir}")
        print(f"{'='*60}")

        files = get_all_files(modality_dir, extensions)
        if not files:
            print(f"未找到文件")
            continue

        print(f"找到 {len(files)} 个文件")

        for file_path in files:
            process_file(tool, file_path, modality_dir, stats)

    # 输出统计
    print(f"\n{'='*60}")
    print(f"完成统计")
    print(f"{'='*60}")
    print(f"总计: {stats['total']}")
    print(f"成功: {stats['success']}")
    print(f"失败: {stats['failed']}")
    if stats['total'] > 0:
        print(f"成功率: {stats['success']/stats['total']*100:.1f}%")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n任务被用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
