"""
任务C：批量提取验证
对所有水印文件提取水印，验证消息是否与原始消息一致
"""

import sys
from pathlib import Path
import csv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.unified.watermark_tool import WatermarkTool

WATERMARKED_DIR = Path('class/result/watermarked')
OUTPUT_FILE = Path('class/extraction_report.csv')


EXPECTED_MESSAGES = {
    'images': 'Img_01',
    'video': 'Vid_01',
    'audio': 'Aud_01'
}


MODALITY_MAP = {
    'images': 'image',
    'video': 'video',
    'audio': 'audio'
}


CSV_COLUMNS = [
    'file_name',
    'modality',
    'detected',
    'original_message',
    'extracted_message',
    'match',
    'confidence'
]


def get_all_files(modality_dir):
    """获取所有文件（递归遍历子目录）"""
    modality_path = WATERMARKED_DIR / modality_dir
    if not modality_path.exists():
        return []

    files = [f for f in modality_path.rglob('*') if f.is_file()]
    return sorted(files)


def extract_watermark(tool, file_path, modality):
    """提取水印"""
    try:
        result = tool.extract(str(file_path), modality)
        return result
    except Exception as e:
        return {
            'detected': False,
            'message': '',
            'confidence': 0.0,
            'error': str(e)
        }


def process_file(tool, file_path, modality_dir, stats):
    """处理单个文件"""
    stats['total'] += 1
    modality = MODALITY_MAP[modality_dir]
    expected_msg = EXPECTED_MESSAGES[modality_dir]

    modality_base = WATERMARKED_DIR / modality_dir
    relative_path = file_path.relative_to(modality_base)

    print(f"[{stats['total']}] {relative_path}")

    result = extract_watermark(tool, file_path, modality)

    detected = result.get('detected', False)
    extracted_msg = result.get('message', '')
    confidence = result.get('confidence', 0.0)
    match = (extracted_msg == expected_msg)

    if detected:
        stats['detected'] += 1
    if match:
        stats['matched'] += 1

    if match:
        print(f"    ✓ 匹配 | 置信度: {confidence:.3f}")
    elif detected:
        print(f"    ⚠️  不匹配 | 期望: '{expected_msg}' | 实际: '{extracted_msg}'")
    else:
        print(f"    ✗ 未检测到水印")

    return {
        'file_name': str(relative_path),
        'modality': modality_dir,
        'detected': detected,
        'original_message': expected_msg,
        'extracted_message': extracted_msg,
        'match': match,
        'confidence': confidence
    }


def write_csv(results):
    """写入 CSV 报告"""
    print(f"\n生成 CSV 报告: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)
    print(f"✓ 已保存 {len(results)} 条记录")


def main():
    """主函数"""

    tool = WatermarkTool()
    stats = {'total': 0, 'detected': 0, 'matched': 0}
    all_results = []

    for modality_dir in ['images', 'video', 'audio']:

        files = get_all_files(modality_dir)
        if not files:
            print(f"未找到文件")
            continue

        for file_path in files:
            result = process_file(tool, file_path, modality_dir, stats)
            all_results.append(result)

    # 写入 CSV
    write_csv(all_results)

    # 输出统计
    print(f"\n{'='*60}")
    print(f"完成统计")
    print(f"{'='*60}")
    print(f"总计: {stats['total']}")
    print(f"检测到: {stats['detected']}")
    print(f"匹配: {stats['matched']}")
    if stats['total'] > 0:
        print(f"检测率: {stats['detected']/stats['total']*100:.1f}%")
        print(f"匹配率: {stats['matched']/stats['total']*100:.1f}%")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
