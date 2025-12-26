"""
任务B：质量分析
计算原始文件与水印文件的质量指标，生成 CSV 报告
"""

import sys
from pathlib import Path
import csv
import importlib.util

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'benchmarks' / 'Image-Bench'))
sys.path.insert(0, str(project_root / 'benchmarks' / 'Video-Bench'))
sys.path.insert(0, str(project_root / 'benchmarks' / 'Audio-Bench'))

RESULT_DIR = Path('class/result')
ORIGINAL_DIR = RESULT_DIR / 'original'
WATERMARKED_DIR = RESULT_DIR / 'watermarked'
OUTPUT_FILE = Path('class/quality_report.csv')

CSV_COLUMNS = ['file_name', 'modality', 'psnr', 'ssim', 'snr', 'status']


def import_audio_snr():
    """导入音频 SNR 计算函数"""
    try:
        audio_path = project_root / 'benchmarks' / 'Audio-Bench' / 'metrics' / 'quality.py'
        spec = importlib.util.spec_from_file_location("audio_quality", audio_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✓ 音频质量指标模块已导入")
        return module.compute_snr
    except Exception as e:
        print(f"⚠️  音频质量指标导入失败: {e}")
        return None


def compute_image_quality(original_path, watermarked_path):
    """计算图像质量指标：PSNR, SSIM"""
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        from PIL import Image
        import numpy as np

        original = np.array(Image.open(original_path))
        watermarked = np.array(Image.open(watermarked_path))

        psnr_value = psnr(original, watermarked, data_range=255)

        if original.ndim == 3:  
            ssim_value = ssim(original, watermarked, channel_axis=2, data_range=255)
        else:  
            ssim_value = ssim(original, watermarked, data_range=255)

        return {'psnr': psnr_value, 'ssim': ssim_value}

    except Exception as e:
        print(f"    ✗ 计算失败: {e}")
        return {}


def compute_audio_quality(original_path, watermarked_path, compute_fn):
    """计算音频质量指标"""
    if compute_fn is None:
        return {}
    try:
        import torchaudio
        original_audio, _ = torchaudio.load(str(original_path))
        watermarked_audio, _ = torchaudio.load(str(watermarked_path))
        snr = compute_fn(signal=original_audio, noisy_signal=watermarked_audio)
        return {'snr': snr}
    except Exception as e:
        print(f"    ✗ 计算失败: {e}")
        return {}


def compute_video_quality(original_path, watermarked_path):
    """计算视频质量指标：逐帧计算 PSNR 和 SSIM，然后求平均"""
    try:
        import cv2
        import numpy as np
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim

        # 打开视频
        cap_orig = cv2.VideoCapture(str(original_path))
        cap_wm = cv2.VideoCapture(str(watermarked_path))

        psnr_values = []
        ssim_values = []

        while True:
            ret_orig, frame_orig = cap_orig.read()
            ret_wm, frame_wm = cap_wm.read()
            if not ret_orig or not ret_wm:
                break

            frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
            frame_wm = cv2.cvtColor(frame_wm, cv2.COLOR_BGR2RGB)

            psnr_value = psnr(frame_orig, frame_wm, data_range=255)
            psnr_values.append(psnr_value)

            ssim_value = ssim(frame_orig, frame_wm, channel_axis=2, data_range=255)
            ssim_values.append(ssim_value)

        cap_orig.release()
        cap_wm.release()

        if not psnr_values:
            return {}
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)

        return {'psnr': avg_psnr, 'ssim': avg_ssim}

    except Exception as e:
        print(f"    ✗ 计算失败: {e}")
        return {}


def process_file_pair(original_file, watermarked_file, modality, audio_snr_fn, stats):
    """处理单对文件"""
    stats['total'] += 1

    original_base = ORIGINAL_DIR / modality
    relative_path = original_file.relative_to(original_base)

    print(f"[{stats['total']}] {relative_path}")

    result = {
        'file_name': str(relative_path),
        'modality': modality,
        'psnr': None,
        'ssim': None,
        'snr': None,
        'status': 'pending'
    }

    try:
        if modality == 'images':
            metrics = compute_image_quality(original_file, watermarked_file)
            result.update({
                'psnr': metrics.get('psnr'),
                'ssim': metrics.get('ssim')
            })
        elif modality == 'video':
            metrics = compute_video_quality(original_file, watermarked_file)
            result.update({
                'psnr': metrics.get('psnr'),
                'ssim': metrics.get('ssim')
            })
        elif modality == 'audio':
            metrics = compute_audio_quality(original_file, watermarked_file, audio_snr_fn)
            result['snr'] = metrics.get('snr')

        result['status'] = 'success'
        stats['success'] += 1
        print(f"    ✓ 成功")

    except Exception as e:
        result['status'] = f'failed: {e}'
        stats['failed'] += 1
        print(f"    ✗ 错误: {e}")

    return result


def process_modality(modality, audio_snr_fn, stats):
    """处理特定模态的所有文件"""

    original_dir = ORIGINAL_DIR / modality
    watermarked_dir = WATERMARKED_DIR / modality

    if not original_dir.exists() or not watermarked_dir.exists():
        return []

    original_files = sorted([f for f in original_dir.rglob('*') if f.is_file()])

    results = []
    for original_file in original_files:
        relative_path = original_file.relative_to(original_dir)
        watermarked_file = watermarked_dir / relative_path

        if not watermarked_file.exists():
            print(f"⚠️  水印文件不存在: {relative_path}")
            continue

        result = process_file_pair(original_file, watermarked_file, modality, audio_snr_fn, stats)
        results.append(result)

    return results


def write_csv(results):
    """写入 CSV 报告"""
    print(f"\n正在生成 CSV 报告...")
    try:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(results)
        print(f"✓ CSV 报告已生成: {OUTPUT_FILE.resolve()}")
        print(f"  共 {len(results)} 条记录")
    except Exception as e:
        print(f"✗ 写入 CSV 失败: {e}")


def main():
    """主函数"""
    audio_snr_fn = import_audio_snr()
    stats = {'total': 0, 'success': 0, 'failed': 0}
    all_results = []

    for modality in ['images', 'video', 'audio']:
        results = process_modality(modality, audio_snr_fn, stats)
        all_results.extend(results)

    write_csv(all_results)

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
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
