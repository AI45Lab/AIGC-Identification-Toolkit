"""
Bark TTS + AudioSeal 音频水印完整集成测试
测试Bark TTS音频生成与AudioSeal水印嵌入、提取的完整流程
"""
# python3 tests/test_bark_audioseal.py -v
import os
import sys
import logging
import unittest
from pathlib import Path
import warnings

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.audio_watermark.audio_watermark import AudioWatermark
from src.utils.path_manager import PathManager

# 检查Bark依赖
try:
    from src.audio_watermark.bark_generator import BarkGenerator
    HAS_BARK = True
except ImportError:
    HAS_BARK = False


class TestBarkAudioSealIntegration(unittest.TestCase):
    """Bark TTS + AudioSeal 完整集成测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(__name__)
        cls.logger.info("=" * 70)
        cls.logger.info("开始 Bark TTS + AudioSeal 完整集成测试")
        cls.logger.info("=" * 70)

        # 检测设备
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cls.logger.info(f"使用设备: {cls.device}")

        # 创建测试结果目录
        cls.test_results_dir = project_root / "tests" / "test_results"
        cls.test_results_dir.mkdir(parents=True, exist_ok=True)

        # 初始化路径管理器
        cls.path_manager = PathManager()
        cls.logger.info("✓ 路径管理器初始化成功")

        # 创建AudioWatermark实例
        cls.config = {
            'algorithm': 'audioseal',
            'model_name': 'audioseal_wm_16bits',
            'detector_name': 'audioseal_detector_16bits',
            'sample_rate': 16000,
            'message_bits': 16,
            'device': cls.device,
            'bark': {
                'model_size': 'large',
                'use_gpu': (cls.device == 'cuda'),
                'temperature': 0.8,
                'default_voice': 'v2/en_speaker_6'
            }
        }
        cls.audio_watermark = AudioWatermark()
        cls.audio_watermark.algorithm = 'audioseal'
        cls.audio_watermark.config.update(cls.config)
        cls.logger.info("✓ AudioWatermark实例创建成功 (algorithm=audioseal)")

        # 检查Bark是否可用
        if not HAS_BARK:
            cls.logger.warning("⚠️  Bark TTS未安装，部分测试将被跳过")
            cls.logger.warning("   安装命令: pip install git+https://github.com/suno-ai/bark.git")

    def test_01_path_manager_validation(self):
        """测试1: 路径管理器验证（无硬编码路径）"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试1: 路径管理器验证")
        self.logger.info("=" * 60)

        try:
            # 验证HF Hub缓存路径
            hf_hub_dir = self.path_manager.get_hf_hub_dir()
            self.assertTrue(hf_hub_dir.exists(), "HF Hub缓存目录应该存在")
            self.logger.info(f"✓ HF Hub缓存目录: {hf_hub_dir}")

            # 验证Bark缓存路径
            bark_cache = self.path_manager.get_bark_cache_dir()
            self.assertTrue(bark_cache.exists(), "Bark缓存目录应该存在")
            self.logger.info(f"✓ Bark缓存目录: {bark_cache}")

            # 验证缓存根目录
            cache_root = self.path_manager.get_cache_root()
            self.assertTrue(cache_root.exists(), "缓存根目录应该存在")
            self.logger.info(f"✓ 缓存根目录: {cache_root}")

            # 验证项目输出目录
            output_dir = self.path_manager.get_project_output_dir('outputs')
            self.assertTrue(output_dir.exists(), "项目输出目录应该存在")
            self.logger.info(f"✓ 项目输出目录: {output_dir}")

            # 检查是否有硬编码路径
            self.assertIsNotNone(hf_hub_dir, "HF Hub路径不应为None")
            self.assertNotIn('/home/', str(hf_hub_dir), "路径不应包含硬编码的/home/")
            self.logger.info("✓ 未检测到硬编码路径")

        except Exception as e:
            self.logger.error(f"❌ 路径管理器验证失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def test_02_audio_watermark_embed_generation(self):
        """测试2: 音频水印嵌入（Bark TTS生成 + AudioSeal水印）"""
        if not HAS_BARK:
            self.skipTest("Bark TTS未安装，跳过此测试")

        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试2: 音频水印嵌入（Bark TTS生成 + AudioSeal水印）")
        self.logger.info("=" * 60)

        try:
            # 生成音频并嵌入水印
            test_text = "Hello, this is a test of the audio watermarking system."
            test_message = "test_audio_2025"

            self.logger.info(f"文本内容: {test_text}")
            self.logger.info(f"水印消息: {test_message}")
            self.logger.info(f"采样率: {self.config['sample_rate']} Hz")

            # 使用generate_audio_with_watermark进行AI生成模式
            result = self.audio_watermark.generate_audio_with_watermark(
                prompt=test_text,
                message=test_message,
                voice_preset=self.config['bark']['default_voice'],
                return_original=True
            )

            # 验证输出
            if isinstance(result, dict):
                watermarked_audio = result['watermarked']
                original_audio = result.get('original')
            else:
                watermarked_audio = result
                original_audio = None

            self.assertIsInstance(watermarked_audio, (torch.Tensor, str),
                                "水印音频应该是torch.Tensor或文件路径")

            # 如果是Tensor，转换为numpy进行日志输出
            if isinstance(watermarked_audio, torch.Tensor):
                duration = watermarked_audio.shape[-1] / self.config['sample_rate']
                self.logger.info(f"✓ 水印音频生成成功: {watermarked_audio.shape}, 时长: {duration:.2f}秒")
            else:
                self.logger.info(f"✓ 水印音频生成成功: {watermarked_audio}")

            # 保存音频供后续测试使用
            original_path = str(self.test_results_dir / "test_audio_original.wav")
            watermarked_path = str(self.test_results_dir / "test_audio_watermarked.wav")

            # 保存音频文件
            if original_audio is not None:
                self.audio_watermark.save_audio(original_audio, original_path)
                self.logger.info(f"✓ 原始音频已保存: {original_path}")

            self.audio_watermark.save_audio(watermarked_audio, watermarked_path)
            self.logger.info(f"✓ 水印音频已保存: {watermarked_path}")

            # 保存路径供后续测试使用
            self.__class__.watermarked_audio = watermarked_audio
            self.__class__.watermarked_audio_path = watermarked_path
            self.__class__.test_message = test_message

        except Exception as e:
            self.logger.error(f"❌ 音频水印嵌入失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def test_03_audio_watermark_extract(self):
        """测试3: 音频水印提取与验证"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试3: 音频水印提取与验证")
        self.logger.info("=" * 60)

        try:
            # 如果测试2被跳过，使用简单的合成音频进行测试
            if not hasattr(self.__class__, 'watermarked_audio'):
                self.logger.info("测试2被跳过，使用合成音频进行测试")

                # 创建简单的测试音频（1秒，16kHz）
                test_audio = torch.randn(1, 16000).to(self.device)
                test_message = "test_audio_synthetic"

                # 嵌入水印
                watermarked_audio = self.audio_watermark.embed_watermark(
                    audio_input=test_audio,
                    message=test_message
                )

                self.__class__.watermarked_audio = watermarked_audio
                self.__class__.test_message = test_message
                self.logger.info("✓ 使用合成音频创建测试样本")

            watermarked_audio = self.__class__.watermarked_audio
            test_message = self.__class__.test_message

            self.logger.info("从音频提取水印...")

            # 提取水印
            result = self.audio_watermark.extract_watermark(watermarked_audio)

            # 验证结果
            self.logger.info("提取结果:")
            self.logger.info(f"  detected: {result['detected']}")
            self.logger.info(f"  message: {result.get('message', 'N/A')}")
            self.logger.info(f"  confidence: {result.get('confidence', 'N/A')}")

            self.assertTrue(result['detected'], "应该检测到水印")
            self.assertEqual(result['message'], test_message, "水印消息应该匹配")
            self.logger.info("✓ 音频水印提取验证通过")

        except Exception as e:
            self.logger.error(f"❌ 音频水印提取失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def test_04_upload_mode(self):
        """测试4: 上传模式测试（对已有音频嵌入水印）"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试4: 上传模式测试")
        self.logger.info("=" * 60)

        try:
            # 创建或读取测试音频
            original_path = str(self.test_results_dir / "test_audio_original.wav")

            # 如果没有原始音频，创建一个简单的测试音频
            if not Path(original_path).exists():
                test_audio = torch.randn(1, 16000).to(self.device)
                self.audio_watermark.save_audio(test_audio, original_path)
                self.logger.info(f"创建测试音频: {original_path}")

            upload_message = "upload_test_2025"
            self.logger.info(f"上传音频路径: {original_path}")
            self.logger.info(f"水印消息: {upload_message}")

            # 使用上传模式嵌入水印
            watermarked_audio = self.audio_watermark.embed_watermark(
                audio_input=original_path,
                message=upload_message
            )

            # 验证输出
            self.assertIsInstance(watermarked_audio, (torch.Tensor, str),
                                "水印音频应该是torch.Tensor或文件路径")

            if isinstance(watermarked_audio, torch.Tensor):
                duration = watermarked_audio.shape[-1] / self.config['sample_rate']
                self.logger.info(f"✓ 上传模式水印嵌入成功: {watermarked_audio.shape}, 时长: {duration:.2f}秒")
            else:
                self.logger.info(f"✓ 上传模式水印嵌入成功: {watermarked_audio}")

            # 保存水印音频
            upload_watermarked_path = str(self.test_results_dir / "test_audio_upload_watermarked.wav")
            self.audio_watermark.save_audio(watermarked_audio, upload_watermarked_path)
            self.logger.info(f"✓ 上传模式水印音频已保存: {upload_watermarked_path}")

            # 提取水印验证
            result = self.audio_watermark.extract_watermark(watermarked_audio)

            self.logger.info("上传模式提取结果:")
            self.logger.info(f"  detected: {result['detected']}")
            self.logger.info(f"  message: {result.get('message', 'N/A')}")
            self.logger.info(f"  confidence: {result.get('confidence', 'N/A')}")

            self.assertTrue(result['detected'], "应该检测到水印")
            self.assertEqual(result['message'], upload_message, "水印消息应该匹配")
            self.logger.info("✓ 上传模式水印验证通过")

        except Exception as e:
            self.logger.error(f"❌ 上传模式测试失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        cls.logger.info("\n" + "=" * 70)
        cls.logger.info("Bark TTS + AudioSeal 完整集成测试完成")
        cls.logger.info("=" * 70)

        # 清理管道以释放内存
        if hasattr(cls, 'audio_watermark'):
            try:
                if hasattr(cls.audio_watermark, 'clear_cache'):
                    cls.audio_watermark.clear_cache()
                cls.logger.info("✓ 缓存已清理")
            except Exception as e:
                cls.logger.warning(f"清理缓存时出错: {e}")

        # 显示生成的文件
        cls.logger.info("\n生成的文件:")
        test_results = cls.test_results_dir
        if test_results.exists():
            for audio_file in sorted(test_results.glob("test_audio_*.wav")):
                size = audio_file.stat().st_size / 1024  # KB
                cls.logger.info(f"  - {audio_file.name} ({size:.2f} KB)")


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
