"""
Wan2.1 + VideoSeal 水印完整集成测试
测试Wan2.1视频生成与VideoSeal水印嵌入、提取的完整流程
"""
# python3 tests/test_wan_videoseal_integration.py -v
import os
import sys
import logging
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.video_watermark.video_watermark import VideoWatermark


class TestWanVideoSealIntegration(unittest.TestCase):
    """Wan2.1 + VideoSeal 完整集成测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(__name__)
        cls.logger.info("=" * 70)
        cls.logger.info("开始 Wan2.1 + VideoSeal 完整集成测试")
        cls.logger.info("=" * 70)

        # 创建测试结果目录
        cls.test_results_dir = project_root / "tests" / "test_results"
        cls.test_results_dir.mkdir(parents=True, exist_ok=True)

        # 创建VideoWatermark实例
        cls.video_watermark = VideoWatermark()
        cls.logger.info("✓ VideoWatermark实例创建成功")

    def test_01_basic_watermark_embedding(self):
        """测试1: 基础水印嵌入（文生视频 + 水印）"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试1: 基础水印嵌入（文生视频 + 水印）")
        self.logger.info("=" * 60)

        try:
            # 生成视频并嵌入水印
            test_prompt = "A cat walks on the grass, realistic"
            test_message = "integration_test_2025"

            self.logger.info(f"提示词: {test_prompt}")
            self.logger.info(f"水印消息: {test_message}")
            self.logger.info("参数: 81帧, 480x832分辨率, 50推理步数")

            output_path = str(self.test_results_dir / "test_integration_basic.mp4")

            watermarked_path = self.video_watermark.generate_video_with_watermark(
                prompt=test_prompt,
                message=test_message,
                num_frames=81,
                height=480,
                width=832,
                num_inference_steps=50,
                guidance_scale=5.0,
                seed=42,
                output_path=output_path,
                return_original=False
            )

            # 验证输出
            self.assertTrue(Path(watermarked_path).exists(), "水印视频文件应该存在")
            file_size = Path(watermarked_path).stat().st_size / (1024 * 1024)  # MB
            self.logger.info(f"✓ 水印视频已保存: {watermarked_path}")
            self.logger.info(f"✓ 文件大小: {file_size:.2f} MB")
            self.assertGreater(file_size, 0.5, "文件大小应该 > 0.5 MB")

            # 保存路径供后续测试使用
            self.__class__.watermarked_video_path = watermarked_path

        except Exception as e:
            self.logger.error(f"❌ 基础水印嵌入失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def test_02_watermark_extraction(self):
        """测试2: 水印提取与验证"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试2: 水印提取与验证")
        self.logger.info("=" * 60)

        try:
            # 确保测试1已运行
            if not hasattr(self.__class__, 'watermarked_video_path'):
                self.skipTest("需要先运行测试1")

            watermarked_path = self.__class__.watermarked_video_path
            self.logger.info(f"从视频提取水印: {watermarked_path}")

            # 提取水印
            result = self.video_watermark.extract_watermark(watermarked_path)

            # 验证结果
            self.logger.info("提取结果:")
            self.logger.info(f"  detected: {result['detected']}")
            self.logger.info(f"  message: {result['message']}")
            self.logger.info(f"  confidence: {result.get('confidence', 'N/A')}")

            self.assertTrue(result['detected'], "应该检测到水印")
            self.assertEqual(result['message'], "integration_test_2025", "水印消息应该匹配")
            self.logger.info("✓ 水印提取验证通过")

        except Exception as e:
            self.logger.error(f"❌ 水印提取失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise


    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        cls.logger.info("\n" + "=" * 70)
        cls.logger.info("Wan2.1 + VideoSeal 完整集成测试完成")
        cls.logger.info("=" * 70)

        # 清理管道以释放内存
        if hasattr(cls, 'video_watermark'):
            try:
                cls.video_watermark.clear_cache()
                cls.logger.info("✓ 缓存已清理")
            except Exception as e:
                cls.logger.warning(f"清理缓存时出错: {e}")

        # 显示生成的文件
        cls.logger.info("\n生成的文件:")
        test_results = cls.test_results_dir
        if test_results.exists():
            for video_file in sorted(test_results.glob("test_integration_*.mp4")):
                size = video_file.stat().st_size / (1024 * 1024)
                cls.logger.info(f"  - {video_file.name} ({size:.2f} MB)")


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
