"""
WatermarkTool统一接口上传模式测试
测试图像、音频、视频三个模态的上传文件水印嵌入与提取
使用WatermarkTool的统一embed()和extract()接口
"""
# python3 tests/test_unified_upload_mode.py -v
import sys
import logging
import unittest
import shutil
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.unified.watermark_tool import WatermarkTool


class TestUnifiedUploadMode(unittest.TestCase):
    """WatermarkTool统一接口上传模式测试"""

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(__name__)

        try:
            cls.tool = WatermarkTool()
            cls.logger.info("✓ WatermarkTool 初始化成功")
        except Exception as e:
            cls.logger.error(f"❌ WatermarkTool 初始化失败: {e}")
            raise

        # 设置测试资源目录
        cls.test_results_dir = project_root / "tests" / "test_results"
        cls.output_dir = cls.test_results_dir / "unified_upload_mode"
        cls.output_dir.mkdir(parents=True, exist_ok=True)

        # 准备测试资源文件路径
        cls.test_image_path = cls.test_results_dir / "test_image_original.png"
        cls.test_audio_path = cls.test_results_dir / "test_audio_original.wav"
        cls.test_video_path = cls.test_results_dir / "test_wan_basic_output.mp4"

    def test_01_image_upload_watermark(self):
        """测试1: 图像上传模式水印嵌入与提取"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试1: 图像上传模式水印嵌入与提取")
        self.logger.info("=" * 60)

        try:
            watermarked_image = self.tool.embed(
                content="",  
                message="unified_test_image_2025",
                modality='image',
                image_input=str(self.test_image_path),
                output_path=str(self.output_dir / "image_watermarked.png")
            )

            self.assertIsNotNone(watermarked_image, "图像水印嵌入应返回有效内容")
            self.logger.info("✓ 图像水印嵌入成功")


            result = self.tool.extract(
                content=watermarked_image,
                modality='image'
            )

            self.assertIsNotNone(result, "提取应返回结果字典")
            self.assertIn('detected', result, "结果应包含'detected'字段")
            self.logger.info(f"检测结果: {result}")

            # 验证水印检测成功
            self.assertTrue(
                result['detected'],
                f"图像水印应被检测到 (结果: {result})"
            )
            self.logger.info("✓ 图像水印检测成功")

            # 验证置信度
            self.assertIn('confidence', result, "结果应包含'confidence'字段")
            confidence = result['confidence']
            self.assertGreaterEqual(
                confidence, 0.8,
                f"图像检测置信度应 >= 0.8 (实际: {confidence})"
            )
            self.logger.info(f"✓ 检测置信度: {confidence:.4f}")

            # 验证消息（如果存在）
            if 'message' in result and result['message']:
                self.assertIn(
                    "unified_test",
                    result['message'].lower(),
                    f"提取的消息应包含'unified_test' (实际: {result['message']})"
                )
                self.logger.info(f"✓ 提取的消息: {result['message']}")

        except Exception as e:
            import traceback
            self.logger.error(traceback.format_exc())
            self.fail(f"图像水印测试失败: {e}")

    def test_02_audio_upload_watermark(self):
        """测试2: 音频上传模式水印嵌入与提取"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试2: 音频上传模式水印嵌入与提取")
        self.logger.info("=" * 60)

        try:
            # 嵌入水印
            self.logger.info(f"正在嵌入音频水印: {self.test_audio_path}")
            watermarked_audio = self.tool.embed(
                content="",  # 占位符
                message="unified_test_audio_2025",
                modality='audio',
                audio_input=str(self.test_audio_path),
                output_path=str(self.output_dir / "audio_watermarked.wav")
            )

            # 验证嵌入结果
            self.assertIsNotNone(watermarked_audio, "音频水印嵌入应返回有效内容")
            self.logger.info("✓ 音频水印嵌入成功")

            # 验证输出文件存在
            output_audio_path = self.output_dir / "audio_watermarked.wav"
            self.assertTrue(
                output_audio_path.exists(),
                f"输出音频文件应存在: {output_audio_path}"
            )
            self.logger.info(f"✓ 输出音频文件已保存: {output_audio_path}")

            # 提取水印
            self.logger.info("正在提取音频水印...")
            result = self.tool.extract(
                content=str(output_audio_path),
                modality='audio'
            )

            # 验证提取结果
            self.assertIsNotNone(result, "提取应返回结果字典")
            self.assertIn('detected', result, "结果应包含'detected'字段")
            self.logger.info(f"检测结果: {result}")

            # 验证水印检测成功
            self.assertTrue(
                result['detected'],
                f"音频水印应被检测到 (结果: {result})"
            )
            self.logger.info("✓ 音频水印检测成功")

            # 验证置信度
            self.assertIn('confidence', result, "结果应包含'confidence'字段")
            confidence = result['confidence']
            self.assertGreaterEqual(
                confidence, 0.7,
                f"音频检测置信度应 >= 0.7 (实际: {confidence})"
            )
            self.logger.info(f"✓ 检测置信度: {confidence:.4f}")

            # 验证消息（如果存在）
            if 'message' in result and result['message']:
                self.logger.info(f"✓ 提取的消息: {result['message']}")

        except Exception as e:
            self.logger.error(f"❌ 音频水印测试失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.fail(f"音频水印测试失败: {e}")

    def test_03_video_upload_watermark(self):
        """测试3: 视频上传模式水印嵌入与提取"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试3: 视频上传模式水印嵌入与提取")
        self.logger.info("=" * 60)

        # 检查测试视频是否可用
        if self.test_video_path is None:
            self.skipTest("测试视频未创建成功，跳过视频测试")

        try:
            # 嵌入水印
            self.logger.info(f"正在嵌入视频水印: {self.test_video_path}")
            watermarked_video = self.tool.embed(
                content="",  # 占位符
                message="unified_test_video_2025",
                modality='video',
                video_input=str(self.test_video_path)
                # 注意：不指定 output_path，使用返回的路径
                # 因为视频会被转码为浏览器兼容格式，实际输出路径会不同
            )

            # 验证嵌入结果
            self.assertIsNotNone(watermarked_video, "视频水印嵌入应返回有效内容")
            self.logger.info(f"✓ 视频水印嵌入成功")

            # 视频水印返回的是实际保存的文件路径（字符串）
            if isinstance(watermarked_video, str):
                output_video_path = watermarked_video
            else:
                self.fail(f"视频水印应返回文件路径字符串，实际类型: {type(watermarked_video)}")

            # 验证输出文件存在
            self.assertTrue(
                Path(output_video_path).exists(),
                f"输出视频文件应存在: {output_video_path}"
            )
            self.logger.info(f"✓ 输出视频文件已保存: {output_video_path}")

            # 提取水印
            # 注意：不传递额外参数，让统一接口使用默认值
            self.logger.info("正在提取视频水印...")
            result = self.tool.extract(
                content=str(output_video_path),
                modality='video'
            )

            # 验证提取结果
            self.assertIsNotNone(result, "提取应返回结果字典")
            self.assertIn('detected', result, "结果应包含'detected'字段")
            self.logger.info(f"检测结果: {result}")

            # 验证水印检测成功
            self.assertTrue(
                result['detected'],
                f"视频水印应被检测到 (结果: {result})"
            )
            self.logger.info("✓ 视频水印检测成功")

            # 验证置信度
            self.assertIn('confidence', result, "结果应包含'confidence'字段")
            confidence = result['confidence']
            self.assertGreaterEqual(
                confidence, 0.7,
                f"视频检测置信度应 >= 0.7 (实际: {confidence})"
            )
            self.logger.info(f"✓ 检测置信度: {confidence:.4f}")

            # 验证消息（如果存在）
            if 'message' in result and result['message']:
                self.logger.info(f"✓ 提取的消息: {result['message']}")

        except Exception as e:
            self.logger.error(f"❌ 视频水印测试失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.fail(f"视频水印测试失败: {e}")

    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        cls.logger.info("\n" + "=" * 70)
        cls.logger.info("开始清理测试临时文件...")
        cls.logger.info("=" * 70)

        try:
            # 清理输出目录
            if cls.output_dir.exists():
                shutil.rmtree(cls.output_dir, ignore_errors=True)
                cls.logger.info(f"✓ 已删除输出目录: {cls.output_dir}")

            # 注意：不删除测试视频，因为它是预先上传的资源文件

        except Exception as e:
            cls.logger.warning(f"⚠ 清理过程中出现警告: {e}")

        cls.logger.info("=" * 70)
        cls.logger.info("统一接口上传模式测试完成，已清理临时文件")
        cls.logger.info("=" * 70)


if __name__ == '__main__':
    unittest.main(verbosity=2)
