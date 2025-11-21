"""
Stable Diffusion + VideoSeal 图像水印完整集成测试
测试SD图像生成与VideoSeal水印嵌入、提取的完整流程
"""
# python3 tests/test_sd_videoseal.py -v
import os
import sys
import logging
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image
from src.image_watermark.image_watermark import ImageWatermark
from src.utils.path_manager import PathManager


class TestSDVideoSealIntegration(unittest.TestCase):
    """Stable Diffusion + VideoSeal 完整集成测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(__name__)
        cls.logger.info("=" * 70)
        cls.logger.info("开始 Stable Diffusion + VideoSeal 完整集成测试")
        cls.logger.info("=" * 70)

        # 创建测试结果目录
        cls.test_results_dir = project_root / "tests" / "test_results"
        cls.test_results_dir.mkdir(parents=True, exist_ok=True)

        # 初始化路径管理器
        cls.path_manager = PathManager()
        cls.logger.info("✓ 路径管理器初始化成功")

        # 创建ImageWatermark实例（使用videoseal算法）
        cls.config = {
            'algorithm': 'videoseal',
            'model_name': 'stabilityai/stable-diffusion-2-1-base',
            'resolution': 512,
            'num_inference_steps': 30,
            'guidance_scale': 7.5,
            'lowres_attenuation': True,
            'device': 'cuda'
        }
        cls.image_watermark = ImageWatermark()
        cls.image_watermark.algorithm = 'videoseal'
        cls.image_watermark.config.update(cls.config)
        cls.logger.info("✓ ImageWatermark实例创建成功 (algorithm=videoseal)")

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

            # 验证HF Home路径
            hf_home = self.path_manager.get_hf_home()
            self.assertTrue(hf_home.exists(), "HF Home目录应该存在")
            self.logger.info(f"✓ HF Home目录: {hf_home}")

            # 验证缓存根目录
            cache_root = self.path_manager.get_cache_root()
            self.assertTrue(cache_root.exists(), "缓存根目录应该存在")
            self.logger.info(f"✓ 缓存根目录: {cache_root}")

            # 验证项目输出目录
            output_dir = self.path_manager.get_project_output_dir('outputs')
            self.assertTrue(output_dir.exists(), "项目输出目录应该存在")
            self.logger.info(f"✓ 项目输出目录: {output_dir}")

            # 检查是否有硬编码路径（确保所有路径都是从PathManager获取的）
            self.assertIsNotNone(hf_hub_dir, "HF Hub路径不应为None")
            self.assertNotIn('/home/', str(hf_hub_dir), "路径不应包含硬编码的/home/")
            self.logger.info("✓ 未检测到硬编码路径")

        except Exception as e:
            self.logger.error(f"❌ 路径管理器验证失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def test_02_image_watermark_embed_generation(self):
        """测试2: 图像水印嵌入（SD生成 + VideoSeal水印）"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试2: 图像水印嵌入（SD生成 + VideoSeal水印）")
        self.logger.info("=" * 60)

        try:
            # 生成图像并嵌入水印
            test_prompt = "A beautiful cat sitting in a garden"
            test_message = "test_image_2025"

            self.logger.info(f"提示词: {test_prompt}")
            self.logger.info(f"水印消息: {test_message}")
            self.logger.info(f"参数: {self.config['resolution']}分辨率, {self.config['num_inference_steps']}推理步数")

            # 使用embed_watermark进行AI生成模式
            result = self.image_watermark.embed_watermark(
                image_input=None,  # None表示生成模式
                prompt=test_prompt,
                message=test_message,
                return_original=True,
                resolution=self.config['resolution'],
                num_inference_steps=self.config['num_inference_steps'],
                guidance_scale=self.config['guidance_scale'],
                seed=42
            )

            # 验证输出
            if isinstance(result, dict):
                watermarked_image = result['watermarked']
                original_image = result.get('original')
            else:
                watermarked_image = result
                original_image = None

            self.assertIsInstance(watermarked_image, Image.Image, "水印图像应该是PIL.Image对象")
            self.logger.info(f"✓ 水印图像生成成功: {watermarked_image.size}")

            # 保存图像供后续测试使用
            original_path = str(self.test_results_dir / "test_image_original.png")
            watermarked_path = str(self.test_results_dir / "test_image_watermarked.png")

            if original_image:
                original_image.save(original_path)
                self.logger.info(f"✓ 原始图像已保存: {original_path}")

            watermarked_image.save(watermarked_path)
            self.logger.info(f"✓ 水印图像已保存: {watermarked_path}")

            # 保存路径供后续测试使用
            self.__class__.watermarked_image = watermarked_image
            self.__class__.watermarked_image_path = watermarked_path
            self.__class__.test_message = test_message

        except Exception as e:
            self.logger.error(f"❌ 图像水印嵌入失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def test_03_image_watermark_extract(self):
        """测试3: 图像水印提取与验证"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试3: 图像水印提取与验证")
        self.logger.info("=" * 60)

        try:
            # 确保测试2已运行
            if not hasattr(self.__class__, 'watermarked_image'):
                self.skipTest("需要先运行测试2")

            watermarked_image = self.__class__.watermarked_image
            test_message = self.__class__.test_message

            self.logger.info("从图像提取水印...")

            # 提取水印
            result = self.image_watermark.extract_watermark(
                watermarked_image,
                replicate=16,  # 增强检测
                chunk_size=16
            )

            # 验证结果
            self.logger.info("提取结果:")
            self.logger.info(f"  detected: {result['detected']}")
            self.logger.info(f"  message: {result.get('message', 'N/A')}")
            self.logger.info(f"  confidence: {result.get('confidence', 'N/A')}")

            self.assertTrue(result['detected'], "应该检测到水印")
            self.assertEqual(result['message'], test_message, "水印消息应该匹配")
            self.logger.info("✓ 图像水印提取验证通过")

        except Exception as e:
            self.logger.error(f"❌ 图像水印提取失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def test_04_upload_mode(self):
        """测试4: 上传模式测试（对已有图像嵌入水印）"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试4: 上传模式测试")
        self.logger.info("=" * 60)

        try:
            # 确保测试2已运行，使用原始图像作为上传的图像
            if not hasattr(self.__class__, 'watermarked_image_path'):
                self.skipTest("需要先运行测试2")

            # 读取之前保存的原始图像（如果存在）
            original_path = str(self.test_results_dir / "test_image_original.png")
            if not Path(original_path).exists():
                # 如果没有原始图像，创建一个简单的测试图像
                test_image = Image.new('RGB', (512, 512), color='blue')
                test_image.save(original_path)
                self.logger.info(f"创建测试图像: {original_path}")

            upload_message = "upload_test_2025"
            self.logger.info(f"上传图像路径: {original_path}")
            self.logger.info(f"水印消息: {upload_message}")

            # 使用上传模式嵌入水印
            watermarked_image = self.image_watermark.embed_watermark(
                image_input=original_path,
                message=upload_message,
                return_original=False
            )

            # 验证输出
            self.assertIsInstance(watermarked_image, Image.Image, "水印图像应该是PIL.Image对象")
            self.logger.info(f"✓ 上传模式水印嵌入成功: {watermarked_image.size}")

            # 保存水印图像
            upload_watermarked_path = str(self.test_results_dir / "test_image_upload_watermarked.png")
            watermarked_image.save(upload_watermarked_path)
            self.logger.info(f"✓ 上传模式水印图像已保存: {upload_watermarked_path}")

            # 提取水印验证
            result = self.image_watermark.extract_watermark(
                watermarked_image,
                replicate=16,
                chunk_size=16
            )

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
        cls.logger.info("Stable Diffusion + VideoSeal 完整集成测试完成")
        cls.logger.info("=" * 70)

        # 清理管道以释放内存
        if hasattr(cls, 'image_watermark'):
            try:
                if hasattr(cls.image_watermark, 'clear_cache'):
                    cls.image_watermark.clear_cache()
                cls.logger.info("✓ 缓存已清理")
            except Exception as e:
                cls.logger.warning(f"清理缓存时出错: {e}")

        # 显示生成的文件
        cls.logger.info("\n生成的文件:")
        test_results = cls.test_results_dir
        if test_results.exists():
            for image_file in sorted(test_results.glob("test_image_*.png")):
                size = image_file.stat().st_size / 1024  # KB
                cls.logger.info(f"  - {image_file.name} ({size:.2f} KB)")


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
