"""
Mistral-7B + PostMark 文本水印完整集成测试
测试PostMark后处理水印的嵌入和提取完整流程
"""
# python3 tests/test_mistral_postmark.py -v
import os
import sys
import logging
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.text_watermark.text_watermark import TextWatermark
from src.text_watermark.postmark_watermark import PostMarkWatermark
from src.utils.path_manager import PathManager


class TestMistralPostMarkIntegration(unittest.TestCase):
    """Mistral-7B + PostMark 完整集成测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(__name__)
        cls.logger.info("=" * 70)
        cls.logger.info("开始 Mistral-7B + PostMark 完整集成测试")
        cls.logger.info("=" * 70)

        # 创建测试结果目录
        cls.test_results_dir = project_root / "tests" / "test_results"
        cls.test_results_dir.mkdir(parents=True, exist_ok=True)

        # 初始化路径管理器
        cls.path_manager = PathManager()
        cls.logger.info("✓ 路径管理器初始化成功")

        # 创建PostMark配置
        cls.config = {
            'embedder': 'nomic',
            'inserter': 'mistral-7b-inst',
            'ratio': 0.12,
            'iterate': 'v2',
            'threshold': 0.7
        }

        # 创建PostMarkWatermark实例
        cls.text_watermark = PostMarkWatermark(cls.config)
        cls.logger.info("✓ PostMarkWatermark实例创建成功")

        # 测试文本（模拟LLM生成的文本）
        cls.sample_text = """
        Artificial intelligence has revolutionized many aspects of modern life.
        Machine learning algorithms can now process vast amounts of data to identify
        patterns and make predictions with remarkable accuracy. Deep learning techniques
        have enabled significant breakthroughs in computer vision, natural language processing,
        and speech recognition. These advancements are transforming industries ranging from
        healthcare and finance to transportation and entertainment.
        """

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

            # 验证Transformers缓存路径
            transformers_cache = self.path_manager.get_transformers_cache()
            self.assertTrue(transformers_cache.exists(), "Transformers缓存目录应该存在")
            self.logger.info(f"✓ Transformers缓存目录: {transformers_cache}")

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

            # 验证PostMark模型路径
            mistral_model = self.path_manager.find_model_in_hub("mistralai/Mistral-7B-Instruct-v0.2")
            if mistral_model:
                self.logger.info(f"✓ 找到Mistral模型: {mistral_model}")
            else:
                self.logger.warning("⚠️  Mistral模型未在缓存中找到，嵌入测试可能需要下载")

            nomic_model = self.path_manager.find_model_in_hub("nomic-ai/nomic-embed-text-v1")
            if nomic_model:
                self.logger.info(f"✓ 找到Nomic模型: {nomic_model}")
            else:
                self.logger.warning("⚠️  Nomic模型未在缓存中找到，嵌入测试可能需要下载")

        except Exception as e:
            self.logger.error(f"❌ 路径管理器验证失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def test_02_text_watermark_embed(self):
        """测试2: 文本水印嵌入（PostMark后处理）"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试2: 文本水印嵌入（PostMark后处理）")
        self.logger.info("=" * 60)

        try:
            # 准备测试数据
            test_text = self.sample_text.strip()
            test_message = "test_text_2025"

            self.logger.info(f"原始文本 ({len(test_text.split())} 词):")
            self.logger.info(f"  {test_text[:150]}...")
            self.logger.info(f"水印消息: {test_message}")
            self.logger.info(f"配置: embedder={self.config['embedder']}, inserter={self.config['inserter']}, ratio={self.config['ratio']}")

            # 嵌入水印
            self.logger.info("\n⏳ 正在嵌入PostMark水印...")
            result = self.text_watermark.embed(test_text, message=test_message)

            # 验证结果
            self.assertTrue(result['success'], "水印嵌入应该成功")
            self.assertIn('watermarked_text', result, "结果应包含watermarked_text字段")
            self.assertIn('watermark_words', result, "结果应包含watermark_words字段")

            watermarked_text = result['watermarked_text']
            watermark_words = result['watermark_words']

            self.logger.info("✓ 水印嵌入成功")
            self.logger.info(f"  - 水印词数量: {len(watermark_words)}")
            self.logger.info(f"  - 嵌入模型: {result['metadata']['inserter']}")
            self.logger.info(f"  - 水印文本 (前150字符):")
            self.logger.info(f"    {watermarked_text[:150]}...")

            # 保存文本供后续测试使用
            original_path = str(self.test_results_dir / "test_text_original.txt")
            watermarked_path = str(self.test_results_dir / "test_text_watermarked.txt")

            with open(original_path, 'w', encoding='utf-8') as f:
                f.write(test_text)
            self.logger.info(f"✓ 原始文本已保存: {original_path}")

            with open(watermarked_path, 'w', encoding='utf-8') as f:
                f.write(watermarked_text)
            self.logger.info(f"✓ 水印文本已保存: {watermarked_path}")

            # 保存数据供后续测试使用
            self.__class__.watermarked_text = watermarked_text
            self.__class__.watermark_words = watermark_words
            self.__class__.test_message = test_message

        except Exception as e:
            self.logger.error(f"❌ 文本水印嵌入失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def test_03_text_watermark_extract(self):
        """测试3: 文本水印提取与验证"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试3: 文本水印提取与验证")
        self.logger.info("=" * 60)

        try:
            # 确保测试2已运行
            if not hasattr(self.__class__, 'watermarked_text'):
                self.skipTest("需要先运行测试2")

            watermarked_text = self.__class__.watermarked_text
            watermark_words = self.__class__.watermark_words
            test_message = self.__class__.test_message

            self.logger.info("从文本提取水印...")
            self.logger.info(f"水印文本长度: {len(watermarked_text)} 字符")
            self.logger.info(f"水印词数量: {len(watermark_words)}")

            # 提取水印
            result = self.text_watermark.extract(
                watermarked_text,
                original_words=watermark_words
            )

            # 验证结果
            self.logger.info("提取结果:")
            self.logger.info(f"  success: {result['success']}")
            self.logger.info(f"  detected: {result['detected']}")
            self.logger.info(f"  confidence: {result.get('confidence', 0):.2%}")
            self.logger.info(f"  presence_score: {result.get('presence_score', 0):.2%}")
            self.logger.info(f"  detected_words: {len(result.get('watermark_words', []))}")

            self.assertTrue(result['success'], "提取操作应该成功")
            self.assertTrue(result['detected'], "应该检测到水印")
            self.assertGreater(result.get('confidence', 0), 0.5, "置信度应该 > 50%")
            self.logger.info("✓ 文本水印提取验证通过")

        except Exception as e:
            self.logger.error(f"❌ 文本水印提取失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def test_04_upload_mode(self):
        """测试4: 上传模式测试（对已有文本嵌入水印）"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试4: 上传模式测试")
        self.logger.info("=" * 60)

        try:
            # 使用不同的测试文本
            upload_text = """
            Natural language processing is a branch of artificial intelligence that helps
            computers understand, interpret and manipulate human language. NLP draws from
            many disciplines including computer science and computational linguistics in its
            pursuit to fill the gap between human communication and computer understanding.
            Modern NLP systems leverage machine learning to extract meaningful information
            from large volumes of text data.
            """

            upload_message = "upload_test_2025"
            self.logger.info(f"上传文本长度: {len(upload_text.strip())} 字符")
            self.logger.info(f"水印消息: {upload_message}")

            # 嵌入水印
            self.logger.info("\n⏳ 正在为上传文本嵌入水印...")
            result = self.text_watermark.embed(upload_text.strip(), message=upload_message)

            # 验证结果
            self.assertTrue(result['success'], "上传模式水印嵌入应该成功")
            self.assertIn('watermarked_text', result, "结果应包含watermarked_text字段")

            watermarked_text = result['watermarked_text']
            watermark_words = result['watermark_words']

            self.logger.info("✓ 上传模式水印嵌入成功")
            self.logger.info(f"  - 水印词数量: {len(watermark_words)}")
            self.logger.info(f"  - 水印文本 (前150字符):")
            self.logger.info(f"    {watermarked_text[:150]}...")

            # 保存文件
            upload_watermarked_path = str(self.test_results_dir / "test_text_upload_watermarked.txt")
            with open(upload_watermarked_path, 'w', encoding='utf-8') as f:
                f.write(watermarked_text)
            self.logger.info(f"✓ 上传模式水印文本已保存: {upload_watermarked_path}")

            # 提取验证
            self.logger.info("\n⏳ 正在验证上传模式水印...")
            extract_result = self.text_watermark.extract(
                watermarked_text,
                original_words=watermark_words
            )

            self.logger.info("上传模式提取结果:")
            self.logger.info(f"  success: {extract_result['success']}")
            self.logger.info(f"  detected: {extract_result['detected']}")
            self.logger.info(f"  confidence: {extract_result.get('confidence', 0):.2%}")
            self.logger.info(f"  presence_score: {extract_result.get('presence_score', 0):.2%}")

            self.assertTrue(extract_result['success'], "提取操作应该成功")
            self.assertTrue(extract_result['detected'], "应该检测到水印")
            self.assertGreater(extract_result.get('confidence', 0), 0.5, "置信度应该 > 50%")
            self.logger.info("✓ 上传模式水印验证通过")

        except Exception as e:
            self.logger.error(f"❌ 上传模式测试失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def test_05_unified_interface(self):
        """测试5: 统一TextWatermark接口测试"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("测试5: 统一TextWatermark接口")
        self.logger.info("=" * 60)

        try:
            # 创建统一接口实例
            unified_watermark = TextWatermark()

            # 验证默认算法
            self.logger.info(f"默认算法: {unified_watermark.algorithm}")
            self.assertEqual(unified_watermark.algorithm, 'postmark', "默认算法应为postmark")

            # 使用统一接口嵌入水印
            test_text = "This is a test text for unified watermark interface."
            test_message = "unified_test_2025"

            self.logger.info(f"\n测试文本: {test_text}")
            self.logger.info(f"水印消息: {test_message}")

            # 嵌入水印
            self.logger.info("\n⏳ 使用统一接口嵌入水印...")
            watermarked_text = unified_watermark.embed_watermark(
                text=test_text,
                message=test_message
            )

            self.assertIsInstance(watermarked_text, str, "水印文本应该是字符串")
            self.logger.info(f"✓ 统一接口水印嵌入成功")
            self.logger.info(f"  水印文本: {watermarked_text}")

            # 提取水印
            self.logger.info("\n⏳ 使用统一接口提取水印...")
            detection = unified_watermark.extract_watermark(watermarked_text)

            self.logger.info("统一接口提取结果:")
            self.logger.info(f"  detected: {detection.get('detected', False)}")
            self.logger.info(f"  confidence: {detection.get('confidence', 0):.2%}")

            # 注意：PostMark需要original_words才能准确检测，这里只做基本验证
            self.logger.info("✓ 统一接口测试完成")

        except Exception as e:
            self.logger.error(f"❌ 统一接口测试失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        cls.logger.info("\n" + "=" * 70)
        cls.logger.info("Mistral-7B + PostMark 完整集成测试完成")
        cls.logger.info("=" * 70)

        # 清理处理器以释放内存
        if hasattr(cls, 'text_watermark'):
            try:
                if hasattr(cls.text_watermark, 'clear_cache'):
                    cls.text_watermark.clear_cache()
                cls.logger.info("✓ 缓存已清理")
            except Exception as e:
                cls.logger.warning(f"清理缓存时出错: {e}")

        # 显示生成的文件
        cls.logger.info("\n生成的文件:")
        test_results = cls.test_results_dir
        if test_results.exists():
            for text_file in sorted(test_results.glob("test_text_*.txt")):
                size = text_file.stat().st_size / 1024  # KB
                cls.logger.info(f"  - {text_file.name} ({size:.2f} KB)")


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
