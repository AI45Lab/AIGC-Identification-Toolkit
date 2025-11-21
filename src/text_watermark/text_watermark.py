"""
文本水印类 - 负责文本水印的嵌入和提取
统一门面，支持多种文本水印算法（CredID, PostMark等）
"""

import yaml
import logging
from typing import Dict, Any, Optional, Union, List
from transformers import PreTrainedModel, PreTrainedTokenizer

try:
    # 相对导入（当作为包运行时）
    from .credid_watermark import CredIDWatermark
    from .postmark_watermark import PostMarkWatermark
except ImportError:
    # 绝对导入（当 src 在路径中时）
    from text_watermark.credid_watermark import CredIDWatermark
    from text_watermark.postmark_watermark import PostMarkWatermark


class TextWatermark:
    """
    文本水印处理类

    统一门面类，支持多种文本水印算法：
    - CredID: 生成时嵌入水印（修改模型logits）
    - PostMark: 后处理嵌入水印（修改已生成文本）

    通过配置文件中的 'algorithm' 参数切换算法
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化文本水印处理器

        Args:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.algorithm = self.config.get('algorithm', 'postmark')  # 默认使用PostMark

        # 延迟初始化具体算法处理器，避免无关依赖在构造时被加载
        self.watermark_processor = None
        self._initialized_algorithm = None

        self.logger.info(f"TextWatermark初始化: algorithm={self.algorithm}")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        加载配置文件，从 default_config.yaml 的 text_watermark 节读取

        Args:
            config_path: 配置文件路径（如果为None，则加载默认配置）

        Returns:
            配置字典
        """
        if config_path is None:
            # 尝试加载项目默认配置
            from pathlib import Path
            import os

            # 查找项目根目录
            current = Path(__file__).parent
            while current != current.parent:
                default_config = current / 'config' / 'default_config.yaml'
                if default_config.exists():
                    config_path = str(default_config)
                    break
                current = current.parent

            # 如果找不到配置文件，使用内置默认
            if config_path is None:
                return {
                    'algorithm': 'postmark',
                    'postmark': {
                        'embedder': 'nomic',
                        'inserter': 'mistral-7b-inst',
                        'ratio': 0.12,
                        'iterate': 'v2',
                        'threshold': 0.7
                    }
                }

        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)

        # 提取 text_watermark 节
        text_config = full_config.get('text_watermark', {})

        # 如果没有 text_watermark 节，尝试直接使用根配置（向后兼容）
        if not text_config:
            text_config = full_config

        # 确保有 algorithm 字段，默认为 postmark
        if 'algorithm' not in text_config:
            text_config['algorithm'] = 'postmark'

        return text_config

    def _setup_model(self):
        """初始化模型"""
        self.logger.info(f"设置 {self.algorithm} 模型...")

        if self.algorithm == 'credid':
            # 初始化CredID水印处理器
            credid_config = self.config.get('credid', {})
            self.watermark_processor = CredIDWatermark(credid_config)

        elif self.algorithm == 'postmark':
            # 初始化PostMark水印处理器
            postmark_config = self.config.get('postmark', {})
            self.watermark_processor = PostMarkWatermark(postmark_config)

        else:
            raise ValueError(f"不支持的算法: {self.algorithm}. 支持的算法: credid, postmark")

        # 记录已初始化的算法类型
        self._initialized_algorithm = self.algorithm
        self.logger.info(f"{self.algorithm} 模型设置完成")

    def _ensure_model(self):
        """确保对应算法的处理器已初始化"""
        if self.watermark_processor is None or self._initialized_algorithm != self.algorithm:
            self._setup_model()

    def embed_watermark(self,
                       content: str,
                       message: str,
                       model: Optional[PreTrainedModel] = None,
                       tokenizer: Optional[PreTrainedTokenizer] = None,
                       **kwargs) -> Union[str, Dict[str, Any]]:
        """
        在文本中嵌入水印

        根据算法类型自动路由到相应的处理方法：
        - CredID: content是prompt，需要model和tokenizer，生成时嵌入
        - PostMark: content是已生成文本，后处理嵌入

        Args:
            content: 输入内容
                - CredID: prompt（提示词）
                - PostMark: 已生成的完整文本
            message: 要嵌入的水印消息
            model: 语言模型（CredID需要）
            tokenizer: 分词器（CredID需要）
            **kwargs: 额外参数

        Returns:
            CredID: 返回嵌入结果字典（包含watermarked_text等）
            PostMark: 返回嵌入结果字典（包含watermarked_text等）
        """
        # 确保模型按当前算法已就绪
        self._ensure_model()

        if self.algorithm == 'credid':
            # CredID: 需要model和tokenizer
            if model is None or tokenizer is None:
                raise ValueError("CredID算法需要提供model和tokenizer参数")

            result = self.watermark_processor.embed(
                model=model,
                tokenizer=tokenizer,
                prompt=content,
                message=message,
                **kwargs
            )
            return result

        elif self.algorithm == 'postmark':
            # PostMark: 后处理模式，content是已生成的文本
            result = self.watermark_processor.embed(
                text=content,
                message=message,
                **kwargs
            )
            return result

        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")

    def extract_watermark(self,
                         watermarked_text: str,
                         model: Optional[PreTrainedModel] = None,
                         tokenizer: Optional[PreTrainedTokenizer] = None,
                         candidates_messages: Optional[List[str]] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        从文本中提取水印

        Args:
            watermarked_text: 含水印的文本
            model: 语言模型（CredID需要）
            tokenizer: 分词器（CredID需要）
            candidates_messages: 候选消息列表
            **kwargs: 额外参数
                - original_words: 原始水印词列表（PostMark检测时使用）

        Returns:
            提取结果，统一格式：
            {
                'detected': bool,
                'message': str,
                'confidence': float,
                'success': bool,
                'metadata': dict
            }
        """
        # 确保模型按当前算法已就绪
        self._ensure_model()

        if self.algorithm == 'credid':
            # CredID需要模型和分词器
            if model is None or tokenizer is None:
                raise ValueError("CredID算法需要提供model和tokenizer参数")

            result = self.watermark_processor.extract(
                watermarked_text=watermarked_text,
                model=model,
                tokenizer=tokenizer,
                candidates_messages=candidates_messages,
                **kwargs
            )

            # 统一返回格式
            return {
                'detected': result.get('success', False),
                'message': result.get('extracted_message', ''),
                'confidence': result.get('confidence', 0.0),
                'success': result.get('success', False),
                'metadata': result.get('metadata', {})
            }

        elif self.algorithm == 'postmark':
            # PostMark检测
            result = self.watermark_processor.extract(
                watermarked_text=watermarked_text,
                candidates_messages=candidates_messages,
                **kwargs
            )

            # PostMark已经返回统一格式
            return result

        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")

    def generate_with_watermark(self,
                                prompt: str,
                                message: str,
                                model: PreTrainedModel = None,
                                tokenizer: PreTrainedTokenizer = None,
                                **kwargs) -> Union[str, Dict[str, Any]]:
        """
        生成带水印的文本（支持CredID和PostMark两种算法）

        根据当前算法自动选择生成方式：
        - CredID: 需要model和tokenizer，生成时嵌入水印
        - PostMark: 使用内置LLM生成文本，然后后处理嵌入水印

        Args:
            prompt: 生成提示词
            message: 水印消息
            model: 语言模型（CredID必需，PostMark可选）
            tokenizer: 分词器（CredID必需，PostMark可选）
            **kwargs: 额外参数
                - PostMark专用参数：
                    - llm: LLM模型名称，默认使用config中的llm_for_generation
                    - max_tokens: 最大生成长度，默认600
                    - temperature: 采样温度，默认0.7

        Returns:
            生成的含水印文本或结果字典

        Raises:
            ValueError: 当CredID算法缺少必需参数时
            RuntimeError: 当PostMark生成失败时
        """
        # 确保处理器已初始化
        self._ensure_model()

        if self.algorithm == 'credid':
            # CredID算法：需要model和tokenizer
            if model is None or tokenizer is None:
                raise ValueError("CredID算法需要提供model和tokenizer参数")

            return self.embed_watermark(
                content=prompt,
                message=message,
                model=model,
                tokenizer=tokenizer,
                **kwargs
            )

        elif self.algorithm == 'postmark':
            # PostMark算法：使用内置LLM生成，然后后处理水印
            result = self.watermark_processor.generate_with_watermark(
                prompt=prompt,
                message=message,
                **kwargs
            )

            if result.get('success'):
                return result['watermarked_text']
            else:
                raise RuntimeError(f"PostMark生成失败: {result.get('error', 'Unknown')}")

        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")

    def set_algorithm(self, algorithm: str):
        """
        切换水印算法

        Args:
            algorithm: 算法名称 ('credid', 'postmark')
        """
        if algorithm not in ['credid', 'postmark']:
            raise ValueError(f"不支持的算法: {algorithm}. 支持的算法: credid, postmark")

        self.algorithm = algorithm
        self.logger.info(f"切换算法到: {algorithm}")

        # 重新初始化处理器
        self.watermark_processor = None
        self._initialized_algorithm = None

    def get_algorithm(self) -> str:
        """获取当前使用的算法"""
        return self.algorithm

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()

    def reset(self):
        """重置处理器，清除缓存"""
        if self.watermark_processor is not None:
            if hasattr(self.watermark_processor, 'reset'):
                self.watermark_processor.reset()
        self.watermark_processor = None
        self._initialized_algorithm = None
        self.logger.info("TextWatermark重置完成")


# 便捷工厂函数
def create_text_watermark(config_path: Optional[str] = None,
                         algorithm: Optional[str] = None) -> TextWatermark:
    """
    创建文本水印处理器的便捷函数

    Args:
        config_path: 配置文件路径
        algorithm: 指定算法（可选，会覆盖配置文件中的设置）

    Returns:
        TextWatermark实例
    """
    watermark = TextWatermark(config_path)

    if algorithm is not None:
        watermark.set_algorithm(algorithm)

    return watermark
