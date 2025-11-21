"""
PostMark文本水印算法封装
"""

import os
import sys
import torch
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# 添加PostMark目录到路径
POSTMARK_DIR = Path(__file__).parent / "PostMark"
if str(POSTMARK_DIR) not in sys.path:
    sys.path.insert(0, str(POSTMARK_DIR))

# PostMark相关依赖可能并非所有环境都安装，延迟记录导入错误，
# 避免在未使用PostMark功能时直接阻断整个包的加载。
POSTMARK_IMPORT_ERROR: Optional[BaseException] = None
Watermarker = NomicEmbed = LLM = compute_presence = None  # type: ignore

try:
    from PostMark.postmark.models import Watermarker, NomicEmbed, LLM  # type: ignore
    from PostMark.postmark.utils import compute_presence  # type: ignore
except ImportError as upper_exc:
    try:
        from postmark.models import Watermarker, NomicEmbed, LLM  # type: ignore
        from postmark.utils import compute_presence  # type: ignore
    except Exception as lower_exc:  # 捕获底层依赖缺失等异常
        POSTMARK_IMPORT_ERROR = lower_exc
        Watermarker = NomicEmbed = LLM = compute_presence = None  # type: ignore


class PostMarkWatermark:
    """
    PostMark文本水印算法封装

    PostMark是一个后处理水印方法，工作流程：
    1. 使用LLM生成原始文本
    2. 选择与文本语义相关的水印词
    3. 使用另一个LLM将水印词插入文本中
    4. 检测时计算水印词的存在率

    与CredID的关键区别：
    - CredID：生成时嵌入（修改logits）
    - PostMark：后处理嵌入（修改已生成文本）

    Example:
        config = {
            'embedder': 'nomic',
            'inserter': 'mistral-7b-inst',
            'ratio': 0.12,
            'iterate': 'v2'
        }
        watermark = PostMarkWatermark(config)

        # 对已生成的文本进行水印嵌入
        result = watermark.embed("This is generated text...", "watermark_message")

        # 检测水印
        detection = watermark.extract(result['watermarked_text'],
                                     candidates_messages=['watermark_message'])
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化PostMark水印处理器

        Args:
            config: 配置字典，包含以下参数：
                - embedder: 嵌入模型选择 ('nomic' 推荐本地部署)
                - inserter: 插入LLM选择 ('mistral-7b-inst', 'llama-3-8b-chat')
                - ratio: 水印词比例，默认0.12 (即12%)
                - iterate: 迭代插入版本，默认'v2'
                - threshold: 检测阈值，默认0.7
                - device: 设备设置，默认auto
                - llm_for_generation: 用于生成原始文本的LLM（可选）
        """
        if POSTMARK_IMPORT_ERROR is not None or Watermarker is None:
            raise ImportError(
                "PostMark依赖未正确安装。请先执行 `pip install together` 并确认PostMark子模块完整。"
            ) from POSTMARK_IMPORT_ERROR

        self.config = config
        self.logger = logging.getLogger(__name__)

        # 设备设置
        device_config = config.get('device', 'auto')
        if device_config == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_config)

        # 提取配置参数
        self.embedder_type = config.get('embedder', 'nomic')
        self.inserter_type = config.get('inserter', 'mistral-7b-inst')
        self.ratio = config.get('ratio', 0.12)
        self.iterate = config.get('iterate', 'v2')
        self.threshold = config.get('threshold', 0.7)
        self.inserter_temperature = float(config.get('inserter_temperature', 0.01))

        # 懒加载核心组件
        self._watermarker = None
        self._embedder = None
        self._llm = None

        # 切换到PostMark目录以确保文件路径正确
        self.original_dir = os.getcwd()
        self.postmark_dir = str(POSTMARK_DIR)

        self.logger.info(f"PostMarkWatermark初始化: embedder={self.embedder_type}, "
                        f"inserter={self.inserter_type}, ratio={self.ratio}")

    def _ensure_watermarker(self):
        """懒加载Watermarker实例"""
        if self._watermarker is None:
            try:
                # 切换到PostMark目录
                os.chdir(self.postmark_dir)

                # 初始化Watermarker
                # llm参数：用于生成原始文本（在我们的场景中，文本已由外部生成，此参数仅用于完整性）
                llm_for_gen = self.config.get('llm_for_generation', 'mistral-7b-inst')

                self._watermarker = Watermarker(
                    llm=llm_for_gen,
                    embedder=self.embedder_type,
                    inserter=self.inserter_type,
                    ratio=self.ratio,
                    iterate=self.iterate,
                    temperature=self.inserter_temperature
                )

                # 恢复原始目录
                os.chdir(self.original_dir)

                self.logger.info("Watermarker初始化成功")
            except Exception as e:
                os.chdir(self.original_dir)  # 确保恢复目录
                self.logger.error(f"Watermarker初始化失败: {e}")
                raise

    def _ensure_embedder(self):
        """懒加载Embedder实例（用于独立的水印词提取）"""
        if self._embedder is None:
            try:
                os.chdir(self.postmark_dir)

                if self.embedder_type == 'nomic':
                    self._embedder = NomicEmbed(ratio=self.ratio)
                else:
                    raise ValueError(f"不支持的embedder类型: {self.embedder_type}")

                os.chdir(self.original_dir)
                self.logger.info(f"Embedder ({self.embedder_type}) 初始化成功")
            except Exception as e:
                os.chdir(self.original_dir)
                self.logger.error(f"Embedder初始化失败: {e}")
                raise

    def generate_with_watermark(self, prompt: str, message: str, **kwargs) -> Dict[str, Any]:
        """
        从prompt生成文本并嵌入水印（PostMark两步式流程）

        PostMark AI生成流程：
        1. 使用LLM根据prompt生成原始文本
        2. 调用embed()方法在生成的文本中嵌入水印

        与CredID的区别：
        - CredID: 生成时嵌入（修改logits）
        - PostMark: 先生成完整文本，再后处理嵌入水印

        Args:
            prompt: 文本生成提示词
            message: 水印消息（用于标识）
            **kwargs: 额外参数
                - llm: LLM模型名称，默认使用config中的llm_for_generation
                - max_tokens: 最大生成token数，默认600
                - temperature: 采样温度，默认0.7

        Returns:
            包含水印文本和元数据的字典：
            {
                'watermarked_text': str,     # 带水印的文本
                'original_text': str,        # 生成的原始文本（未加水印）
                'watermark_words': List[str],# 嵌入的水印词列表
                'message': str,              # 水印消息
                'success': bool,             # 是否成功
                'metadata': dict             # 额外元数据
            }

        Raises:
            Exception: 当LLM生成失败或水印嵌入失败时抛出异常
        """
        try:
            # Step 1: 使用LLM生成原始文本
            llm_name = kwargs.get('llm', self.config.get('llm_for_generation', 'mistral-7b-inst'))
            max_tokens = kwargs.get('max_tokens', self.config.get('max_tokens', 600))
            temperature = kwargs.get('temperature', self.config.get('generation_temperature', 0.7))

            self.logger.info(f"PostMark生成流程开始: llm={llm_name}, max_tokens={max_tokens}")

            # 懒加载LLM实例
            if self._llm is None or (hasattr(self._llm, 'model_name') and self._llm.model_name != llm_name):
                try:
                    os.chdir(self.postmark_dir)
                    self._llm = LLM(llm_name)
                    os.chdir(self.original_dir)
                    self.logger.info(f"LLM ({llm_name}) 初始化成功")
                except Exception as e:
                    os.chdir(self.original_dir)
                    raise RuntimeError(f"LLM初始化失败: {e}")

            # 生成原始文本
            try:
                os.chdir(self.postmark_dir)
                generated_text = self._llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)
                os.chdir(self.original_dir)
                self.logger.info(f"LLM生成成功，文本长度: {len(generated_text)} 字符")
            except Exception as e:
                os.chdir(self.original_dir)
                raise RuntimeError(f"LLM生成失败: {e}")

            # Step 2: 在生成的文本上应用PostMark水印
            watermark_result = self.embed(text=generated_text, message=message, max_tokens=max_tokens)

            if not watermark_result['success']:
                raise RuntimeError(f"水印嵌入失败: {watermark_result.get('error', 'Unknown')}")

            # 更新元数据，标记这是AI生成+水印流程
            watermark_result['metadata']['generation_mode'] = 'ai_generated'
            watermark_result['metadata']['llm_model'] = llm_name
            watermark_result['metadata']['prompt'] = prompt
            watermark_result['metadata']['temperature'] = temperature

            self.logger.info("PostMark生成+水印流程成功完成")
            return watermark_result

        except Exception as e:
            os.chdir(self.original_dir)
            error_msg = f"PostMark生成水印失败: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def embed(self, text: str, message: str, **kwargs) -> Dict[str, Any]:
        """
        在已生成的文本中嵌入水印

        PostMark后处理流程：
        1. 接收已生成的文本（非prompt）
        2. 根据文本语义选择水印词
        3. 使用LLM将水印词自然地插入文本中

        Args:
            text: 已生成的文本（PostMark是后处理，所以这里是完整文本而非prompt）
            message: 水印消息（用于生成水印词，当前版本主要用于标识）
            **kwargs: 额外参数
                - max_tokens: 最大token数，默认600

        Returns:
            包含水印文本和元数据的字典：
            {
                'watermarked_text': str,     # 带水印的文本
                'original_text': str,        # 原始文本
                'watermark_words': List[str],# 嵌入的水印词列表
                'message': str,              # 水印消息
                'success': bool,             # 是否成功
                'metadata': dict             # 额外元数据
            }
        """
        try:
            self._ensure_watermarker()

            # PostMark的insert_watermark方法
            # text1 = 原始文本（在我们的场景中，就是输入的text）
            # text2 = 水印文本
            max_tokens = kwargs.get('max_tokens', 600)

            # 切换目录执行
            os.chdir(self.postmark_dir)
            result = self._watermarker.insert_watermark(text, max_tokens=max_tokens)
            os.chdir(self.original_dir)

            # 标准化返回格式
            return {
                'watermarked_text': result['text2'],
                'original_text': result['text1'],
                'watermark_words': result.get('list2', []),
                'original_words': result.get('list1', []),
                'message': message,  # PostMark不直接嵌入消息，这里仅作标识
                'success': True,
                'metadata': {
                    'algorithm': 'postmark',
                    'embedder': self.embedder_type,
                    'inserter': self.inserter_type,
                    'ratio': self.ratio,
                    'num_watermark_words': len(result.get('list2', [])),
                    'iterate': self.iterate
                }
            }

        except Exception as e:
            os.chdir(self.original_dir)
            self.logger.error(f"PostMark水印嵌入失败: {e}")
            return {
                'watermarked_text': None,
                'original_text': text,
                'message': message,
                'success': False,
                'error': str(e),
                'metadata': {'algorithm': 'postmark'}
            }

    def extract(self, watermarked_text: str,
                candidates_messages: Optional[List[str]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        从文本中检测水印

        PostMark检测流程：
        1. 提取文本中的候选水印词
        2. 计算水印词的存在率（presence score）
        3. 基于阈值判断是否存在水印

        Args:
            watermarked_text: 待检测的文本
            candidates_messages: 候选消息列表（PostMark主要用于标识，检测基于词存在率）
            **kwargs: 额外参数
                - threshold: 相似度阈值，默认0.7
                - original_words: 原始水印词列表（如果已知）

        Returns:
            包含检测结果的字典：
            {
                'detected': bool,            # 是否检测到水印
                'message': str,              # 识别的消息（PostMark主要返回存在率）
                'confidence': float,         # 置信度（水印词存在率）
                'watermark_words': List[str],# 检测到的水印词
                'presence_score': float,     # 水印词存在率
                'metadata': dict             # 额外信息
            }
        """
        try:
            self._ensure_embedder()

            threshold = kwargs.get('threshold', self.threshold)

            # 方案1: 如果提供了原始水印词列表，直接计算存在率
            if 'original_words' in kwargs and kwargs['original_words']:
                original_words = kwargs['original_words']

                os.chdir(self.postmark_dir)
                presence_score = compute_presence(watermarked_text, original_words, threshold=threshold)
                os.chdir(self.original_dir)

                # 判断是否检测到水印（存在率 > 50%认为有水印）
                detected = presence_score > 0.5

                return {
                    'detected': detected,
                    'message': f"Presence: {presence_score:.2%}",
                    'confidence': float(presence_score),
                    'watermark_words': original_words,
                    'presence_score': float(presence_score),
                    'success': True,
                    'metadata': {
                        'algorithm': 'postmark',
                        'detection_method': 'presence_score',
                        'threshold': threshold,
                        'num_words_checked': len(original_words)
                    }
                }

            # 方案2: 如果没有原始词列表，提取候选水印词
            else:
                os.chdir(self.postmark_dir)
                detected_words = self._embedder.get_words(watermarked_text)
                os.chdir(self.original_dir)

                # 如果有候选消息，尝试匹配
                if candidates_messages and len(detected_words) > 0:
                    # 简单策略：返回检测到的词列表
                    detected = len(detected_words) > 0
                    confidence = len(detected_words) / max(1, int(len(watermarked_text.split()) * self.ratio))
                    confidence = min(1.0, confidence)  # 限制在[0, 1]

                    return {
                        'detected': detected,
                        'message': candidates_messages[0] if candidates_messages else "Unknown",
                        'confidence': float(confidence),
                        'watermark_words': detected_words,
                        'presence_score': float(confidence),
                        'success': True,
                        'metadata': {
                            'algorithm': 'postmark',
                            'detection_method': 'word_extraction',
                            'num_detected_words': len(detected_words),
                            'candidates_provided': len(candidates_messages) if candidates_messages else 0
                        }
                    }
                else:
                    # 无足够信息进行检测
                    return {
                        'detected': False,
                        'message': '',
                        'confidence': 0.0,
                        'watermark_words': detected_words,
                        'presence_score': 0.0,
                        'success': True,
                        'metadata': {
                            'algorithm': 'postmark',
                            'detection_method': 'word_extraction',
                            'note': '需要原始水印词列表以准确检测'
                        }
                    }

        except Exception as e:
            os.chdir(self.original_dir)
            self.logger.error(f"PostMark水印检测失败: {e}")
            return {
                'detected': False,
                'message': '',
                'confidence': 0.0,
                'success': False,
                'error': str(e),
                'metadata': {'algorithm': 'postmark'}
            }

    def get_watermark_words(self, text: str) -> List[str]:
        """
        获取文本的候选水印词列表

        这是一个辅助方法，用于提取与文本语义相关的候选水印词

        Args:
            text: 输入文本

        Returns:
            候选水印词列表
        """
        try:
            self._ensure_embedder()
            os.chdir(self.postmark_dir)
            words = self._embedder.get_words(text)
            os.chdir(self.original_dir)
            return words
        except Exception as e:
            os.chdir(self.original_dir)
            self.logger.error(f"获取水印词失败: {e}")
            return []

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()

    def reset(self):
        """重置处理器，清除缓存"""
        self._watermarker = None
        self._embedder = None
        self._llm = None
        os.chdir(self.original_dir)
        self.logger.info("PostMarkWatermark重置完成")

    def __del__(self):
        """析构时确保恢复原始目录"""
        try:
            os.chdir(self.original_dir)
        except:
            pass


# 便捷工厂函数
def create_postmark_watermark(config: Optional[Dict[str, Any]] = None) -> PostMarkWatermark:
    """
    创建PostMark水印处理器的便捷函数

    Args:
        config: 配置字典，如果为None则使用默认配置

    Returns:
        PostMarkWatermark实例
    """
    if config is None:
        config = {
            'embedder': 'nomic',
            'inserter': 'mistral-7b-inst',
            'ratio': 0.12,
            'iterate': 'v2',
            'threshold': 0.7,
            'device': 'auto'
        }

    return PostMarkWatermark(config)
