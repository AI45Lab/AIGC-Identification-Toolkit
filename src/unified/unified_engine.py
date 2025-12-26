"""
多模态水印工具统一引擎
"""

import torch
import logging
from typing import Dict, Any, Optional, Union
from PIL import Image

try:
    from ..text_watermark.text_watermark import TextWatermark
    from ..image_watermark.image_watermark import ImageWatermark
    from ..audio_watermark.audio_watermark import AudioWatermark
    from ..video_watermark.video_watermark import VideoWatermark
except ImportError:
    try:
        from text_watermark.text_watermark import TextWatermark
        from image_watermark.image_watermark import ImageWatermark
        from audio_watermark.audio_watermark import AudioWatermark
        from video_watermark.video_watermark import VideoWatermark
    except ImportError as e:
        raise ImportError(f"无法导入水印模块: {e}. 请确保从项目根目录运行，并且 src 目录在 Python 路径中。")


class UnifiedWatermarkEngine:
    """
    多模态水印统一引擎
    
    - 统一的embed/extract接口
    - 支持text/image/audio/video四种模态
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化统一水印引擎
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.logger = logging.getLogger(__name__)
        
        self._text_watermark = None
        self._image_watermark = None  
        self._audio_watermark = None
        self._video_watermark = None

        self._text_model = None
        self._text_tokenizer = None
        
        self.config_path = config_path
        
        self.logger.info("UnifiedWatermarkEngine初始化完成")
    
    def _project_root(self) -> str:
        """获取项目根目录"""
        import os
        return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    def _candidate_cache_dirs(self) -> list:
        """返回可能的本地缓存目录列表"""
        import os
        candidates = []
        if os.getenv('HF_HOME'):
            candidates.append(os.path.join(os.getenv('HF_HOME'), 'hub'))
        if os.getenv('HF_HUB_CACHE'):
            candidates.append(os.getenv('HF_HUB_CACHE'))
        candidates.append(os.path.join(self._project_root(), 'models'))
        candidates.append(os.path.expanduser('~/.cache/huggingface/hub'))
        seen = set()
        ordered = []
        for p in candidates:
            if p and p not in seen:
                seen.add(p)
                ordered.append(p)
        return ordered

    def _load_text_config(self) -> Dict[str, Any]:
        """加载文本水印配置。从 config/default_config.yaml 的 text_watermark 节读取。"""
        import os
        import yaml
        cfg_path = None
        if self.config_path and os.path.isfile(self.config_path):
            cfg_path = self.config_path
        else:
            default_path = os.path.join(self._project_root(), 'config', 'default_config.yaml')
            if os.path.isfile(default_path):
                cfg_path = default_path
        if cfg_path is None:
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
        with open(cfg_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        return data.get('text_watermark', {
            'algorithm': 'postmark',
            'postmark': {
                'embedder': 'nomic',
                'inserter': 'mistral-7b-inst',
                'ratio': 0.12
            }
        })

    def _init_text_model_tokenizer(self):
        """使用与 test_complex_messages_real.py 一致的策略初始化文本模型与分词器（离线优先）。"""
        if self._text_model is not None and self._text_tokenizer is not None:
            return
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')

        cfg = self._load_text_config()
        primary_model = cfg.get('model_name', 'sshleifer/tiny-gpt2')
        model_cfg = cfg.get('model_config', {})

        candidate_models = [m for m in [primary_model, 'sshleifer/tiny-gpt2'] if m]

        candidate_cache_dirs = []
        if model_cfg.get('cache_dir'):
            candidate_cache_dirs.append(model_cfg.get('cache_dir'))
        candidate_cache_dirs.extend(self._candidate_cache_dirs())

        trust_remote_code = bool(model_cfg.get('trust_remote_code', True))
        last_error = None

        for model_name in candidate_models:
            for cache_dir in candidate_cache_dirs:
                try:
                    if cache_dir and not os.path.isdir(cache_dir):
                        continue
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        trust_remote_code=trust_remote_code,
                        use_fast=True
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        trust_remote_code=trust_remote_code,
                        device_map=model_cfg.get('device_map', 'auto'),
                        torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32)
                    )
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    self._text_model = model
                    self._text_tokenizer = tokenizer
                    return
                except Exception as e:
                    last_error = e
                    continue

        self.logger.warning(f"离线加载文本模型失败，稍后在调用时仍将报错。最后错误: {last_error}")

    def _get_text_watermark(self) -> TextWatermark:
        """获取文本水印处理器"""
        if self._text_watermark is None:
            self._text_watermark = TextWatermark(self.config_path)
            if self._text_watermark.algorithm == 'credid':
                self._init_text_model_tokenizer()
        return self._text_watermark
    
    def _get_image_watermark(self) -> ImageWatermark:
        """获取图像水印处理器"""
        if self._image_watermark is None:
            self._image_watermark = ImageWatermark(self.config_path)
            self._image_watermark.algorithm = 'videoseal'
        return self._image_watermark
    
    def _get_audio_watermark(self) -> AudioWatermark:
        """获取音频水印处理器"""
        if self._audio_watermark is None:
            self._audio_watermark = AudioWatermark(self.config_path)
        return self._audio_watermark
    
    def _get_video_watermark(self) -> VideoWatermark:
        """获取视频水印处理器"""
        if self._video_watermark is None:
            from ..video_watermark.video_watermark import create_video_watermark
            self._video_watermark = create_video_watermark()
        return self._video_watermark
    
    def embed(self, content: str, message: str, modality: str, operation: str = 'watermark', **kwargs) -> Any:
        """
        统一嵌入接口

        Args:
            content: 输入内容
                - watermark模式：提示词(AI生成) 或 实际内容(文件上传)
                - visible_mark模式：要添加标识的实际内容
            message: 要嵌入的水印消息或标识文本
            modality: 模态类型 ('text', 'image', 'audio', 'video')
            operation: 操作类型 ('watermark', 'visible_mark')，默认为 'watermark'
            **kwargs: 额外参数（如model, tokenizer等）

        Returns:
            处理后的内容（具体类型取决于模态和操作）
            - text: str
            - image: PIL.Image
            - audio: torch.Tensor 或 str（如果指定output_path）
            - video: str（视频文件路径）
        """
        
        try:
            if operation == 'watermark':
                return self._embed_watermark(content, message, modality, **kwargs)
            elif operation == 'visible_mark':
                return self._embed_visible_mark(content, message, modality, **kwargs)
            else:
                raise ValueError(f"不支持的操作类型: {operation}")

        except Exception as e:
            self.logger.error(f"{operation}操作失败: {e}")
            raise

    def _embed_watermark(self, content: str, message: str, modality: str, **kwargs) -> Any:
        """
        原有的水印嵌入逻辑

        Args:
            content: 输入提示（原prompt参数）
            message: 水印消息
            modality: 模态类型
            **kwargs: 额外参数
        """
        try:
            if modality == 'text':

                watermark = self._get_text_watermark()
                if watermark.algorithm == 'credid':
                    model = kwargs.get('model') or self._text_model
                    tokenizer = kwargs.get('tokenizer') or self._text_tokenizer

                    if model is None or tokenizer is None:
                        raise ValueError("CredID算法需要提供model和tokenizer参数")

                    result = watermark.embed_watermark(content, message, model=model, tokenizer=tokenizer)

                    if result.get('success'):
                        return result['watermarked_text']
                    else:
                        raise RuntimeError(f"CredID水印嵌入失败: {result.get('error', 'Unknown error')}")

                elif watermark.algorithm == 'postmark':
                    if 'text_input' in kwargs:
                        result = watermark.embed_watermark(content, message, **kwargs)
                    else:
                        result = watermark.generate_with_watermark(
                            prompt=content,
                            message=message,
                            **kwargs
                        )
                        if isinstance(result, str):
                            return result
                        elif isinstance(result, dict) and result.get('success'):
                            return result['watermarked_text']
                        else:
                            raise RuntimeError(f"PostMark生成失败: {result.get('error', 'Unknown error') if isinstance(result, dict) else 'Unknown'}")

                    if result.get('success'):
                        return result['watermarked_text']
                    else:
                        raise RuntimeError(f"PostMark水印嵌入失败: {result.get('error', 'Unknown error')}")

                else:
                    raise ValueError(f"不支持的文本水印算法: {watermark.algorithm}")


            elif modality == 'image':
                watermark = self._get_image_watermark()
                if 'image_input' in kwargs:
                    image_input = kwargs.pop('image_input')  
                    return watermark.embed_watermark(
                        image_input,
                        message=message,
                        **kwargs
                    )
                else:
                    return watermark.generate_with_watermark(
                        content,
                        message=message,
                        return_original=True,  
                        **kwargs
                    )

            elif modality == 'audio':
                watermark = self._get_audio_watermark()
                if 'audio_input' in kwargs:
                    audio_input = kwargs.pop('audio_input')  
                    return watermark.embed_watermark(
                        audio_input,
                        message,
                        **kwargs
                    )
                else:
                    return watermark.generate_audio_with_watermark(
                        content,
                        message,
                        return_original=True,
                        **kwargs
                    )

            elif modality == 'video':
                watermark = self._get_video_watermark()
                if 'video_input' in kwargs:
                    video_input = kwargs.pop('video_input')  
                    return watermark.embed_watermark(
                        video_input,
                        message,
                        **kwargs
                    )
                else:
                    if 'height' not in kwargs:
                        kwargs['height'] = 320
                    if 'width' not in kwargs:
                        kwargs['width'] = 512
                    return watermark.generate_video_with_watermark(
                        content,
                        message,
                        return_original=True,  
                        **kwargs
                    )
            else:
                raise ValueError(f"不支持的模态类型: {modality}")

        except Exception as e:
            self.logger.error(f"{modality}水印嵌入失败: {e}")
            raise

    def _embed_visible_mark(self, content: Any, message: str, modality: str, **kwargs) -> Any:
        """
        显式标识嵌入逻辑

        Args:
            content: 要添加标识的实际内容
                - text: 文本字符串
                - image: PIL.Image对象或图像文件路径
                - audio: 音频文件路径
                - video: 视频文件路径
            message: 标识文本
            modality: 模态类型
            **kwargs: 额外参数
        """
        try:
            from ..utils.visible_mark import (
                add_text_mark_to_text,
                add_overlay_to_image,
                add_overlay_to_video_ffmpeg,
                add_voice_mark_to_audio
            )
        except ImportError:
            try:
                from src.utils.visible_mark import (
                    add_text_mark_to_text,
                    add_overlay_to_image,
                    add_overlay_to_video_ffmpeg,
                    add_voice_mark_to_audio
                )
            except ImportError:
                from utils.visible_mark import (
                    add_text_mark_to_text,
                    add_overlay_to_image,
                    add_overlay_to_video_ffmpeg,
                    add_voice_mark_to_audio
                )

        try:
            if modality == 'text':
                position = kwargs.get('position', 'start')
                return add_text_mark_to_text(content, message, position)

            elif modality == 'image':
                if isinstance(content, str):
                    from PIL import Image
                    image = Image.open(content)
                else:
                    image = content

                return add_overlay_to_image(
                    image,
                    message,
                    position=kwargs.get('position', 'bottom_right'),
                    font_percent=kwargs.get('font_percent', 5.0),
                    font_color=kwargs.get('font_color', '#FFFFFF'),
                    bg_rgba=kwargs.get('bg_rgba', None)
                )

            elif modality == 'audio':
                output_path = kwargs.get('output_path')
                if not output_path:
                    output_path = self._generate_output_path(content, 'audio', 'visible_mark')

                return add_voice_mark_to_audio(
                    content,
                    output_path,
                    message,
                    position=kwargs.get('position', 'start'),
                    voice_preset=kwargs.get('voice_preset', 'v2/zh_speaker_6')
                )

            elif modality == 'video':
                output_path = kwargs.get('output_path')
                if not output_path:
                    output_path = self._generate_output_path(content, 'video', 'visible_mark')

                return add_overlay_to_video_ffmpeg(
                    content,
                    output_path,
                    message,
                    position=kwargs.get('position', 'bottom_right'),
                    font_percent=kwargs.get('font_percent', 5.0),
                    duration_seconds=kwargs.get('duration_seconds', 2.0),
                    font_color=kwargs.get('font_color', 'white'),
                    box_color=kwargs.get('box_color', 'transparent')
                )
            else:
                raise ValueError(f"不支持的模态类型: {modality}")

        except Exception as e:
            self.logger.error(f"{modality}显式标识添加失败: {e}")
            raise

    def _generate_output_path(self, input_path: str, modality: str, operation: str) -> str:
        """生成统一的输出路径"""
        import os
        from pathlib import Path
        from datetime import datetime

        input_path = Path(input_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if operation == 'visible_mark':
            suffix = 'marked'
        else:
            suffix = 'watermarked'

        if modality == 'audio':
            ext = '.wav'
        elif modality == 'video':
            ext = '.mp4'
        else:
            ext = input_path.suffix

        output_name = f"{input_path.stem}_{suffix}_{timestamp}{ext}"
        output_dir = Path("demo_outputs")
        output_dir.mkdir(exist_ok=True)

        return str(output_dir / output_name)
    
    def extract(self, content: Any, modality: str, operation: str = 'watermark', **kwargs) -> Dict[str, Any]:
        """
        统一提取接口

        Args:
            content: 待检测内容
                - text: str
                - image: PIL.Image 或 str（文件路径）
                - audio: torch.Tensor 或 str（文件路径）
                - video: str（文件路径）
            modality: 模态类型 ('text', 'image', 'audio', 'video')
            operation: 操作类型 ('watermark', 'visible_mark')，默认为 'watermark'
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 统一格式的检测结果
                - detected: bool, 是否检测到水印/标识
                - message: str, 提取的消息
                - confidence: float, 置信度 (0.0-1.0)
                - metadata: dict, 额外信息
        """

        try:
            if operation == 'watermark':
                return self._extract_watermark(content, modality, **kwargs)
            elif operation == 'visible_mark':
                return self._extract_visible_mark(content, modality, **kwargs)
            else:
                raise ValueError(f"不支持的操作类型: {operation}")

        except Exception as e:
            self.logger.error(f"{operation}提取失败: {e}")
            return {
                'detected': False,
                'message': '',
                'confidence': 0.0,
                'metadata': {'error': str(e), 'operation': operation}
            }

    def _extract_watermark(self, content: Any, modality: str, **kwargs) -> Dict[str, Any]:
        """
        原有的水印提取逻辑

        Args:
            content: 待检测内容
            modality: 模态类型
            **kwargs: 额外参数
        """
        try:
            if modality == 'text':
                watermark = self._get_text_watermark()

                if watermark.algorithm == 'credid':
                    model = kwargs.get('model') or self._text_model
                    tokenizer = kwargs.get('tokenizer') or self._text_tokenizer

                    if model is None or tokenizer is None:
                        raise ValueError("CredID算法需要提供model和tokenizer参数")

                    result = watermark.extract_watermark(
                        content,
                        model=model,
                        tokenizer=tokenizer,
                        candidates_messages=kwargs.get('candidates_messages'),
                        **kwargs
                    )

                elif watermark.algorithm == 'postmark':
                    result = watermark.extract_watermark(
                        content,
                        candidates_messages=kwargs.get('candidates_messages'),
                        **kwargs
                    )

                else:
                    raise ValueError(f"不支持的文本水印算法: {watermark.algorithm}")

                return result

            elif modality == 'image':
                watermark = self._get_image_watermark()
                result = watermark.extract_watermark(
                    content,
                    replicate=kwargs.get('replicate', 32),
                    chunk_size=kwargs.get('chunk_size', 16),
                    **kwargs
                )
                return {
                    'detected': result.get('detected', False),
                    'message': result.get('message', ''),
                    'confidence': result.get('confidence', 0.0),
                    'metadata': result.get('metadata', {})
                }

            elif modality == 'audio':
                watermark = self._get_audio_watermark()
                result = watermark.extract_watermark(content, **kwargs)
                return {
                    'detected': result.get('detected', False),
                    'message': result.get('message', ''),
                    'confidence': result.get('confidence', 0.0),
                    'metadata': result.get('metadata', {})
                }

            elif modality == 'video':
                watermark = self._get_video_watermark()
                result = watermark.extract_watermark(
                    content,
                    chunk_size=kwargs.get('chunk_size', 16),
                    **kwargs
                )
                return {
                    'detected': result.get('detected', False),
                    'message': result.get('message', ''),
                    'confidence': result.get('confidence', 0.0),
                    'metadata': result.get('metadata', {})
                }
            else:
                raise ValueError(f"不支持的模态类型: {modality}")

        except Exception as e:
            self.logger.error(f"{modality}水印提取失败: {e}")
            return {
                'detected': False,
                'message': '',
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }

    def _extract_visible_mark(self, content: Any, modality: str, **kwargs) -> Dict[str, Any]:
        """
        显式标识检测逻辑（预留接口）

        Args:
            content: 待检测内容
            modality: 模态类型
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 统一格式的检测结果
        """
        self.logger.info(f"显式标识检测: modality={modality}")

        try:
            if modality == 'text':
                lines = content.split('\n')
                for line in lines:
                    if '人工智能' in line and ('生成' in line or '合成' in line):
                        return {
                            'detected': True,
                            'message': line.strip(),
                            'confidence': 1.0,
                            'metadata': {
                                'operation': 'visible_mark',
                                'modality': modality,
                                'detection_method': 'text_pattern_match'
                            }
                        }
                return {
                    'detected': False,
                    'message': '',
                    'confidence': 0.0,
                    'metadata': {'operation': 'visible_mark', 'modality': modality}
                }
            else:
                return {
                    'detected': None,  
                    'message': '显式标识检测暂不支持此模态',
                    'confidence': 0.0,
                    'metadata': {
                        'operation': 'visible_mark',
                        'modality': modality,
                        'note': '显式标识通常是可见的，无需提取'
                    }
                }

        except Exception as e:
            self.logger.error(f"显式标识检测失败: {e}")
            return {
                'detected': False,
                'message': '',
                'confidence': 0.0,
                'metadata': {'error': str(e), 'operation': 'visible_mark'}
            }
    
    def get_supported_modalities(self) -> list:
        """获取支持的模态列表"""
        return ['text', 'image', 'audio', 'video']
    
    def get_default_algorithms(self) -> Dict[str, str]:
        """获取各模态的默认算法"""
        return {
            'text': 'postmark',  
            'image': 'videoseal', 
            'audio': 'audioseal',
            'video': 'hunyuan+videoseal'
        }

    def get_supported_operations(self) -> list:
        """获取支持的操作列表"""
        return ['watermark', 'visible_mark']

    def get_operation_info(self) -> Dict[str, Dict]:
        """获取操作信息"""
        return {
            'watermark': {
                'description': '隐式水印，用于版权保护和内容溯源',
                'modalities': ['text', 'image', 'audio', 'video'],
                'supports_extract': True
            },
            'visible_mark': {
                'description': '显式标识，用于AI生成内容合规标注',
                'modalities': ['text', 'image', 'audio', 'video'],
                'supports_extract': True  
            }
        }



def create_unified_engine(config_path: Optional[str] = None) -> UnifiedWatermarkEngine:
    """
    创建统一水印引擎的便捷函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        UnifiedWatermarkEngine: 统一水印引擎实例
    """
    return UnifiedWatermarkEngine(config_path)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    
    engine = create_unified_engine()
    