"""
统一水印工具类 - 提供文本、图像、音频和视频水印的统一接口
基于UnifiedWatermarkEngine重构，支持更简洁的API
"""

import torch
from typing import Dict, Any, Optional, Union
from PIL import Image
import logging

try:
    from .unified_engine import UnifiedWatermarkEngine
    from ..text_watermark.credid_watermark import CredIDWatermark
    from ..image_watermark.image_watermark import ImageWatermark
    from ..audio_watermark.audio_watermark import AudioWatermark
    from ..video_watermark.video_watermark import VideoWatermark
    HAS_ALL_WATERMARKS = True
except ImportError:
    try:
        from unified_engine import UnifiedWatermarkEngine
        from text_watermark.credid_watermark import CredIDWatermark
        from image_watermark.image_watermark import ImageWatermark
        from audio_watermark.audio_watermark import AudioWatermark
        from video_watermark.video_watermark import VideoWatermark
        HAS_ALL_WATERMARKS = True
    except ImportError as e:
        HAS_ALL_WATERMARKS = False
        try:
            from .unified_engine import UnifiedWatermarkEngine
        except ImportError:
            try:
                from unified_engine import UnifiedWatermarkEngine
            except ImportError:
                raise ImportError(f"无法导入UnifiedWatermarkEngine: {e}. 请确保从项目根目录运行，并且 src 目录在 Python 路径中。")


class WatermarkTool:
    """
    统一水印工具类
    
    基于UnifiedWatermarkEngine的高级接口，提供向后兼容的API
    支持text/image/audio/video四种模态的水印操作
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化统一水印工具
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        self.engine = UnifiedWatermarkEngine(config_path)
        
        # 向后兼容
        if HAS_ALL_WATERMARKS:
            self.text_watermark = None  
            self.image_watermark = None    
            self.audio_watermark = None 
            self.video_watermark = None  
    
    def embed(self, content: str, message: str, modality: str, operation: str = 'watermark', **kwargs) -> Any:
        """
        统一嵌入接口

        Args:
            content: 输入内容（提示词或实际内容）
            message: 水印消息或标识文本
            modality: 模态类型 ('text', 'image', 'audio', 'video')
            operation: 操作类型 ('watermark', 'visible_mark')，默认为 'watermark'
            **kwargs: 额外参数

        Returns:
            处理后的内容
        """
        return self.engine.embed(content, message, modality, operation, **kwargs)
    
    def extract(self, content: Any, modality: str, operation: str = 'watermark', **kwargs) -> Dict[str, Any]:
        """
        统一提取接口

        Args:
            content: 待检测内容
            modality: 模态类型 ('text', 'image', 'audio', 'video')
            operation: 操作类型 ('watermark', 'visible_mark')，默认为 'watermark'
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 检测结果
        """
        return self.engine.extract(content, modality, operation, **kwargs)
    
    # ========== 向后兼容 ==========
    
    # 音频水印接口
    def embed_audio_watermark(self, 
                             audio_input: Union[str, torch.Tensor], 
                             message: str,
                             output_path: Optional[str] = None,
                             **kwargs) -> Union[torch.Tensor, str]:
        """嵌入音频水印（向后兼容接口）"""
        return self.engine.embed("audio content", message, 'audio', 
                                audio_input=audio_input, output_path=output_path, **kwargs)
    
    def extract_audio_watermark(self, 
                               audio_input: Union[str, torch.Tensor],
                               **kwargs) -> Dict[str, Any]:
        """提取音频水印（向后兼容接口）"""
        return self.engine.extract(audio_input, 'audio', **kwargs)
    
    def generate_audio_with_watermark(self, 
                                    prompt: str, 
                                    message: str,
                                    output_path: Optional[str] = None,
                                    **kwargs) -> Union[torch.Tensor, str]:
        """生成带水印的音频（向后兼容接口）"""
        return self.engine.embed(prompt, message, 'audio', 
                                output_path=output_path, **kwargs)

    
     
    # ========== 视频水印接口 ==========
    
    def embed_video_watermark(self,
                             video_input: str,
                             message: str,
                             output_path: Optional[str] = None,
                             **kwargs) -> str:
        """嵌入视频水印"""
        return self.engine.embed("video content", message, 'video',
                                video_input=video_input, output_path=output_path, **kwargs)
    
    def extract_video_watermark(self,
                               video_input: str,
                               **kwargs) -> Dict[str, Any]:
        """提取视频水印"""
        return self.engine.extract(video_input, 'video', **kwargs)
    
    def generate_video_with_watermark(self,
                                     prompt: str,
                                     message: str,
                                     output_path: Optional[str] = None,
                                     **kwargs) -> str:
        """生成带水印的视频"""
        return self.engine.embed(prompt, message, 'video', 
                                output_path=output_path, **kwargs)
    
    # ========== 显式标识接口 ==========

    def add_visible_mark(self, content: Any, message: str, modality: str, **kwargs) -> Any:
        """
        添加显式标识的便捷方法

        Args:
            content: 要添加标识的内容
            message: 标识文本
            modality: 模态类型
            **kwargs: 额外参数

        Returns:
            添加标识后的内容
        """
        return self.engine.embed(content, message, modality, operation='visible_mark', **kwargs)

    def detect_visible_mark(self, content: Any, modality: str, **kwargs) -> Dict[str, Any]:
        """
        检测显式标识的便捷方法

        Args:
            content: 待检测内容
            modality: 模态类型
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 检测结果
        """
        return self.engine.extract(content, modality, operation='visible_mark', **kwargs)

    # ========== 通用接口 ==========

    def get_supported_algorithms(self) -> Dict[str, list]:
        """获取支持的算法列表"""
        return self.engine.get_default_algorithms()

    def get_supported_modalities(self) -> list:
        """获取支持的模态列表"""
        return self.engine.get_supported_modalities()

    def get_supported_operations(self) -> list:
        """获取支持的操作列表"""
        return self.engine.get_supported_operations()

    def get_operation_info(self) -> Dict[str, Dict]:
        """获取操作信息"""
        return self.engine.get_operation_info()
    
    def set_algorithm(self, modality: str, algorithm: str):
        """设置指定模态的算法"""

        if modality == 'text':
            if self.text_watermark is None:
                self.engine._get_text_watermark()
            pass
        elif modality == 'image':
            if self.image_watermark is None:
                self.engine._get_image_watermark()
            self.engine._get_image_watermark().algorithm = algorithm
        elif modality == 'audio':
            if self.audio_watermark is None:
                self.engine._get_audio_watermark()
            self.engine._get_audio_watermark().algorithm = algorithm
        elif modality == 'video':
            pass
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        import torch
        
        info = {
            'supported_modalities': self.get_supported_modalities(),
            'supported_algorithms': self.get_supported_algorithms(),
            'has_all_watermarks': HAS_ALL_WATERMARKS,
            'device': {
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'current_device': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
            },
            'config_path': self.config_path
        }
        
        return info



if __name__ == "__main__":
    main() 