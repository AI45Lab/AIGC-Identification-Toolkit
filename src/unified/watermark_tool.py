"""
统一水印工具类 - 提供文本、图像、音频和视频水印的统一接口
基于UnifiedWatermarkEngine重构，支持更简洁的API
"""

import torch
from typing import Dict, Any, Optional, Union
from PIL import Image
import logging

try:
    # 首先尝试相对导入（当作为包运行时）
    from .unified_engine import UnifiedWatermarkEngine
    from ..text_watermark.credid_watermark import CredIDWatermark
    from ..image_watermark.image_watermark import ImageWatermark
    from ..audio_watermark.audio_watermark import AudioWatermark
    from ..video_watermark.video_watermark import VideoWatermark
    HAS_ALL_WATERMARKS = True
except ImportError:
    try:
        # 回退到绝对导入（当 src 在路径中时）
        from unified_engine import UnifiedWatermarkEngine
        from text_watermark.credid_watermark import CredIDWatermark
        from image_watermark.image_watermark import ImageWatermark
        from audio_watermark.audio_watermark import AudioWatermark
        from video_watermark.video_watermark import VideoWatermark
        HAS_ALL_WATERMARKS = True
    except ImportError as e:
        HAS_ALL_WATERMARKS = False
        logging.warning(f"部分水印模块不可用: {e}")
        # 如果某些模块不可用，仅导入可用的模块
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
        
        # 基于新的统一引擎
        self.engine = UnifiedWatermarkEngine(config_path)
        
        # 为了向后兼容，保留对原始处理器的引用
        if HAS_ALL_WATERMARKS:
            self.text_watermark = None  # 延迟初始化
            self.image_watermark = None  # 延迟初始化  
            self.audio_watermark = None  # 延迟初始化
            self.video_watermark = None  # 延迟初始化
        
        self.logger.info("WatermarkTool初始化完成，基于UnifiedWatermarkEngine")
    
    # ========== 新的统一接口（推荐使用） ==========
    
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
    
    # ========== 向后兼容的接口 ==========
    
    # 文本水印接口
    def embed_text_watermark(self, text: str, watermark_key: str = None, 
                            model=None, tokenizer=None) -> str:
        """嵌入文本水印（向后兼容接口）"""
        message = watermark_key or "default_message"
        if model is None or tokenizer is None:
            raise ValueError("文本水印需要提供model和tokenizer参数")
        return self.engine.embed(text, message, 'text', model=model, tokenizer=tokenizer)
    
    def extract_text_watermark(self, text: str, watermark_key: str = None,
                              model=None, tokenizer=None) -> Dict[str, Any]:
        """提取文本水印（向后兼容接口）"""
        if model is None or tokenizer is None:
            raise ValueError("文本水印提取需要提供model和tokenizer参数")
        return self.engine.extract(text, 'text', model=model, tokenizer=tokenizer)
    
    def batch_embed_text(self, texts: list, watermark_key: str = None,
                         model=None, tokenizer=None) -> list:
        """批量文本水印嵌入"""
        message = watermark_key or "default_message"
        if model is None or tokenizer is None:
            raise ValueError("批量文本水印嵌入需要提供model和tokenizer参数")
        return [self.engine.embed(text, message, 'text', model=model, tokenizer=tokenizer) for text in texts]
    
    def batch_extract_text(self, texts: list, watermark_key: str = None,
                          model=None, tokenizer=None) -> list:
        """批量文本水印提取"""
        if model is None or tokenizer is None:
            raise ValueError("批量文本水印提取需要提供model和tokenizer参数")
        return [self.engine.extract(text, 'text', model=model, tokenizer=tokenizer) for text in texts]
    
    # 图像水印接口
    def embed_image_watermark(self, image_input: Union[str, Image.Image], 
                             watermark_key: str = None, 
                             prompt: str = None,
                             message: str = None,
                             **kwargs) -> Image.Image:
        """嵌入图像水印（向后兼容接口）"""
        message = message or watermark_key or "default_message"
        prompt = prompt or "a watermarked image"
        return self.engine.embed(prompt, message, 'image', image_input=image_input, **kwargs)
    
    def extract_image_watermark(self, image_input: Union[str, Image.Image], 
                               watermark_key: str = None,
                               **kwargs) -> Dict[str, Any]:
        """提取图像水印（向后兼容接口）"""
        return self.engine.extract(image_input, 'image', **kwargs)
    
    def generate_image_with_watermark(self, prompt: str, watermark_key: str = None, 
                                     message: str = None, **kwargs) -> Image.Image:
        """生成带水印的图像（向后兼容接口）"""
        message = message or watermark_key or "default_message"
        return self.engine.embed(prompt, message, 'image', **kwargs)
    
    def batch_embed_image(self, images: list, watermark_key: str = None) -> list:
        """批量图像水印嵌入"""
        message = watermark_key or "default_message"
        return [self.engine.embed("a watermarked image", message, 'image', image_input=img) for img in images]
    
    def batch_extract_image(self, images: list, watermark_key: str = None) -> list:
        """批量图像水印提取"""
        return [self.engine.extract(img, 'image') for img in images]
    
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
    
    def batch_embed_audio(self, 
                         audio_inputs: list, 
                         messages: list,
                         output_dir: Optional[str] = None,
                         **kwargs) -> list:
        """批量音频水印嵌入"""
        results = []
        for audio, msg in zip(audio_inputs, messages):
            result = self.engine.embed("audio content", msg, 'audio', 
                                     audio_input=audio, **kwargs)
            results.append(result)
        return results
    
    def batch_extract_audio(self, 
                           audio_inputs: list,
                           **kwargs) -> list:
        """批量音频水印提取"""
        return [self.engine.extract(audio, 'audio', **kwargs) for audio in audio_inputs]
    
    def evaluate_audio_quality(self,
                              original_audio: Union[str, torch.Tensor],
                              watermarked_audio: Union[str, torch.Tensor]) -> Dict[str, float]:
        """评估音频水印对质量的影响"""
        # 这个功能需要直接访问底层音频处理器
        if not HAS_ALL_WATERMARKS:
            raise RuntimeError("音频水印功能不可用，请检查AudioSeal依赖是否已安装")
        
        # 延迟初始化音频水印处理器
        if self.audio_watermark is None:
            from ..audio_watermark.audio_watermark import AudioWatermark
            self.audio_watermark = AudioWatermark(self.config_path)
        
        return self.audio_watermark.evaluate_quality(original_audio, watermarked_audio)
    
    # ========== 视频水印接口（新增） ==========
    
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
    
    def batch_embed_video(self,
                         video_inputs: list,
                         messages: list,
                         output_dir: Optional[str] = None,
                         **kwargs) -> list:
        """批量视频水印嵌入"""
        results = []
        for video, msg in zip(video_inputs, messages):
            result = self.engine.embed("video content", msg, 'video',
                                     video_input=video, **kwargs)
            results.append(result)
        return results
    
    def batch_extract_video(self,
                           video_inputs: list,
                           **kwargs) -> list:
        """批量视频水印提取"""
        return [self.engine.extract(video, 'video', **kwargs) for video in video_inputs]
    
    # ========== 显式标识便捷接口 ==========

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
        # 通过引擎设置算法
        if modality == 'text':
            if self.text_watermark is None:
                self.engine._get_text_watermark()
            # 文本水印算法设置可能需要特殊处理
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
            # 视频水印使用固定的算法组合
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


def main():
    """命令行入口函数"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Unified Watermark Tool - 多模态水印工具")
    parser.add_argument("--mode", choices=['text', 'image', 'audio', 'video'], required=True, 
                       help="水印模态")
    parser.add_argument("--action", choices=['embed', 'extract', 'generate'], required=True,
                       help="执行动作")
    parser.add_argument("--input", help="输入文件或文本（generate动作时不需要）")
    parser.add_argument("--prompt", help="生成提示词（用于image/audio/video模式）")
    parser.add_argument("--message", help="要嵌入的水印消息")
    parser.add_argument("--output", help="输出文件")
    parser.add_argument("--config", help="配置文件路径")
    
    # 音频特殊参数
    parser.add_argument("--voice", help="语音预设（音频TTS生成用）")
    
    # 视频特殊参数
    parser.add_argument("--frames", type=int, help="视频帧数")
    parser.add_argument("--resolution", nargs=2, type=int, help="视频分辨率 [高度 宽度]")
    
    args = parser.parse_args()
    
    # 设置日志
    import logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 创建工具实例
        tool = WatermarkTool(args.config)
        
        # 显示系统信息
        if len(sys.argv) == 1:  # 没有参数时显示帮助
            info = tool.get_system_info()
            print("=== 统一水印工具系统信息 ===")
            print(f"支持的模态: {', '.join(info['supported_modalities'])}")
            print(f"支持的算法: {info['supported_algorithms']}")
            print(f"CUDA可用: {info['device']['cuda_available']}")
            parser.print_help()
            return
        
        # 执行操作
        if args.action == 'generate':
            # 生成模式
            if not args.prompt:
                raise ValueError("生成模式需要提供--prompt参数")
            if not args.message:
                raise ValueError("生成模式需要提供--message参数")
            
            kwargs = {}
            if args.voice:
                kwargs['voice_preset'] = args.voice
            if args.frames:
                kwargs['num_frames'] = args.frames
            if args.resolution:
                kwargs['height'], kwargs['width'] = args.resolution
            if args.output:
                kwargs['output_path'] = args.output
            
            result = tool.embed(args.prompt, args.message, args.mode, **kwargs)
            print(f"✅ 生成完成: {result}")
            
        elif args.action == 'embed':
            # 嵌入模式
            if not args.input:
                raise ValueError("嵌入模式需要提供--input参数")
            if not args.message:
                raise ValueError("嵌入模式需要提供--message参数")
            
            kwargs = {}
            if args.mode == 'text':
                result = tool.embed(args.input, args.message, args.mode, **kwargs)
            else:
                result = tool.embed("embedded content", args.message, args.mode, 
                                   **{f'{args.mode}_input': args.input}, **kwargs)
            print(f"✅ 嵌入完成: {result}")
            
        elif args.action == 'extract':
            # 提取模式
            if not args.input:
                raise ValueError("提取模式需要提供--input参数")
            
            result = tool.extract(args.input, args.mode)
            print(f"提取结果:")
            print(f"  检测到水印: {result['detected']}")
            print(f"  消息: {result['message']}")
            print(f"  置信度: {result['confidence']:.3f}")
            
        else:
            raise ValueError(f"不支持的动作: {args.action}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 