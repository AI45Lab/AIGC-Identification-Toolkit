"""
音频水印模块
提供基于AudioSeal的音频水印功能和Bark的文本转音频生成功能
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "AudioWatermark Team"

# 核心导入
try:
    from .audio_watermark import AudioWatermark, create_audio_watermark
    from .audioseal_wrapper import AudioSealWrapper, MessageEncoder, create_audioseal_wrapper
    from .utils import (
        AudioIOUtils, AudioProcessingUtils, AudioQualityUtils, 
        AudioVisualizationUtils, FileUtils,
        load_audio_simple, save_audio_simple
    )
    
    # 可选导入（Bark相关）
    try:
        from .bark_generator import BarkGenerator, create_bark_generator
        HAS_BARK = True
    except ImportError:
        HAS_BARK = False
    
    # 标记主要功能可用
    AUDIOSEAL_AVAILABLE = True
    
except ImportError as e:
    # 如果核心功能不可用，创建占位符
    AUDIOSEAL_AVAILABLE = False
    HAS_BARK = False
    
    # 创建错误类
    class AudioWatermarkNotAvailable:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"音频水印功能不可用: {e}\n"
                "请安装AudioSeal相关依赖"
            )
    
    # 设置占位符
    AudioWatermark = AudioWatermarkNotAvailable
    AudioSealWrapper = AudioWatermarkNotAvailable
    create_audio_watermark = AudioWatermarkNotAvailable
    create_audioseal_wrapper = AudioWatermarkNotAvailable


# 导出的公共API
__all__ = [
    # 核心类
    'AudioWatermark',
    'AudioSealWrapper', 
    'MessageEncoder',
    
    # 工具类
    'AudioIOUtils',
    'AudioProcessingUtils', 
    'AudioQualityUtils',
    'AudioVisualizationUtils',
    'FileUtils',
    
    # 便捷函数
    'create_audio_watermark',
    'create_audioseal_wrapper',
    'load_audio_simple',
    'save_audio_simple',
    
    # 状态标志
    'AUDIOSEAL_AVAILABLE',
    'HAS_BARK',
    
    # 版本信息
    '__version__',
    '__author__'
]

# 条件导出
if HAS_BARK:
    __all__.extend(['BarkGenerator', 'create_bark_generator'])


def get_version():
    """获取版本信息"""
    return __version__


def get_available_features():
    """获取可用功能列表"""
    features = {
        'audioseal_available': AUDIOSEAL_AVAILABLE,
        'bark_available': HAS_BARK,
        'audio_io': True,  # 基础音频I/O始终可用
    }
    
    if AUDIOSEAL_AVAILABLE:
        features.update({
            'watermark_embed': True,
            'watermark_extract': True,
            'batch_processing': True,
            'quality_evaluation': True
        })
    
    if HAS_BARK:
        features.update({
            'text_to_speech': True,
            'voice_cloning': True,
            'multilingual_support': True,
            'text_to_audio_watermark': True
        })
    
    return features


def print_status():
    """打印模块状态信息"""
    print(f"音频水印模块 v{__version__}")
    print(f"作者: {__author__}")
    print()
    
    features = get_available_features()
    print("功能状态:")
    
    status_symbols = {True: "✅", False: "❌"}
    
    print(f"  {status_symbols[features['audioseal_available']]} AudioSeal 水印功能")
    if features['audioseal_available']:
        print(f"    {status_symbols[features['watermark_embed']]} 水印嵌入")
        print(f"    {status_symbols[features['watermark_extract']]} 水印提取")
        print(f"    {status_symbols[features['batch_processing']]} 批处理")
        print(f"    {status_symbols[features['quality_evaluation']]} 质量评估")
    
    print(f"  {status_symbols[features['bark_available']]} Bark 文本转音频")
    if features['bark_available']:
        print(f"    {status_symbols[features['text_to_speech']]} 文本转语音")
        print(f"    {status_symbols[features['voice_cloning']]} 语音克隆")
        print(f"    {status_symbols[features['multilingual_support']]} 多语言支持")
        print(f"    {status_symbols[features['text_to_audio_watermark']]} 文本→音频→水印")
    
    print(f"  {status_symbols[features['audio_io']]} 音频I/O工具")
    
    if not features['audioseal_available']:
        print("\n安装AudioSeal:")
        print("  pip install audioseal")
    
    if not features['bark_available']:
        print("\n安装Bark:")
        print("  pip install git+https://github.com/suno-ai/bark.git")


if __name__ == "__main__":
    print_status()