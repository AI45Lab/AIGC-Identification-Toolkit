"""
音频水印类 - 负责音频水印的嵌入和提取
提供统一的音频水印接口，类似于ImageWatermark设计
"""

import torch
import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import warnings

from .audioseal_wrapper import AudioSealWrapper
from .utils import AudioIOUtils, AudioProcessingUtils, FileUtils

# 可选导入BarkGenerator
try:
    from .bark_generator import BarkGenerator
    HAS_BARK_GENERATOR = True
except ImportError:
    HAS_BARK_GENERATOR = False


class AudioWatermark:
    """音频水印处理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化音频水印处理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.algorithm = self.config.get('algorithm', 'audioseal')
        
        # 延迟初始化具体算法处理器，避免在构造时无关依赖被加载
        self.watermark_processor = None
        self._initialized_algorithm = None
        
        # 延迟初始化Bark生成器
        self.bark_generator = None
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"AudioWatermark初始化完成，算法: {self.algorithm}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            # 使用默认配置
            return {
                'algorithm': 'audioseal',
                'model_name': 'audioseal_wm_16bits',
                'detector_name': 'audioseal_detector_16bits',
                'sample_rate': 16000,
                'message_bits': 16,
                'device': None,
                'watermark_strength': 1.0,
                'detection_threshold': 0.5,
                'message_threshold': 0.5,
                'output_dir': 'audio_outputs',
                'bark': {
                    'model_size': 'large',
                    'use_gpu': True,
                    'temperature': 0.8,
                    'default_voice': 'v2/en_speaker_6'
                }
            }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('audio_watermark', {})
        except Exception as e:
            self.logger.warning(f"加载配置文件失败: {e}，使用默认配置")
            return self._load_config(None)
    
    def _setup_model(self):
        """初始化模型"""
        self.logger.info(f"设置 {self.algorithm} 模型...")
        
        if self.algorithm == 'audioseal':
            # 初始化AudioSeal水印处理器
            self.watermark_processor = AudioSealWrapper(
                device=self.config.get('device'),
                nbits=self.config.get('message_bits', 16)
            )
        else:
            # 其他算法的占位符
            self.watermark_processor = None
            self.logger.warning(f"算法 {self.algorithm} 尚未实现")
        
        # 记录已初始化的算法类型
        self._initialized_algorithm = self.algorithm
        self.logger.info(f"{self.algorithm} 模型设置完成")
    
    def _ensure_model(self):
        """确保对应算法的处理器已初始化"""
        if self.watermark_processor is None or self._initialized_algorithm != self.algorithm:
            self._setup_model()
    
    def _ensure_bark_generator(self):
        """确保Bark生成器已初始化"""
        if self.bark_generator is None:
            if not HAS_BARK_GENERATOR:
                raise ImportError(
                    "BarkGenerator不可用。请安装Bark: "
                    "pip install git+https://github.com/suno-ai/bark.git"
                )
            
            self.logger.info("初始化Bark生成器...")
            bark_config = self.config.get('bark', {})
            
            self.bark_generator = BarkGenerator(
                device=self.config.get('device'),
                model_size=bark_config.get('model_size', 'large'),
                use_gpu=bark_config.get('use_gpu', True),
                target_sample_rate=self.config.get('sample_rate', 16000)
            )
            
            # 设置默认语音
            default_voice = bark_config.get('default_voice')
            if default_voice:
                try:
                    self.bark_generator.set_default_voice(default_voice)
                except ValueError:
                    self.logger.warning(f"无效的默认语音: {default_voice}")
            
            self.logger.info("Bark生成器初始化完成")
    
    def embed_watermark(self, 
                       audio_input: Union[str, torch.Tensor],
                       message: str,
                       output_path: Optional[str] = None,
                       **kwargs) -> Union[torch.Tensor, str]:
        """
        在音频中嵌入水印
        
        Args:
            audio_input: 输入音频（文件路径或torch.Tensor）
            message: 要嵌入的消息字符串
            output_path: 输出文件路径，None则返回tensor
            **kwargs: 其他参数（如watermark_strength等）
            
        Returns:
            Union[torch.Tensor, str]: 含水印的音频tensor或输出文件路径
        """
        self._ensure_model()
        
        if self.algorithm == 'audioseal':
            if self.watermark_processor is None:
                raise RuntimeError("AudioSeal处理器未初始化")
            
            self.logger.info(f"开始嵌入水印，消息: '{message}'")
            
            # 加载音频
            if isinstance(audio_input, str):
                audio_tensor, sample_rate = AudioIOUtils.load_audio(
                    audio_input, 
                    target_sample_rate=self.config.get('sample_rate', 16000)
                )
                self.logger.info(f"音频加载成功: {audio_tensor.shape}, SR: {sample_rate}")
            else:
                audio_tensor = audio_input
                sample_rate = self.config.get('sample_rate', 16000)
            
            # 获取水印强度
            alpha = kwargs.get('watermark_strength', 
                              self.config.get('watermark_strength', 1.0))
            
            # 嵌入水印
            watermarked_audio = self.watermark_processor.embed(
                audio_tensor,
                message,
                input_sample_rate=sample_rate,
                alpha=alpha
            )
            
            # 保存或返回
            if output_path:
                # 确保输出目录存在
                output_path = Path(output_path)
                FileUtils.ensure_dir(output_path.parent)
                
                # 避免文件名冲突
                output_path = FileUtils.get_unique_filename(output_path)
                
                # 保存音频
                AudioIOUtils.save_audio(
                    watermarked_audio, 
                    output_path, 
                    sample_rate
                )
                
                self.logger.info(f"带水印音频已保存: {output_path}")
                return str(output_path)
            else:
                return watermarked_audio
        
        else:
            raise NotImplementedError(f"算法 {self.algorithm} 尚未实现")
    
    def extract_watermark(self, 
                         audio_input: Union[str, torch.Tensor],
                         **kwargs) -> Dict[str, Any]:
        """
        从音频中提取水印
        
        Args:
            audio_input: 含水印的音频（文件路径或torch.Tensor）
            **kwargs: 其他参数（如detection_threshold等）
            
        Returns:
            Dict[str, Any]: 提取结果，包含detected、message、confidence等字段
        """
        self._ensure_model()
        
        if self.algorithm == 'audioseal':
            if self.watermark_processor is None:
                raise RuntimeError("AudioSeal处理器未初始化")
            
            self.logger.info("开始提取水印...")
            
            # 加载音频
            if isinstance(audio_input, str):
                audio_tensor, sample_rate = AudioIOUtils.load_audio(
                    audio_input,
                    target_sample_rate=self.config.get('sample_rate', 16000)
                )
                self.logger.info(f"音频加载成功: {audio_tensor.shape}, SR: {sample_rate}")
            else:
                audio_tensor = audio_input
                sample_rate = self.config.get('sample_rate', 16000)
            
            # 获取检测阈值
            detection_threshold = kwargs.get('detection_threshold',
                                           self.config.get('detection_threshold', 0.5))
            message_threshold = kwargs.get('message_threshold',
                                         self.config.get('message_threshold', 0.5))
            
            # 提取水印
            result = self.watermark_processor.extract(
                audio_tensor,
                input_sample_rate=sample_rate,
                detection_threshold=detection_threshold,
                message_threshold=message_threshold
            )
            
            # 添加额外信息
            result.update({
                "audio_shape": audio_tensor.shape,
                "sample_rate": sample_rate,
                "algorithm": self.algorithm
            })
            
            self.logger.info(
                f"水印提取完成 - 检测: {result['detected']}, "
                f"置信度: {result['confidence']:.3f}, "
                f"消息: '{result['message']}'"
            )
            
            return result
        
        else:
            raise NotImplementedError(f"算法 {self.algorithm} 尚未实现")
    
    def generate_audio_with_watermark(self,
                                    prompt: str,
                                    message: str,
                                    output_path: Optional[str] = None,
                                    return_original: bool = False,
                                    **kwargs) -> Union[torch.Tensor, str, Dict[str, Union[torch.Tensor, str]]]:
        """
        生成带水印的音频（使用Bark生成器）
        
        Args:
            prompt: 文本提示词
            message: 要嵌入的水印消息
            output_path: 输出文件路径，None则返回tensor
            return_original: 是否同时返回原始音频
            **kwargs: 生成和水印参数
            
        Returns:
            Union[torch.Tensor, str]: 生成的含水印音频tensor或文件路径（当return_original=False时）
            Dict[str, Union[torch.Tensor, str]]: 包含'original'和'watermarked'键的字典（当return_original=True时）
        """
        self._ensure_model()
        self._ensure_bark_generator()
        
        self.logger.info(f"开始生成带水印音频，文本: '{prompt[:50]}...', 消息: '{message}'")
        
        # 从kwargs中提取Bark参数
        bark_config = self.config.get('bark', {})
        voice_preset = kwargs.get('voice_preset', bark_config.get('default_voice'))
        temperature = kwargs.get('temperature', bark_config.get('temperature', 0.8))
        seed = kwargs.get('seed')
        
        # 1. 使用Bark生成音频
        self.logger.info("步骤1: 生成原始音频")
        generated_audio = self.bark_generator.generate_audio(
            text=prompt,
            voice_preset=voice_preset,
            temperature=temperature,
            seed=seed
        )
        
        self.logger.info(f"音频生成完成: {generated_audio.shape}")
        
        # 2. 在生成的音频中嵌入水印
        self.logger.info("步骤2: 嵌入水印")
        watermark_strength = kwargs.get('watermark_strength', 
                                       self.config.get('watermark_strength', 1.0))
        
        watermarked_audio = self.watermark_processor.embed(
            generated_audio,
            message,
            input_sample_rate=self.config.get('sample_rate', 16000),
            alpha=watermark_strength
        )
        
        self.logger.info(f"水印嵌入完成: {watermarked_audio.shape}")
        
        # 3. 保存或返回
        if output_path:
            # 确保输出目录存在
            output_path = Path(output_path)
            FileUtils.ensure_dir(output_path.parent)
            
            # 避免文件名冲突
            watermarked_path = FileUtils.get_unique_filename(str(output_path))
            
            # 保存水印音频
            AudioIOUtils.save_audio(
                watermarked_audio, 
                watermarked_path, 
                self.config.get('sample_rate', 16000)
            )
            
            self.logger.info(f"带水印音频已保存: {watermarked_path}")
            
            # 如果需要返回原始音频，也保存原始音频
            if return_original:
                # 生成原始音频文件路径
                original_path = str(output_path).replace('.wav', '_original.wav')
                original_path = FileUtils.get_unique_filename(original_path)
                
                # 保存原始音频
                AudioIOUtils.save_audio(
                    generated_audio, 
                    original_path, 
                    self.config.get('sample_rate', 16000)
                )
                
                self.logger.info(f"原始音频已保存: {original_path}")
                
                return {
                    'original': original_path,
                    'watermarked': watermarked_path
                }
            else:
                return watermarked_path
        else:
            if return_original:
                return {
                    'original': generated_audio,
                    'watermarked': watermarked_audio
                }
            else:
                return watermarked_audio
    
    def batch_embed(self, 
                   audio_inputs: List[Union[str, torch.Tensor]],
                   messages: List[str],
                   output_dir: Optional[str] = None,
                   **kwargs) -> List[Union[torch.Tensor, str]]:
        """
        批量嵌入水印
        
        Args:
            audio_inputs: 输入音频列表
            messages: 消息列表
            output_dir: 输出目录，None则返回tensor列表
            **kwargs: 其他参数
            
        Returns:
            List: 处理结果列表
        """
        self.logger.info(f"开始批量嵌入，共 {len(audio_inputs)} 个音频")
        
        if len(messages) != len(audio_inputs):
            raise ValueError("音频和消息列表长度必须相同")
        
        results = []
        
        for i, (audio_input, message) in enumerate(zip(audio_inputs, messages)):
            try:
                self.logger.info(f"处理第 {i+1}/{len(audio_inputs)} 个音频")
                
                if output_dir:
                    # 生成输出文件名
                    if isinstance(audio_input, str):
                        input_name = Path(audio_input).stem
                    else:
                        input_name = f"audio_{i+1}"
                    
                    output_path = os.path.join(
                        output_dir, 
                        f"{input_name}_watermarked.wav"
                    )
                else:
                    output_path = None
                
                result = self.embed_watermark(
                    audio_input, message, output_path, **kwargs
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"处理第 {i+1} 个音频失败: {e}")
                results.append(None)
        
        success_count = sum(1 for r in results if r is not None)
        self.logger.info(f"批量嵌入完成: {success_count}/{len(audio_inputs)} 成功")
        
        return results
    
    def batch_extract(self,
                     audio_inputs: List[Union[str, torch.Tensor]],
                     **kwargs) -> List[Dict[str, Any]]:
        """
        批量提取水印
        
        Args:
            audio_inputs: 输入音频列表
            **kwargs: 其他参数
            
        Returns:
            List[Dict]: 提取结果列表
        """
        self.logger.info(f"开始批量提取，共 {len(audio_inputs)} 个音频")
        
        results = []
        
        for i, audio_input in enumerate(audio_inputs):
            try:
                self.logger.info(f"处理第 {i+1}/{len(audio_inputs)} 个音频")
                
                result = self.extract_watermark(audio_input, **kwargs)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"处理第 {i+1} 个音频失败: {e}")
                results.append({
                    "detected": False,
                    "message": "",
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r.get("detected", False))
        self.logger.info(f"批量提取完成: {success_count}/{len(audio_inputs)} 检测到水印")
        
        return results
    
    def evaluate_quality(self,
                        original_audio: Union[str, torch.Tensor],
                        watermarked_audio: Union[str, torch.Tensor]) -> Dict[str, float]:
        """
        评估水印对音频质量的影响
        
        Args:
            original_audio: 原始音频
            watermarked_audio: 带水印音频
            
        Returns:
            Dict[str, float]: 质量评估结果
        """
        from .utils import AudioQualityUtils
        
        # 加载音频
        if isinstance(original_audio, str):
            orig_tensor, _ = AudioIOUtils.load_audio(original_audio)
        else:
            orig_tensor = original_audio
        
        if isinstance(watermarked_audio, str):
            wm_tensor, _ = AudioIOUtils.load_audio(watermarked_audio)
        else:
            wm_tensor = watermarked_audio
        
        # 确保形状一致
        min_length = min(orig_tensor.size(-1), wm_tensor.size(-1))
        orig_tensor = orig_tensor[..., :min_length]
        wm_tensor = wm_tensor[..., :min_length]
        
        # 计算质量指标
        snr = AudioQualityUtils.calculate_snr(orig_tensor, wm_tensor)
        mse = AudioQualityUtils.calculate_mse(orig_tensor, wm_tensor)
        correlation = AudioQualityUtils.calculate_correlation(orig_tensor, wm_tensor)
        
        return {
            "snr_db": snr,
            "mse": mse,
            "correlation": correlation
        }

    def save_audio(self,
                   audio: Union[str, torch.Tensor],
                   output_path: Union[str, Path],
                   sample_rate: Optional[int] = None) -> str:
        """保存音频到指定路径，支持Tensor或已有文件路径输入"""
        output_path = Path(output_path)
        FileUtils.ensure_dir(output_path.parent)

        if isinstance(audio, torch.Tensor):
            sr = sample_rate or self.config.get('sample_rate', 16000)
            AudioIOUtils.save_audio(audio, output_path, sample_rate=sr)
            return str(output_path)

        # 若传入的是现有文件路径则直接复制
        source_path = Path(audio)
        if not source_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {source_path}")

        if source_path.resolve() == output_path.resolve():
            return str(output_path)

        import shutil
        shutil.copy2(source_path, output_path)
        return str(output_path)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "algorithm": self.algorithm,
            "config": self.config,
            "processor_initialized": self.watermark_processor is not None
        }
        
        if self.watermark_processor:
            if hasattr(self.watermark_processor, 'get_model_info'):
                info["processor_info"] = self.watermark_processor.get_model_info()
        
        return info
    
    def clear_cache(self):
        """清理内存缓存"""
        self.logger.info("清理AudioWatermark缓存...")
        
        if self.watermark_processor:
            if hasattr(self.watermark_processor, 'clear_cache'):
                self.watermark_processor.clear_cache()
        
        if self.bark_generator:
            if hasattr(self.bark_generator, 'clear_cache'):
                self.bark_generator.clear_cache()
        
        self.watermark_processor = None
        self._initialized_algorithm = None
        self.bark_generator = None
        
        self.logger.info("缓存清理完成")


# 便捷函数
def create_audio_watermark(config_path: Optional[str] = None) -> AudioWatermark:
    """
    创建音频水印处理器的便捷函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        AudioWatermark: 音频水印处理器实例
    """
    return AudioWatermark(config_path=config_path)


if __name__ == "__main__":
    # 简单测试
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("测试AudioWatermark统一接口...")
    
    try:
        # 创建音频水印处理器
        watermark_tool = create_audio_watermark()
        
        # 显示模型信息
        info = watermark_tool.get_model_info()
        print("模型信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 创建测试音频
        test_audio = torch.randn(1, 16000)  # 1秒的随机音频
        test_message = "test_message_2025"
        
        print(f"\n测试消息: '{test_message}'")
        print(f"测试音频形状: {test_audio.shape}")
        
        # 测试嵌入
        print("\n1. 测试水印嵌入...")
        watermarked = watermark_tool.embed_watermark(test_audio, test_message)
        print(f"嵌入成功，输出形状: {watermarked.shape}")
        
        # 测试提取
        print("\n2. 测试水印提取...")
        result = watermark_tool.extract_watermark(watermarked)
        print(f"提取结果: {result}")
        
        # 测试质量评估
        print("\n3. 测试质量评估...")
        quality = watermark_tool.evaluate_quality(test_audio, watermarked)
        print(f"质量评估: {quality}")
        
        # 测试文本转音频+水印 (可选)
        if HAS_BARK_GENERATOR:
            print("\n4. 测试文本转音频+水印...")
            try:
                test_prompt = "Hello, this is a test audio generated with watermark."
                generated_wm_audio = watermark_tool.generate_audio_with_watermark(
                    prompt=test_prompt,
                    message=test_message,
                    temperature=0.7,
                    seed=42
                )
                print(f"生成+嵌入成功，形状: {generated_wm_audio.shape}")
                
                # 测试从生成的音频中提取水印
                extract_result = watermark_tool.extract_watermark(generated_wm_audio)
                print(f"从生成音频提取结果: {extract_result}")
                
            except Exception as e:
                print(f"文本转音频测试失败: {e}")
        else:
            print("\n4. 跳过文本转音频测试 (Bark未安装)")
        
        # 验证
        success = result["detected"] and test_message in result["message"]
        print(f"\n验证结果: {'✅ 成功' if success else '❌ 失败'}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()