"""
AudioSeal水印封装器
提供AudioSeal水印嵌入和提取的统一接口
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, Union
import sys
import os

# 添加AudioSeal路径到系统路径
audioseal_path = os.path.join(os.path.dirname(__file__), 'audioseal', 'src')
if audioseal_path not in sys.path:
    sys.path.insert(0, audioseal_path)

try:
    from audioseal.loader import AudioSeal
    from audioseal.models import AudioSealWM, AudioSealDetector
except ImportError as e:
    print(f"警告: 无法导入AudioSeal模块: {e}")
    print("请确保AudioSeal依赖已正确安装")


class MessageEncoder:
    """消息编码器 - 将字符串消息转换为二进制位"""
    
    @staticmethod
    def string_to_bits(message: str, nbits: int = 16) -> torch.Tensor:
        """
        将字符串消息转换为固定长度的二进制位
        
        Args:
            message: 要编码的字符串消息
            nbits: 目标二进制位数
            
        Returns:
            torch.Tensor: 二进制位张量 (1, nbits)
        """
        # 将字符串转换为字节
        message_bytes = message.encode('utf-8')
        
        # 计算哈希以获得固定长度的位表示
        import hashlib
        hash_object = hashlib.sha256(message_bytes)
        hash_bytes = hash_object.digest()
        
        # 将哈希字节转换为二进制位
        bits = []
        for byte in hash_bytes:
            for i in range(8):
                bits.append((byte >> (7-i)) & 1)
        
        # 截取或填充到所需长度
        if len(bits) >= nbits:
            bits = bits[:nbits]
        else:
            # 如果位数不够，用消息的重复哈希填充
            while len(bits) < nbits:
                remaining = nbits - len(bits)
                bits.extend(bits[:min(remaining, len(bits))])
        
        return torch.tensor(bits, dtype=torch.int32).unsqueeze(0)
    
    @staticmethod
    def bits_to_string(bits: torch.Tensor, original_messages: list = None) -> str:
        """
        将二进制位转换回字符串消息
        
        Args:
            bits: 二进制位张量
            original_messages: 原始消息列表，用于验证匹配
            
        Returns:
            str: 解码的消息字符串
        """
        if bits.dim() > 1:
            bits = bits.flatten()
        
        # 转换为字节
        bit_list = bits.cpu().numpy().tolist()
        
        # 如果有原始消息列表，尝试匹配
        if original_messages:
            best_match = None
            best_score = -1
            
            for msg in original_messages:
                encoded_bits = MessageEncoder.string_to_bits(msg, len(bit_list))
                # 计算匹配度
                matches = torch.sum(bits == encoded_bits.flatten()).item()
                score = matches / len(bit_list)
                
                if score > best_score:
                    best_score = score
                    best_match = msg
            
            if best_score > 0.7:  # 70%以上匹配度认为有效
                return best_match
        
        # 如果没有原始消息或匹配度低，返回位模式的字符串表示
        bit_string = ''.join(map(str, bit_list))
        return f"bits_{bit_string[:16]}..."  # 只显示前16位


class AudioSealWrapper:
    """AudioSeal水印封装器"""
    
    def __init__(self, device: Optional[str] = None, nbits: int = 16):
        """
        初始化AudioSeal封装器
        
        Args:
            device: 计算设备 ('cuda', 'cpu' 或 None 自动选择)
            nbits: 消息位数 (默认16位)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.nbits = nbits
        self.sample_rate = 16000  # AudioSeal要求的采样率
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 延迟加载模型
        self.generator = None
        self.detector = None
        
        # 消息缓存，用于提取时的匹配
        self._embedded_messages = []
        
        self.logger.info(f"AudioSealWrapper初始化完成 - 设备: {self.device}, 位数: {nbits}")
    
    def _ensure_models(self):
        """确保模型已加载"""
        if self.generator is None:
            self.logger.info("加载AudioSeal生成器模型...")
            try:
                self.generator = AudioSeal.load_generator(
                    "audioseal_wm_16bits", 
                    device=self.device,
                    nbits=self.nbits
                )
                self.logger.info("AudioSeal生成器加载成功")
            except Exception as e:
                self.logger.error(f"加载AudioSeal生成器失败: {e}")
                raise
        
        if self.detector is None:
            self.logger.info("加载AudioSeal检测器模型...")
            try:
                self.detector = AudioSeal.load_detector(
                    "audioseal_detector_16bits",
                    device=self.device, 
                    nbits=self.nbits
                )
                self.logger.info("AudioSeal检测器加载成功")
            except Exception as e:
                self.logger.error(f"加载AudioSeal检测器失败: {e}")
                raise
    
    def _preprocess_audio(self, audio: torch.Tensor, 
                         input_sample_rate: int = None) -> torch.Tensor:
        """
        预处理音频：重采样到16kHz，确保正确的维度
        
        Args:
            audio: 输入音频张量
            input_sample_rate: 输入音频的采样率
            
        Returns:
            torch.Tensor: 预处理后的音频张量
        """
        # 确保是浮点类型
        if audio.dtype != torch.float32:
            audio = audio.float()
        
        # 确保值域在[-1, 1]
        if audio.abs().max() > 1.0:
            audio = audio / audio.abs().max()
        
        # 如果是多声道，转换为单声道
        if audio.dim() > 1 and audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # 确保正确的3维张量格式 (batch, channels, time)
        if audio.dim() == 1:
            # 1D音频: (time,) -> (1, 1, time)
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            # 2D音频: (channels, time) 或 (batch, time)
            if audio.size(0) == 1 or audio.size(0) > audio.size(1):
                # 如果第一维度为1或者比第二维度大，认为是(batch, time)格式
                # (batch, time) -> (batch, 1, time)
                audio = audio.unsqueeze(1)
            else:
                # 否则认为是(channels, time)格式
                # (channels, time) -> (1, channels, time)
                audio = audio.unsqueeze(0)
        
        # 重采样到16kHz (如果需要)
        if input_sample_rate and input_sample_rate != self.sample_rate:
            try:
                import torchaudio
                audio = torchaudio.functional.resample(
                    audio, input_sample_rate, self.sample_rate
                )
            except ImportError:
                self.logger.warning("torchaudio未安装，跳过重采样")
        
        return audio.to(self.device)
    
    def _restore_original_shape(self, audio: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        """
        将处理后的3D音频恢复为原始形状
        
        Args:
            audio: 处理后的3D音频张量 (batch, channels, time)
            original_shape: 原始输入的形状
            
        Returns:
            torch.Tensor: 恢复为原始形状的音频张量
        """
        if len(original_shape) == 1:
            # 原始是1D: (time,)
            return audio.squeeze(0).squeeze(0)
        elif len(original_shape) == 2:
            # 原始是2D: (channels, time) 或 (batch, time)
            return audio.squeeze(0)  # 移除batch维度，保留(channels, time)
        else:
            # 原始就是3D或更高维，保持不变
            return audio

    def embed(self, audio: torch.Tensor, message: str, 
              input_sample_rate: int = None, alpha: float = 1.0) -> torch.Tensor:
        """
        在音频中嵌入水印消息
        
        Args:
            audio: 输入音频张量
            message: 要嵌入的字符串消息
            input_sample_rate: 输入音频采样率
            alpha: 水印强度 (0.0-2.0, 默认1.0)
            
        Returns:
            torch.Tensor: 带水印的音频张量，形状与输入相同
        """
        self._ensure_models()
        
        self.logger.info(f"嵌入水印消息: '{message}'")
        
        # 保存原始形状
        original_shape = audio.shape
        
        # 预处理音频
        processed_audio = self._preprocess_audio(audio, input_sample_rate)
        
        # 编码消息
        message_bits = MessageEncoder.string_to_bits(message, self.nbits)
        message_bits = message_bits.to(self.device)
        
        # 缓存消息用于后续提取验证
        if message not in self._embedded_messages:
            self._embedded_messages.append(message)
        
        self.logger.debug(f"原始音频形状: {original_shape}, 处理后形状: {processed_audio.shape}, 消息位: {message_bits.shape}")
        
        # 嵌入水印
        try:
            watermarked_audio = self.generator(
                processed_audio,
                sample_rate=self.sample_rate,
                message=message_bits,
                alpha=alpha
            )
            
            # 恢复原始形状
            restored_audio = self._restore_original_shape(watermarked_audio, original_shape)
            
            self.logger.info(f"水印嵌入成功，输出形状: {restored_audio.shape}")
            return restored_audio
            
        except Exception as e:
            self.logger.error(f"水印嵌入失败: {e}")
            raise
    
    def extract(self, watermarked_audio: torch.Tensor,
                input_sample_rate: int = None,
                detection_threshold: float = 0.5,
                message_threshold: float = 0.5) -> Dict[str, Any]:
        """
        从音频中提取水印消息
        
        Args:
            watermarked_audio: 带水印的音频张量
            input_sample_rate: 输入音频采样率
            detection_threshold: 检测阈值
            message_threshold: 消息解码阈值
            
        Returns:
            Dict: 提取结果 {detected: bool, message: str, confidence: float, raw_bits: torch.Tensor}
        """
        self._ensure_models()
        
        self.logger.info("开始提取水印...")
        
        # 预处理音频
        processed_audio = self._preprocess_audio(watermarked_audio, input_sample_rate)
        
        try:
            # 检测水印
            detection_prob, message_bits = self.detector.detect_watermark(
                processed_audio,
                sample_rate=self.sample_rate,
                detection_threshold=detection_threshold,
                message_threshold=message_threshold
            )
            
            # 确保message_bits在CPU上进行后续处理
            if message_bits is not None and isinstance(message_bits, torch.Tensor):
                message_bits = message_bits.cpu()
            
            # 处理检测概率
            if isinstance(detection_prob, torch.Tensor):
                confidence = detection_prob.item()
            else:
                confidence = float(detection_prob)
            
            detected = confidence > detection_threshold
            
            # 解码消息
            if detected and message_bits is not None:
                decoded_message = MessageEncoder.bits_to_string(
                    message_bits, self._embedded_messages
                )
            else:
                decoded_message = ""
            
            result = {
                "detected": detected,
                "message": decoded_message,
                "confidence": confidence,
                "raw_bits": message_bits.cpu() if message_bits is not None else None,
                "detection_threshold": detection_threshold
            }
            
            self.logger.info(
                f"水印提取完成 - 检测: {detected}, 置信度: {confidence:.3f}, "
                f"消息: '{decoded_message}'"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"水印提取失败: {e}")
            return {
                "detected": False,
                "message": "",
                "confidence": 0.0,
                "raw_bits": None,
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "device": self.device,
            "sample_rate": self.sample_rate,
            "nbits": self.nbits,
            "generator_loaded": self.generator is not None,
            "detector_loaded": self.detector is not None,
            "cached_messages": len(self._embedded_messages)
        }
    
    def clear_cache(self):
        """清理内存缓存"""
        self.logger.info("清理AudioSeal缓存...")
        if self.generator is not None:
            del self.generator
            self.generator = None
        if self.detector is not None:
            del self.detector
            self.detector = None
        self._embedded_messages.clear()
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("缓存清理完成")


# 便捷函数
def create_audioseal_wrapper(device: str = None, nbits: int = 16) -> AudioSealWrapper:
    """
    创建AudioSeal封装器的便捷函数
    
    Args:
        device: 计算设备
        nbits: 消息位数
        
    Returns:
        AudioSealWrapper: 封装器实例
    """
    return AudioSealWrapper(device=device, nbits=nbits)


if __name__ == "__main__":
    # 简单测试
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("测试AudioSealWrapper...")
    
    try:
        # 创建封装器
        wrapper = create_audioseal_wrapper()
        
        # 显示模型信息
        info = wrapper.get_model_info()
        print("模型信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 创建测试音频
        test_audio = torch.randn(1, 16000)  # 1秒的随机音频
        test_message = "hello_audioseal_2025"
        
        print(f"\n测试消息: '{test_message}'")
        print(f"测试音频形状: {test_audio.shape}")
        
        # 测试嵌入
        print("\n1. 测试水印嵌入...")
        watermarked = wrapper.embed(test_audio, test_message)
        print(f"嵌入成功，输出形状: {watermarked.shape}")
        
        # 测试提取
        print("\n2. 测试水印提取...")
        result = wrapper.extract(watermarked)
        print(f"提取结果: {result}")
        
        # 验证
        success = result["detected"] and test_message in result["message"]
        print(f"\n验证结果: {'✅ 成功' if success else '❌ 失败'}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()