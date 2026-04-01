"""
VideoSeal水印算法简单封装
基于现有的VideoSeal实现，提供统一的水印嵌入和提取接口
"""

import os
import sys
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path

# 添加videoseal路径到sys.path
videoseal_path = Path(__file__).parent / "videoseal"
if str(videoseal_path) not in sys.path:
    sys.path.insert(0, str(videoseal_path))

try:
    import videoseal
    # 注意：不再从videoseal.models导入Videoseal类
    # 而是使用videoseal.load()函数来加载模型
    VIDEOSEAL_AVAILABLE = True
except ImportError as e:
    VIDEOSEAL_AVAILABLE = False
    logging.warning(f"VideoSeal not available: {e}. Please check the videoseal directory.")


class VideoSealWrapper:
    """VideoSeal水印算法包装器"""
    
    def __init__(self, device: Optional[str] = None, model_card: str = "videoseal_1.0"):
        """
        初始化VideoSeal包装器

        Args:
            device: 计算设备 ('cuda', 'cpu', 或None自动选择)
            model_card: VideoSeal模型卡名称
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_card = model_card
        self.model = None
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 检查依赖
        if not VIDEOSEAL_AVAILABLE:
            raise ImportError(
                "VideoSeal not available. Please ensure the videoseal directory exists and is properly configured."
            )
    
    def _load_model(self):
        """延迟加载VideoSeal模型"""
        if self.model is not None:
            return
        
        self.logger.info("正在加载VideoSeal模型...")
        
        try:
            # 切换到videoseal目录以确保正确的路径
            current_dir = os.getcwd()
            videoseal_dir = Path(__file__).parent / "videoseal"
            
            os.chdir(videoseal_dir)
            try:
                # 使用videoseal的默认模型加载方式（指定模型卡）
                self.model = videoseal.load(self.model_card)
            finally:
                # 恢复工作目录
                os.chdir(current_dir)
            self.model.eval()
            self.model.to(self.device)
            
            # 如果有编译支持，启用编译优化
            if hasattr(self.model, 'compile'):
                try:
                    self.model.compile()
                except Exception as e:
                    self.logger.warning(f"模型编译失败，将使用未编译版本: {e}")
            
            self.logger.info(f"VideoSeal模型加载完成，设备: {self.device}")
            
        except Exception as e:
            self.logger.error(f"加载VideoSeal模型失败: {e}")
            raise RuntimeError(f"Failed to load VideoSeal model: {e}")
    
    def _string_to_bits(self, message: str) -> torch.Tensor:
        """
        将字符串转换为VideoSeal兼容的bit tensor
        
        Args:
            message: 输入消息字符串
            
        Returns:
            torch.Tensor: bit tensor，形状为 (1, n_bits)
        """
        # 将字符串转换为bytes
        message_bytes = message.encode('utf-8')
        
        # 转换为bit array
        bit_array = []
        for byte in message_bytes:
            # 将每个字节转换为8个bit
            for i in range(8):
                bit_array.append((byte >> i) & 1)
        
        # 如果bits太少，填充到最小长度（比如64 bits）
        min_bits = 64
        while len(bit_array) < min_bits:
            bit_array.append(0)
        
        # VideoSeal通常使用256-bit消息，截断或填充到256
        target_bits = 256
        if len(bit_array) > target_bits:
            bit_array = bit_array[:target_bits]
        else:
            while len(bit_array) < target_bits:
                bit_array.append(0)
        
        # 转换为tensor
        bits_tensor = torch.tensor(bit_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        return bits_tensor
    
    def _bits_to_string(self, bits_tensor: torch.Tensor) -> str:
        """
        将bit tensor转换回字符串
        
        Args:
            bits_tensor: bit tensor
            
        Returns:
            str: 解码的字符串
        """
        # 转换为numpy array并取整
        if isinstance(bits_tensor, torch.Tensor):
            bits_array = (bits_tensor > 0.5).cpu().numpy().astype(int).flatten()
        else:
            bits_array = (bits_tensor > 0.5).astype(int).flatten()
        
        # 按8个bit为一组转换为字节
        bytes_list = []
        for i in range(0, len(bits_array), 8):
            if i + 8 <= len(bits_array):
                byte_bits = bits_array[i:i+8]
                # 转换bit array为byte值
                byte_val = 0
                for j, bit in enumerate(byte_bits):
                    byte_val += bit * (2 ** j)
                
                if byte_val > 0:  # 忽略0字节（padding）
                    bytes_list.append(byte_val)
        
        # 转换为字符串
        try:
            message = bytes(bytes_list).decode('utf-8', errors='ignore')
            if message:
                # 移除尾部的控制字符和填充内容（常见的\x00、\x01等）
                control_chars = ''.join(chr(i) for i in range(0, 32))
                message = message.rstrip(control_chars)
            return message
        except Exception as e:
            self.logger.warning(f"字符串解码失败: {e}")
            return ""
    
    def embed_watermark(
        self,
        video_tensor: torch.Tensor,
        message: str,
        is_video: bool = True,
        lowres_attenuation: bool = True
    ) -> torch.Tensor:
        """
        在视频tensor中嵌入水印
        
        Args:
            video_tensor: 输入视频tensor，形状为 (frames, channels, height, width)，值域[0, 1]
            message: 要嵌入的消息字符串
            is_video: 是否为视频（True）还是图像序列（False）
            lowres_attenuation: 是否启用低分辨率衰减
            
        Returns:
            torch.Tensor: 带水印的视频tensor
        """
        self._load_model()
        
        self.logger.info(f"开始嵌入水印: '{message}'")
        self.logger.info(f"视频tensor形状: {video_tensor.shape}")
        
        # 确保tensor在正确的设备上
        video_tensor = video_tensor.to(self.device)
        
        # 将消息转换为bits
        message_bits = self._string_to_bits(message)
        
        try:
            with torch.no_grad():
                # 使用VideoSeal嵌入水印
                outputs = self.model.embed(
                    video_tensor,
                    msgs=message_bits,
                    is_video=is_video,
                    lowres_attenuation=lowres_attenuation
                )
                
                watermarked_video = outputs["imgs_w"]
                
                self.logger.info(f"水印嵌入完成: {watermarked_video.shape}")
                return watermarked_video
                
        except Exception as e:
            self.logger.error(f"水印嵌入失败: {e}")
            raise RuntimeError(f"Failed to embed watermark: {e}")
    
    def extract_watermark(
        self,
        watermarked_video: torch.Tensor,
        is_video: bool = True,
        chunk_size: int = 16
    ) -> Dict[str, Any]:
        """
        从带水印的视频中提取水印
        
        Args:
            watermarked_video: 带水印的视频tensor，形状为 (frames, channels, height, width)
            is_video: 是否为视频（True）还是图像序列（False）
            
        Returns:
            Dict[str, Any]: 提取结果，包含detected、message、confidence等字段
        """
        self._load_model()
        
        self.logger.info(f"开始提取水印，视频形状: {watermarked_video.shape}")
        
        # 确保tensor在正确的设备上
        watermarked_video = watermarked_video.to(self.device)
        
        try:
            with torch.no_grad():
                # 实现分块处理逻辑，对齐inference_streaming.py
                num_frames = watermarked_video.shape[0]
                
                if num_frames <= chunk_size:
                    # 如果帧数不足chunk_size，直接处理
                    outputs = self.model.detect(watermarked_video, is_video=is_video)
                    preds = outputs["preds"]
                    
                    # 处理预测结果，排除第一个bit（可能用于检测）
                    if len(preds.shape) > 1:
                        bits_pred = preds[0, 1:]
                    else:
                        bits_pred = preds[1:]
                else:
                    # 分块处理，与inference_streaming.py对齐
                    self.logger.info(f"使用分块处理: {num_frames}帧，chunk_size={chunk_size}")
                    
                    soft_msgs = []
                    num_chunks = (num_frames + chunk_size - 1) // chunk_size  # 向上取整
                    
                    for i in range(0, num_frames, chunk_size):
                        end_idx = min(i + chunk_size, num_frames)
                        chunk = watermarked_video[i:end_idx]
                        
                        # 对每个chunk进行检测（完全对齐detect_video_clip函数）
                        chunk_outputs = self.model.detect(chunk, is_video=is_video)
                        chunk_preds = chunk_outputs["preds"]
                        
                        # 排除第一个bit（完全对齐inference_streaming.py）
                        output_bits = chunk_preds[:, 1:]  # 保持batch维度，形状类似[frames_in_chunk, 255]
                        
                        soft_msgs.append(output_bits)
                    
                    # 拼接所有chunk的结果并取平均（关键步骤）
                    if soft_msgs:
                        soft_msgs_tensor = torch.cat(soft_msgs, dim=0)
                        bits_pred = soft_msgs_tensor.mean(dim=0)  # 跨chunk平均，这是提高准确率的关键
                        self.logger.info(f"分块处理完成: {len(soft_msgs)}个chunk，平均后bits形状: {bits_pred.shape}")
                    else:
                        # 备用处理
                        outputs = self.model.detect(watermarked_video, is_video=is_video)
                        preds = outputs["preds"]
                        if len(preds.shape) > 1:
                            bits_pred = preds[0, 1:]
                        else:
                            bits_pred = preds[1:]
                
                # 计算置信度（软预测的平均值）
                confidence = torch.mean(torch.abs(bits_pred - 0.5)).item() * 2  # 转换到[0,1]范围
                
                # 判断是否检测到水印（基于置信度阈值）
                # 参考inference_streaming.py，使用更宽松的阈值或基于bit准确率
                detection_threshold = 0.05  # 降低阈值，因为分块平均后置信度计算更准确
                detected = confidence > detection_threshold
                
                # 将bits转换为消息字符串
                extracted_message = ""
                if detected:
                    try:
                        # 扩展bits到256长度（VideoSeal标准）
                        if len(bits_pred) < 256:
                            # 填充0
                            padded_bits = torch.zeros(256, device=bits_pred.device)
                            padded_bits[:len(bits_pred)] = bits_pred
                            bits_pred = padded_bits
                        
                        extracted_message = self._bits_to_string(bits_pred)
                    except Exception as e:
                        self.logger.warning(f"消息解码失败: {e}")
                        extracted_message = ""
                
                result = {
                    "detected": detected,
                    "message": extracted_message,
                    "confidence": confidence,
                    "raw_preds": bits_pred.cpu().numpy(),
                    "detection_threshold": detection_threshold
                }
                
                self.logger.info(
                    f"水印提取完成 - 检测: {detected}, 置信度: {confidence:.3f}, "
                    f"消息: '{extracted_message}'"
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"水印提取失败: {e}")
            return {
                "detected": False,
                "message": "",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_random_message_bits(self) -> torch.Tensor:
        """获取随机消息bits（用于测试）"""
        self._load_model()
        return self.model.get_random_msg()
    
    def calculate_bit_accuracy(self, original_bits: torch.Tensor, extracted_bits: torch.Tensor) -> float:
        """计算bit准确率"""
        if VIDEOSEAL_AVAILABLE:
            return bit_accuracy(extracted_bits, original_bits).item()
        else:
            # 简单的准确率计算
            return torch.mean((original_bits > 0.5) == (extracted_bits > 0.5)).item()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "device": self.device,
            "model_loaded": self.model is not None,
            "videoseal_available": VIDEOSEAL_AVAILABLE
        }
        
        if self.model is not None:
            info.update({
                "model_type": type(self.model).__name__,
                "device_actual": next(self.model.parameters()).device if hasattr(self.model, 'parameters') else 'unknown'
            })
        
        return info
    
    def clear_model(self):
        """清理模型以释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("VideoSeal模型已清理")


# 方便的工具函数
def create_videoseal_wrapper(device: Optional[str] = None) -> VideoSealWrapper:
    """
    创建VideoSeal包装器的快捷函数
    
    Args:
        device: 计算设备
        
    Returns:
        VideoSealWrapper: 包装器实例
    """
    return VideoSealWrapper(device)


if __name__ == "__main__":
    # 测试代码
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("测试VideoSealWrapper...")
    
    try:
        wrapper = create_videoseal_wrapper()
        
        # 显示模型信息
        info = wrapper.get_model_info()
        print("VideoSeal信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 如果命令行参数包含test，进行实际测试
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            print("\n开始水印测试...")
            
            # 创建测试视频tensor（增加帧数以测试分块处理）
            test_video = torch.rand(32, 3, 256, 256)  # 32帧，3通道，256x256
            
            print(f"测试视频形状: {test_video.shape}")
            
            # 测试1：使用随机bit消息（对齐inference_streaming.py）
            print("\n=== 测试1：随机bit消息 ===")
            wrapper._load_model()
            random_bits = wrapper.model.get_random_msg()
            print(f"随机bits形状: {random_bits.shape}")
            
            # 使用VideoSeal原生的embed/detect方式
            with torch.no_grad():
                outputs = wrapper.model.embed(test_video, msgs=random_bits, is_video=True, lowres_attenuation=True)
                watermarked_tensor = outputs["imgs_w"]
                
                # 测试A：不分块处理（原始方式）
                print("\n-- 测试A：不分块处理 --")
                extract_result_no_chunk = wrapper.extract_watermark(watermarked_tensor, chunk_size=999)  # 大于帧数
                original_bits = random_bits[0, 1:]
                extracted_bits_no_chunk = torch.tensor(extract_result_no_chunk['raw_preds'])[:255]
                bit_acc_no_chunk = wrapper.calculate_bit_accuracy(original_bits, extracted_bits_no_chunk) * 100
                
                print(f"  检测结果: {extract_result_no_chunk['detected']}")
                print(f"  置信度: {extract_result_no_chunk['confidence']:.3f}")
                print(f"  Bit准确率: {bit_acc_no_chunk:.1f}%")
                
                # 测试B：分块处理（优化方式）
                print("\n-- 测试B：分块处理(chunk_size=16) --")
                extract_result_chunk = wrapper.extract_watermark(watermarked_tensor, chunk_size=16)
                extracted_bits_chunk = torch.tensor(extract_result_chunk['raw_preds'])[:255]
                bit_acc_chunk = wrapper.calculate_bit_accuracy(original_bits, extracted_bits_chunk) * 100
                
                print(f"  检测结果: {extract_result_chunk['detected']}")
                print(f"  置信度: {extract_result_chunk['confidence']:.3f}")
                print(f"  Bit准确率: {bit_acc_chunk:.1f}%")
                
                print(f"\n📊 对比结果:")
                print(f"  置信度提升: {extract_result_chunk['confidence'] - extract_result_no_chunk['confidence']:.3f}")
                print(f"  准确率提升: {bit_acc_chunk - bit_acc_no_chunk:.1f}%")
            
            # 测试2：字符串消息
            print("\n=== 测试2：字符串消息 ===")
            test_message = "test_videoseal"
            print(f"测试消息: '{test_message}'")
            
            # 嵌入水印
            watermarked_video = wrapper.embed_watermark(test_video, test_message)
            print(f"✅ 水印嵌入完成: {watermarked_video.shape}")
            
            # 提取水印
            result = wrapper.extract_watermark(watermarked_video)
            print(f"✅ 水印提取完成:")
            print(f"  检测结果: {result['detected']}")
            print(f"  提取消息: '{result['message']}'")
            print(f"  置信度: {result['confidence']:.3f}")
            print(f"  验证成功: {result['message'] == test_message}")
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()