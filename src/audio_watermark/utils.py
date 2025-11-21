"""
音频处理工具模块
提供音频I/O、格式转换、质量评估等功能
"""

import torch
import numpy as np
import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any
import warnings

# 可选依赖导入
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False
    warnings.warn("torchaudio未安装，某些功能可能不可用")

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class AudioIOUtils:
    """音频输入输出工具类"""
    
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    
    @staticmethod
    def load_audio(file_path: Union[str, Path], 
                   target_sample_rate: Optional[int] = None,
                   mono: bool = True) -> Tuple[torch.Tensor, int]:
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            target_sample_rate: 目标采样率，None则保持原始采样率
            mono: 是否转换为单声道
            
        Returns:
            Tuple[torch.Tensor, int]: (音频张量, 采样率)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {file_path}")
        
        if file_path.suffix.lower() not in AudioIOUtils.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的音频格式: {file_path.suffix}")
        
        # 优先使用torchaudio加载
        if HAS_TORCHAUDIO:
            try:
                waveform, sample_rate = torchaudio.load(str(file_path))
                
                # 转换为单声道
                if mono and waveform.size(0) > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # 重采样
                if target_sample_rate and target_sample_rate != sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, sample_rate, target_sample_rate
                    )
                    sample_rate = target_sample_rate
                
                return waveform, sample_rate
                
            except Exception as e:
                logging.warning(f"torchaudio加载失败，尝试其他方法: {e}")
        
        # 回退到soundfile
        if HAS_SOUNDFILE:
            try:
                data, sample_rate = sf.read(str(file_path))
                
                # 转换为torch张量
                if data.ndim == 1:
                    waveform = torch.from_numpy(data).float().unsqueeze(0)
                else:
                    waveform = torch.from_numpy(data.T).float()
                    if mono and waveform.size(0) > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # 重采样（使用librosa）
                if target_sample_rate and target_sample_rate != sample_rate:
                    if HAS_LIBROSA:
                        resampled = librosa.resample(
                            waveform.numpy().squeeze(), 
                            orig_sr=sample_rate, 
                            target_sr=target_sample_rate
                        )
                        waveform = torch.from_numpy(resampled).float().unsqueeze(0)
                        sample_rate = target_sample_rate
                    else:
                        logging.warning("librosa未安装，跳过重采样")
                
                return waveform, sample_rate
                
            except Exception as e:
                logging.warning(f"soundfile加载失败: {e}")
        
        # 最后尝试librosa
        if HAS_LIBROSA:
            try:
                data, sample_rate = librosa.load(
                    str(file_path), 
                    sr=target_sample_rate, 
                    mono=mono
                )
                waveform = torch.from_numpy(data).float().unsqueeze(0)
                return waveform, sample_rate
                
            except Exception as e:
                logging.error(f"librosa加载失败: {e}")
        
        raise RuntimeError(f"无法加载音频文件: {file_path}，请安装torchaudio、soundfile或librosa")
    
    @staticmethod
    def save_audio(waveform: torch.Tensor, 
                   file_path: Union[str, Path],
                   sample_rate: int = 16000,
                   format: Optional[str] = None) -> None:
        """
        保存音频文件
        
        Args:
            waveform: 音频张量 (channels, samples)
            file_path: 输出文件路径
            sample_rate: 采样率
            format: 音频格式，None则从文件扩展名推断
        """
        file_path = Path(file_path)
        
        # 确保输出目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 推断格式
        if format is None:
            format = file_path.suffix.lower().lstrip('.')
        
        # 处理不同维度的音频张量
        if waveform.dim() == 3:
            # 3D张量 (batch, channels, time) -> (channels, time)
            if waveform.size(0) == 1:
                waveform = waveform.squeeze(0)  # 移除batch维度
            else:
                # 多batch，取第一个
                waveform = waveform[0]
        elif waveform.dim() == 1:
            # 1D张量 (time,) -> (1, time)
            waveform = waveform.unsqueeze(0)
        
        # 确保数据在正确范围内
        if waveform.abs().max() > 1.0:
            waveform = waveform / waveform.abs().max()
        
        # 优先使用torchaudio保存
        if HAS_TORCHAUDIO:
            try:
                # 确保waveform是CPU tensor
                waveform_cpu = waveform.cpu()
                torchaudio.save(str(file_path), waveform_cpu, sample_rate)
                return
            except Exception as e:
                logging.warning(f"torchaudio保存失败，尝试其他方法: {e}")
        
        # 回退到soundfile
        if HAS_SOUNDFILE:
            try:
                # soundfile需要(samples, channels)格式
                if waveform.dim() == 2:
                    data = waveform.T.numpy()
                else:
                    data = waveform.numpy()
                
                sf.write(str(file_path), data, sample_rate, format=format)
                return
            except Exception as e:
                logging.error(f"soundfile保存失败: {e}")
        
        raise RuntimeError(f"无法保存音频文件: {file_path}，请安装torchaudio或soundfile")
    
    @staticmethod
    def get_audio_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        获取音频文件信息
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            Dict: 音频信息字典
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {file_path}")
        
        info = {
            "file_path": str(file_path),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "format": file_path.suffix.lower()
        }
        
        # 尝试获取详细信息
        if HAS_TORCHAUDIO:
            try:
                metadata = torchaudio.info(str(file_path))
                info.update({
                    "sample_rate": metadata.sample_rate,
                    "num_channels": metadata.num_channels,
                    "num_frames": metadata.num_frames,
                    "duration_seconds": metadata.num_frames / metadata.sample_rate,
                    "bits_per_sample": metadata.bits_per_sample
                })
                return info
            except Exception:
                pass
        
        if HAS_SOUNDFILE:
            try:
                sf_info = sf.info(str(file_path))
                info.update({
                    "sample_rate": sf_info.samplerate,
                    "num_channels": sf_info.channels,
                    "num_frames": sf_info.frames,
                    "duration_seconds": sf_info.duration,
                    "format_name": sf_info.format
                })
                return info
            except Exception:
                pass
        
        return info


class AudioProcessingUtils:
    """音频处理工具类"""
    
    @staticmethod
    def resample(waveform: torch.Tensor, 
                 orig_sr: int, 
                 target_sr: int) -> torch.Tensor:
        """
        重采样音频
        
        Args:
            waveform: 输入音频张量
            orig_sr: 原始采样率
            target_sr: 目标采样率
            
        Returns:
            torch.Tensor: 重采样后的音频张量
        """
        if orig_sr == target_sr:
            return waveform
        
        if HAS_TORCHAUDIO:
            return torchaudio.functional.resample(waveform, orig_sr, target_sr)
        elif HAS_LIBROSA:
            # 使用librosa重采样
            if waveform.dim() == 2:
                resampled_data = []
                for i in range(waveform.size(0)):
                    resampled = librosa.resample(
                        waveform[i].numpy(), 
                        orig_sr=orig_sr, 
                        target_sr=target_sr
                    )
                    resampled_data.append(torch.from_numpy(resampled))
                return torch.stack(resampled_data)
            else:
                resampled = librosa.resample(
                    waveform.numpy(), 
                    orig_sr=orig_sr, 
                    target_sr=target_sr
                )
                return torch.from_numpy(resampled)
        else:
            logging.warning("无法重采样：请安装torchaudio或librosa")
            return waveform
    
    @staticmethod
    def normalize(waveform: torch.Tensor, 
                  method: str = 'peak') -> torch.Tensor:
        """
        音频归一化
        
        Args:
            waveform: 输入音频张量
            method: 归一化方法 ('peak', 'rms')
            
        Returns:
            torch.Tensor: 归一化后的音频张量
        """
        if method == 'peak':
            # 峰值归一化
            max_val = waveform.abs().max()
            if max_val > 0:
                return waveform / max_val
            return waveform
        elif method == 'rms':
            # RMS归一化
            rms = torch.sqrt(torch.mean(waveform ** 2))
            if rms > 0:
                return waveform / rms
            return waveform
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
    
    @staticmethod
    def add_noise(waveform: torch.Tensor, 
                  noise_type: str = 'white',
                  snr_db: float = 20.0) -> torch.Tensor:
        """
        添加噪声到音频
        
        Args:
            waveform: 输入音频张量
            noise_type: 噪声类型 ('white', 'pink')
            snr_db: 信噪比(dB)
            
        Returns:
            torch.Tensor: 添加噪声后的音频张量
        """
        signal_power = torch.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        if noise_type == 'white':
            noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        elif noise_type == 'pink':
            # 简化的粉噪声生成
            white_noise = torch.randn_like(waveform)
            # 应用简单的低通滤波近似粉噪声
            noise = white_noise * torch.sqrt(noise_power)
        else:
            raise ValueError(f"不支持的噪声类型: {noise_type}")
        
        return waveform + noise


class AudioQualityUtils:
    """音频质量评估工具类"""
    
    @staticmethod
    def calculate_snr(clean: torch.Tensor, 
                      noisy: torch.Tensor) -> float:
        """
        计算信噪比(SNR)
        
        Args:
            clean: 原始音频
            noisy: 带噪音频
            
        Returns:
            float: SNR值(dB)
        """
        # 确保张量在同一设备上
        if clean.device != noisy.device:
            noisy = noisy.to(clean.device)
        
        signal_power = torch.mean(clean ** 2)
        noise_power = torch.mean((noisy - clean) ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * torch.log10(signal_power / noise_power)
        return snr.item()
    
    @staticmethod
    def calculate_mse(signal1: torch.Tensor, 
                      signal2: torch.Tensor) -> float:
        """
        计算均方误差(MSE)
        
        Args:
            signal1: 第一个音频信号
            signal2: 第二个音频信号
            
        Returns:
            float: MSE值
        """
        # 确保张量在同一设备上
        if signal1.device != signal2.device:
            signal2 = signal2.to(signal1.device)
        
        return torch.mean((signal1 - signal2) ** 2).item()
    
    @staticmethod
    def calculate_correlation(signal1: torch.Tensor, 
                            signal2: torch.Tensor) -> float:
        """
        计算相关性
        
        Args:
            signal1: 第一个音频信号
            signal2: 第二个音频信号
            
        Returns:
            float: 相关系数
        """
        # 确保张量在同一设备上
        if signal1.device != signal2.device:
            signal2 = signal2.to(signal1.device)
        
        # 展平信号
        s1 = signal1.flatten()
        s2 = signal2.flatten()
        
        # 计算相关系数
        correlation = torch.corrcoef(torch.stack([s1, s2]))[0, 1]
        return correlation.item()


class AudioVisualizationUtils:
    """音频可视化工具类"""
    
    @staticmethod
    def plot_waveform(waveform: torch.Tensor, 
                      sample_rate: int,
                      title: str = "Waveform",
                      save_path: Optional[str] = None) -> None:
        """
        绘制波形图
        
        Args:
            waveform: 音频张量
            sample_rate: 采样率
            title: 图标题
            save_path: 保存路径，None则显示
        """
        if not HAS_MATPLOTLIB:
            logging.warning("matplotlib未安装，无法绘制波形图")
            return
        
        # 转换为numpy数组
        if waveform.dim() > 1:
            audio_data = waveform[0].numpy()  # 使用第一个声道
        else:
            audio_data = waveform.numpy()
        
        # 创建时间轴
        time_axis = np.arange(len(audio_data)) / sample_rate
        
        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, audio_data)
        plt.title(title)
        plt.xlabel("时间 (秒)")
        plt.ylabel("幅度")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_spectrogram(waveform: torch.Tensor,
                        sample_rate: int,
                        title: str = "Spectrogram",
                        save_path: Optional[str] = None) -> None:
        """
        绘制频谱图
        
        Args:
            waveform: 音频张量
            sample_rate: 采样率
            title: 图标题
            save_path: 保存路径，None则显示
        """
        if not HAS_MATPLOTLIB:
            logging.warning("matplotlib未安装，无法绘制频谱图")
            return
        
        # 转换为numpy数组
        if waveform.dim() > 1:
            audio_data = waveform[0].numpy()
        else:
            audio_data = waveform.numpy()
        
        plt.figure(figsize=(12, 6))
        plt.specgram(audio_data, Fs=sample_rate, cmap='viridis')
        plt.title(title)
        plt.xlabel("时间 (秒)")
        plt.ylabel("频率 (Hz)")
        plt.colorbar(label="功率 (dB)")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class FileUtils:
    """文件处理工具类"""
    
    @staticmethod
    def ensure_dir(directory: Union[str, Path]) -> None:
        """确保目录存在"""
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_unique_filename(file_path: Union[str, Path]) -> str:
        """获取唯一的文件名（避免重名）"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return str(file_path)
        
        base = file_path.stem
        suffix = file_path.suffix
        parent = file_path.parent
        
        counter = 1
        while True:
            new_name = f"{base}_{counter}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return str(new_path)
            counter += 1
    
    @staticmethod
    def get_file_size_mb(file_path: Union[str, Path]) -> float:
        """获取文件大小(MB)"""
        return Path(file_path).stat().st_size / (1024 * 1024)


# 便捷函数
def load_audio_simple(file_path: str, 
                      target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    简化的音频加载函数
    
    Args:
        file_path: 音频文件路径
        target_sr: 目标采样率
        
    Returns:
        Tuple[torch.Tensor, int]: (音频张量, 采样率)
    """
    return AudioIOUtils.load_audio(file_path, target_sr, mono=True)


def save_audio_simple(waveform: torch.Tensor, 
                      file_path: str,
                      sample_rate: int = 16000) -> None:
    """
    简化的音频保存函数
    
    Args:
        waveform: 音频张量
        file_path: 输出文件路径
        sample_rate: 采样率
    """
    AudioIOUtils.save_audio(waveform, file_path, sample_rate)


if __name__ == "__main__":
    # 简单测试
    print("测试音频处理工具...")
    
    # 测试基础功能
    print("\n1. 测试张量处理...")
    test_audio = torch.randn(1, 16000)  # 1秒音频
    print(f"原始音频形状: {test_audio.shape}")
    
    # 测试归一化
    normalized = AudioProcessingUtils.normalize(test_audio)
    print(f"归一化后最大值: {normalized.abs().max():.3f}")
    
    # 测试重采样
    if HAS_TORCHAUDIO:
        resampled = AudioProcessingUtils.resample(test_audio, 16000, 8000)
        print(f"重采样后形状: {resampled.shape}")
    
    # 测试噪声添加
    noisy = AudioProcessingUtils.add_noise(test_audio, snr_db=10)
    snr = AudioQualityUtils.calculate_snr(test_audio, noisy)
    print(f"添加噪声后SNR: {snr:.2f} dB")
    
    print("\n✅ 音频处理工具测试完成")