"""
视频水印工具函数集合
包含视频I/O、格式转换、性能监控等实用工具
"""

import os
import time
import torch
import numpy as np
import logging
from typing import Union, Optional, Tuple, Dict, Any
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Video I/O functions will be limited.")

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    logging.warning("ffmpeg-python not available. Some video processing functions will be disabled.")


class VideoIOUtils:
    """视频输入输出工具类"""
    
    @staticmethod
    def read_video_frames(video_path: str, max_frames: Optional[int] = None) -> torch.Tensor:
        """
        读取视频文件为tensor
        
        Args:
            video_path: 视频文件路径
            max_frames: 最大帧数限制
            
        Returns:
            torch.Tensor: 视频tensor，形状为 (frames, channels, height, width)，值域[0, 1]
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for video reading. Install with: pip install opencv-python")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # BGR转RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_count += 1
                
                if max_frames and frame_count >= max_frames:
                    break
                    
        finally:
            cap.release()
        
        if not frames:
            raise ValueError(f"No frames could be read from: {video_path}")
        
        # 转换为tensor
        frames_array = np.array(frames)  # (frames, height, width, channels)
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()  # (frames, channels, height, width)
        frames_tensor = frames_tensor / 255.0  # 归一化到[0, 1]
        
        return frames_tensor
    
    @staticmethod
    def save_video_tensor(video_tensor: torch.Tensor, output_path: str, fps: int = 24) -> str:
        """
        保存视频tensor为文件
        
        Args:
            video_tensor: 视频tensor，形状为 (frames, channels, height, width)，值域[0, 1]
            output_path: 输出文件路径
            fps: 帧率
            
        Returns:
            str: 输出文件路径
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for video writing. Install with: pip install opencv-python")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 转换tensor为numpy array
        if isinstance(video_tensor, torch.Tensor):
            video_array = video_tensor.detach().cpu().numpy()
        else:
            video_array = video_tensor
        
        # 确保值域在[0, 1]
        video_array = np.clip(video_array, 0, 1)
        
        # 转换为uint8并调整维度
        video_array = (video_array * 255).astype(np.uint8)
        video_array = video_array.transpose(0, 2, 3, 1)  # (frames, height, width, channels)
        
        # 获取视频参数
        frames, height, width, channels = video_array.shape
        
        # 设置编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            for frame in video_array:
                if channels == 3:
                    # RGB转BGR（OpenCV使用BGR）
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                else:
                    out.write(frame)
        finally:
            out.release()
        
        return output_path
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """
        获取视频文件信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            Dict[str, Any]: 视频信息字典
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for video info reading.")
        
        cap = cv2.VideoCapture(video_path)
        
        try:
            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
            }
        finally:
            cap.release()
        
        return info


class TensorUtils:
    """Tensor处理工具类"""
    
    @staticmethod
    def normalize_video_tensor(video_tensor: torch.Tensor) -> torch.Tensor:
        """
        标准化视频tensor到[0, 1]范围
        
        Args:
            video_tensor: 输入视频tensor
            
        Returns:
            torch.Tensor: 标准化后的tensor
        """
        # 获取最小值和最大值
        min_val = video_tensor.min()
        max_val = video_tensor.max()
        
        # 如果已经在[0, 1]范围内，直接返回
        if min_val >= 0 and max_val <= 1:
            return video_tensor
        
        # 标准化到[0, 1]
        normalized = (video_tensor - min_val) / (max_val - min_val + 1e-8)
        return normalized
    
    @staticmethod
    def resize_video_tensor(
        video_tensor: torch.Tensor, 
        target_size: Tuple[int, int],
        mode: str = 'bilinear'
    ) -> torch.Tensor:
        """
        调整视频tensor尺寸
        
        Args:
            video_tensor: 输入视频tensor，形状为 (frames, channels, height, width)
            target_size: 目标尺寸 (height, width)
            mode: 插值模式
            
        Returns:
            torch.Tensor: 调整尺寸后的tensor
        """
        import torch.nn.functional as F
        
        frames, channels, height, width = video_tensor.shape
        target_height, target_width = target_size
        
        if height == target_height and width == target_width:
            return video_tensor
        
        # 逐帧调整尺寸
        resized_frames = []
        for i in range(frames):
            frame = video_tensor[i].unsqueeze(0)  # (1, channels, height, width)
            resized_frame = F.interpolate(
                frame, 
                size=target_size, 
                mode=mode, 
                align_corners=False if mode in ['bilinear', 'bicubic'] else None
            )
            resized_frames.append(resized_frame.squeeze(0))
        
        resized_tensor = torch.stack(resized_frames, dim=0)
        return resized_tensor
    
    @staticmethod
    def crop_video_tensor(
        video_tensor: torch.Tensor,
        crop_box: Tuple[int, int, int, int]
    ) -> torch.Tensor:
        """
        裁剪视频tensor
        
        Args:
            video_tensor: 输入视频tensor，形状为 (frames, channels, height, width)
            crop_box: 裁剪框 (x1, y1, x2, y2)
            
        Returns:
            torch.Tensor: 裁剪后的tensor
        """
        x1, y1, x2, y2 = crop_box
        cropped_tensor = video_tensor[:, :, y1:y2, x1:x2]
        return cropped_tensor


class PerformanceTimer:
    """性能计时器"""
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"{self.name} 开始...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        if exc_type is None:
            self.logger.info(f"{self.name} 完成，耗时: {elapsed:.2f}秒")
        else:
            self.logger.error(f"{self.name} 失败，耗时: {elapsed:.2f}秒")
    
    def elapsed(self) -> float:
        """获取已过时间"""
        if self.start_time is None:
            return 0.0
        current_time = self.end_time or time.time()
        return current_time - self.start_time


class MemoryMonitor:
    """内存监控器"""
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """获取GPU内存信息"""
        info = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                info[f"gpu_{i}"] = {
                    "allocated_gb": allocated,
                    "cached_gb": cached
                }
        return info
    
    @staticmethod
    def clear_gpu_cache():
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_tensor_memory_usage(tensor: torch.Tensor) -> float:
        """获取tensor内存使用量（MB）"""
        return tensor.element_size() * tensor.nelement() / (1024**2)


class VideoTranscoder:
    """视频转码工具类"""
    
    @staticmethod
    def transcode_for_browser(input_path: str, output_path: str = None, 
                            target_fps: int = 30, target_resolution: str = None,
                            quality: str = 'medium') -> str:
        """
        转码视频为浏览器友好格式 (H.264 + AAC, MP4 容器，faststart)
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径，如果为None则自动生成
            target_fps: 目标帧率
            target_resolution: 目标分辨率，格式如 "720x480" 或 None保持原分辨率
            quality: 视频质量 ('low', 'medium', 'high')
            
        Returns:
            str: 输出文件路径
        """
        if not FFMPEG_AVAILABLE:
            raise RuntimeError("ffmpeg-python is required for video transcoding. Install with: pip install ffmpeg-python")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")
        
        # 生成输出路径
        if output_path is None:
            input_dir = os.path.dirname(input_path)
            input_name = Path(input_path).stem
            output_path = os.path.join(input_dir, f"{input_name}_web_compatible.mp4")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 质量设置
        quality_settings = {
            'low': {'crf': '28', 'preset': 'fast'},
            'medium': {'crf': '23', 'preset': 'medium'},
            'high': {'crf': '18', 'preset': 'slow'}
        }
        
        settings = quality_settings.get(quality, quality_settings['medium'])
        
        try:
            # 构建 FFmpeg 流
            input_stream = ffmpeg.input(input_path)
            
            # 视频流处理
            video_args = {
                'c:v': 'libx264',          # H.264 编码
                'profile:v': 'main',       # Main profile (兼容性好)
                'level': '4.0',            # Level 4.0
                'crf': settings['crf'],    # 恒定质量因子
                'preset': settings['preset'], # 编码预设
                'r': target_fps,           # 帧率
                'pix_fmt': 'yuv420p',      # 像素格式 (兼容性最佳)
                'movflags': '+faststart'   # Web优化: moov头前置
            }
            
            # 如果指定了分辨率
            if target_resolution:
                width, height = map(int, target_resolution.split('x'))
                video_args['s'] = f'{width}x{height}'
                video_args['sws_flags'] = 'lanczos'  # 高质量缩放
            
            # 音频流处理
            audio_args = {
                'c:a': 'aac',             # AAC 编码
                'b:a': '128k',            # 128kbps 码率
                'ar': '44100'             # 44.1kHz 采样率
            }
            
            # 执行转码
            out = ffmpeg.output(input_stream, output_path, **video_args, **audio_args)
            ffmpeg.run(out, overwrite_output=True, quiet=True)
            
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = f"FFmpeg transcoding failed: {e}"
            if hasattr(e, 'stderr') and e.stderr:
                error_msg += f"\nFFmpeg stderr: {e.stderr.decode()}"
            raise RuntimeError(error_msg)
        except Exception as e:
            raise RuntimeError(f"Video transcoding failed: {str(e)}")
    
    @staticmethod
    def get_video_codec_info(video_path: str) -> Dict[str, Any]:
        """
        获取视频编码信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            Dict[str, Any]: 编码信息
        """
        if not FFMPEG_AVAILABLE:
            raise RuntimeError("ffmpeg-python is required for codec info.")
        
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            
            info = {
                'video': {
                    'codec': video_stream.get('codec_name') if video_stream else None,
                    'profile': video_stream.get('profile') if video_stream else None,
                    'width': video_stream.get('width') if video_stream else None,
                    'height': video_stream.get('height') if video_stream else None,
                    'fps': eval(video_stream.get('r_frame_rate', '0/1')) if video_stream else None
                },
                'audio': {
                    'codec': audio_stream.get('codec_name') if audio_stream else None,
                    'sample_rate': audio_stream.get('sample_rate') if audio_stream else None,
                    'channels': audio_stream.get('channels') if audio_stream else None
                },
                'duration': float(probe.get('format', {}).get('duration', 0)),
                'format': probe.get('format', {}).get('format_name'),
                'size_bytes': int(probe.get('format', {}).get('size', 0))
            }
            
            return info
            
        except Exception as e:
            raise RuntimeError(f"Failed to get codec info: {str(e)}")
    
    @staticmethod
    def is_web_compatible(video_path: str) -> bool:
        """
        检查视频是否已经是浏览器兼容格式
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            bool: 是否兼容
        """
        try:
            info = VideoTranscoder.get_video_codec_info(video_path)
            
            # 检查视频编码
            video_codec = info.get('video', {}).get('codec', '').lower()
            video_profile = info.get('video', {}).get('profile', '').lower()
            
            # 检查音频编码
            audio_codec = info.get('audio', {}).get('codec', '').lower()
            
            # 检查容器格式
            format_name = info.get('format', '').lower()
            
            # H.264 + AAC + MP4 且 profile 合适
            is_compatible = (
                video_codec == 'h264' and
                video_profile in ['baseline', 'main', 'high'] and
                audio_codec == 'aac' and
                'mp4' in format_name
            )
            
            return is_compatible
            
        except Exception:
            return False


class FileUtils:
    """文件处理工具类"""
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]):
        """确保目录存在"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_unique_filename(filepath: str) -> str:
        """获取唯一文件名（避免覆盖）"""
        path = Path(filepath)
        base = path.stem
        suffix = path.suffix
        parent = path.parent
        
        counter = 1
        while path.exists():
            new_name = f"{base}_{counter}{suffix}"
            path = parent / new_name
            counter += 1
        
        return str(path)
    
    @staticmethod
    def get_file_size_mb(filepath: str) -> float:
        """获取文件大小（MB）"""
        return os.path.getsize(filepath) / (1024**2)


def create_test_video_tensor(
    frames: int = 16,
    channels: int = 3,
    height: int = 256,
    width: int = 256,
    pattern: str = "random"
) -> torch.Tensor:
    """
    创建测试用的视频tensor
    
    Args:
        frames: 帧数
        channels: 通道数
        height: 高度
        width: 宽度
        pattern: 模式 ('random', 'gradient', 'checkerboard')
        
    Returns:
        torch.Tensor: 测试视频tensor
    """
    if pattern == "random":
        return torch.rand(frames, channels, height, width)
    
    elif pattern == "gradient":
        # 创建渐变效果
        video = torch.zeros(frames, channels, height, width)
        for f in range(frames):
            for h in range(height):
                for w in range(width):
                    video[f, :, h, w] = (f / frames + h / height + w / width) / 3
        return video
    
    elif pattern == "checkerboard":
        # 创建棋盘效果
        video = torch.zeros(frames, channels, height, width)
        for f in range(frames):
            for h in range(height):
                for w in range(width):
                    if (h // 32 + w // 32 + f // 4) % 2 == 0:
                        video[f, :, h, w] = 1.0
        return video
    
    else:
        return torch.rand(frames, channels, height, width)


# 方便的装饰器
def timing_decorator(name: Optional[str] = None):
    """计时装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            timer_name = name or f"{func.__name__}"
            with PerformanceTimer(timer_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # 测试代码
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("测试视频水印工具函数...")
    
    # 测试tensor创建
    print("\n1. 测试tensor创建")
    test_video = create_test_video_tensor(16, 3, 64, 64, "gradient")
    print(f"创建测试视频: {test_video.shape}, 范围: [{test_video.min():.3f}, {test_video.max():.3f}]")
    
    # 测试内存监控
    print("\n2. 测试内存监控")
    gpu_info = MemoryMonitor.get_gpu_memory_info()
    print(f"GPU内存信息: {gpu_info}")
    
    tensor_size = MemoryMonitor.get_tensor_memory_usage(test_video)
    print(f"测试tensor内存使用: {tensor_size:.2f} MB")
    
    # 测试性能计时
    print("\n3. 测试性能计时")
    with PerformanceTimer("测试操作") as timer:
        time.sleep(0.1)  # 模拟操作
        test_tensor = torch.rand(1000, 1000)
        result = torch.sum(test_tensor)
    
    print(f"操作结果: {result:.2f}")
    
    # 测试文件工具
    print("\n4. 测试文件工具")
    test_dir = "test_output"
    FileUtils.ensure_dir(test_dir)
    print(f"创建测试目录: {test_dir}")
    
    # 如果有OpenCV，测试视频I/O
    if CV2_AVAILABLE and len(sys.argv) > 1 and sys.argv[1] == "video_io":
        print("\n5. 测试视频I/O")
        
        # 保存测试视频
        test_output = os.path.join(test_dir, "test_video.mp4")
        VideoIOUtils.save_video_tensor(test_video, test_output, fps=8)
        print(f"✅ 保存测试视频: {test_output}")
        
        # 读取视频信息
        video_info = VideoIOUtils.get_video_info(test_output)
        print(f"视频信息: {video_info}")
        
        # 读取视频
        read_video = VideoIOUtils.read_video_frames(test_output, max_frames=10)
        print(f"读取视频: {read_video.shape}")
    
    print("✅ 工具函数测试完成")