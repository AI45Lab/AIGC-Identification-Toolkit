"""
HunyuanVideo文生视频生成器
基于腾讯HunyuanVideo模型的文本到视频生成功能
"""

import os
import logging
import torch
import numpy as np
from typing import Optional, Union, Dict, Any
from pathlib import Path

from .model_manager import ModelManager

# 尝试导入diffusers相关模块
try:
    from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
    from diffusers.utils import export_to_video
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("diffusers not available. Please install with: pip install diffusers")


class HunyuanVideoGenerator:
    """HunyuanVideo文生视频生成器"""
    
    def __init__(self, model_manager: ModelManager, device: Optional[str] = None):
        """
        初始化HunyuanVideo生成器
        
        Args:
            model_manager: 模型管理器实例
            device: 计算设备 ('cuda', 'cpu', 或None自动选择)
        """
        self.model_manager = model_manager
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = None
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 检查依赖
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers is required for HunyuanVideo generation. "
                "Install with: pip install diffusers torch torchvision"
            )
    
    def _load_pipeline(self, allow_download: bool = True):
        """延迟加载HunyuanVideo管道"""
        if self.pipeline is not None:
            return
        
        self.logger.info("正在加载HunyuanVideo管道...")
        
        try:
            # 设置镜像环境变量
            original_endpoint = os.environ.get('HF_ENDPOINT')
            if not original_endpoint:
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                self.logger.info("设置HF_ENDPOINT为镜像站点: https://hf-mirror.com")
            
            # 仅使用本地快照路径按工作脚本方式加载
            try:
                local_model_path = self.model_manager.ensure_hunyuan_model(allow_download=False)
            except Exception as e:
                raise RuntimeError(f"未找到本地HunyuanVideo模型，请先下载: {e}")

            self.logger.info(f"从本地快照加载HunyuanVideo: {local_model_path}")

            if self.device == 'cuda':
                transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                    local_model_path,
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                    local_files_only=True
                )
                self.pipeline = HunyuanVideoPipeline.from_pretrained(
                    local_model_path,
                    transformer=transformer,
                    torch_dtype=torch.float16,
                    local_files_only=True
                )
            else:
                transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                    local_model_path,
                    subfolder="transformer",
                    torch_dtype=torch.float32,
                    local_files_only=True
                )
                self.pipeline = HunyuanVideoPipeline.from_pretrained(
                    local_model_path,
                    transformer=transformer,
                    torch_dtype=torch.float32,
                    local_files_only=True
                )

            # 按成功脚本的方式进行内存优化
            if hasattr(self.pipeline, 'vae') and hasattr(self.pipeline.vae, 'enable_tiling'):
                self.pipeline.vae.enable_tiling()
                self.logger.info("启用VAE tiling")
            if self.device == 'cuda' and hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
                self.logger.info("启用模型CPU offload")
            
            self.logger.info(f"HunyuanVideo管道加载完成，设备: {self.device}")
            
        except Exception as e:
            self.logger.error(f"加载HunyuanVideo管道失败: {e}")
            raise RuntimeError(f"Failed to load HunyuanVideo pipeline: {e}")
    
    def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: int = 49,
        height: int = 720,
        width: int = 1280,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        seed: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Union[torch.Tensor, str]:
        """
        生成视频
        
        Args:
            prompt: 文本提示词
            negative_prompt: 负向提示词
            num_frames: 视频帧数 (必须是4*k+1的形式，如49、129等)
            height: 视频高度 (默认720p)
            width: 视频宽度 (默认1280)
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            seed: 随机种子
            output_path: 输出视频路径，如果None则返回tensor
            
        Returns:
            torch.Tensor or str: 视频tensor或输出文件路径
        """
        self._load_pipeline(allow_download=True)  # 允许自动下载
        
        self.logger.info(f"开始生成视频: '{prompt[:50]}...'")
        self.logger.info(f"参数: {num_frames}帧, {height}x{width}, {num_inference_steps}步")
        
        # 验证num_frames格式（必须是4*k+1）
        if (num_frames - 1) % 4 != 0:
            corrected_frames = ((num_frames - 1) // 4) * 4 + 1
            self.logger.warning(f"num_frames={num_frames}不符合4*k+1格式，自动修正为{corrected_frames}")
            num_frames = corrected_frames
        
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        try:
            # 生成视频（带OOM自适应重试）
            attempt = 0
            max_attempts = 3
            current_params = {
                'num_frames': num_frames,
                'height': height,
                'width': width,
                'num_inference_steps': num_inference_steps
            }

            while True:
                try:
                    with torch.no_grad():
                        result = self.pipeline(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_frames=current_params['num_frames'],
                            height=current_params['height'],
                            width=current_params['width'],
                            num_inference_steps=current_params['num_inference_steps'],
                            guidance_scale=guidance_scale,
                            generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
                        )
                    break
                except RuntimeError as re:
                    # 捕获CUDA OOM并自适应降低参数重试
                    message = str(re)
                    if ('CUDA out of memory' in message or 'out of memory' in message) and attempt < max_attempts - 1:
                        self.logger.warning(f"检测到CUDA OOM，进行自适应重试: {message}")
                        attempt += 1
                        # 清理缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # 降低分辨率（减半，但不低于256）
                        current_params['height'] = max(256, (current_params['height'] // 2))
                        current_params['width'] = max(256, (current_params['width'] // 2))
                        # 降低帧数到接近一半，并保持4*k+1格式
                        reduced_frames = max(9, current_params['num_frames'] // 2)
                        if (reduced_frames - 1) % 4 != 0:
                            reduced_frames = ((reduced_frames - 1) // 4) * 4 + 1
                        current_params['num_frames'] = max(9, reduced_frames)
                        # 降低推理步数
                        current_params['num_inference_steps'] = max(8, current_params['num_inference_steps'] - 5)
                        self.logger.info(
                            f"重试参数 -> frames: {current_params['num_frames']}, "
                            f"size: {current_params['height']}x{current_params['width']}, "
                            f"steps: {current_params['num_inference_steps']}"
                        )
                        continue
                    raise

            # 跳出重试循环后，统一解析输出结果，确保video_frames已赋值
            if hasattr(result, 'frames') and result.frames is not None:
                video_frames = result.frames[0]  # 标准格式
            elif hasattr(result, 'videos') and result.videos is not None:
                video_frames = result.videos[0]  # 另一种可能的格式
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                video_frames = result[0]  # 直接返回列表
            else:
                # 如果都不是，尝试直接使用result
                video_frames = result

            self.logger.info(f"管道输出类型: {type(result)}")
            if hasattr(result, '__dict__'):
                self.logger.info(f"管道输出属性: {list(result.__dict__.keys())}")

            # 详细检查输出结构
            if hasattr(result, 'frames'):
                self.logger.info(f"result.frames 类型: {type(result.frames)}")
                if result.frames is not None:
                    self.logger.info(f"result.frames 长度: {len(result.frames) if hasattr(result.frames, '__len__') else 'N/A'}")
                    if hasattr(result.frames, '__len__') and len(result.frames) > 0:
                        self.logger.info(f"result.frames[0] 类型: {type(result.frames[0])}")
                        if isinstance(result.frames[0], list) and len(result.frames[0]) > 0:
                            self.logger.info(f"result.frames[0][0] 类型: {type(result.frames[0][0])}")
                            if hasattr(result.frames[0][0], 'size'):
                                self.logger.info(f"第一帧大小: {result.frames[0][0].size}")
                                # 检查第一帧的像素值
                                import numpy as np
                                frame_array = np.array(result.frames[0][0])
                                self.logger.info(f"第一帧数据范围: min={frame_array.min():.3f}, max={frame_array.max():.3f}")
            
            self.logger.info(f"视频生成完成: {type(video_frames)} - {getattr(video_frames, 'shape', 'no shape attr')}")
            
            # 如果指定输出路径，保存视频文件
            if output_path:
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 使用diffusers的export_to_video保存 (官方推荐fps=15)
                export_to_video(video_frames, output_path, fps=15)
                
                self.logger.info(f"视频已保存到: {output_path}")
                return output_path
            else:
                # 返回video tensor (numpy array)
                return video_frames
                
        except Exception as e:
            # 如果仍然OOM且是CUDA设备，尝试CPU回退一次
            message = str(e)
            if self.device != 'cpu' and ('CUDA out of memory' in message or 'out of memory' in message):
                try:
                    self.logger.warning("持续OOM，尝试切换到CPU并以更小参数重试一次")
                    # 切换到CPU
                    self.pipeline = self.pipeline.to('cpu')
                    self.device = 'cpu'
                    # 进一步降低参数
                    retry_frames = max(9, ((num_frames // 2) // 4) * 4 + 1)
                    retry_height = max(256, height // 2)
                    retry_width = max(256, width // 2)
                    retry_steps = max(8, num_inference_steps - 10)
                    with torch.no_grad():
                        result = self.pipeline(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_frames=retry_frames,
                            height=retry_height,
                            width=retry_width,
                            num_inference_steps=retry_steps,
                            guidance_scale=guidance_scale,
                            generator=torch.Generator(device='cpu').manual_seed(seed) if seed else None
                        )
                    # 处理输出
                    if hasattr(result, 'frames') and result.frames is not None:
                        video_frames = result.frames[0]
                    elif hasattr(result, 'videos') and result.videos is not None:
                        video_frames = result.videos[0]
                    elif isinstance(result, (list, tuple)) and len(result) > 0:
                        video_frames = result[0]
                    else:
                        video_frames = result
                    self.logger.info(f"CPU回退成功: frames={retry_frames}, size={retry_height}x{retry_width}, steps={retry_steps}")
                    if output_path:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        export_to_video(video_frames, output_path, fps=15)
                        self.logger.info(f"视频已保存到: {output_path}")
                        return output_path
                    return video_frames
                except Exception as e_cpu:
                    self.logger.error(f"CPU回退仍失败: {e_cpu}")
                    # 继续抛出原始错误
            self.logger.error(f"视频生成失败: {e}")
            import traceback
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to generate video: {e}")
    
    def generate_video_tensor(
        self,
        prompt: str,
        **kwargs
    ) -> torch.Tensor:
        """
        生成视频tensor (用于后续水印处理)
        
        Args:
            prompt: 文本提示词
            **kwargs: 其他生成参数
            
        Returns:
            torch.Tensor: 视频tensor，形状为 (frames, channels, height, width)
        """
        # 强制不保存文件，只返回tensor
        kwargs['output_path'] = None
        video_frames = self.generate_video(prompt, **kwargs)
        
        # 转换为torch tensor
        self.logger.info(f"generate_video_tensor 收到数据类型: {type(video_frames)}")
        
        if isinstance(video_frames, np.ndarray):
            # video_frames形状: (frames, height, width, channels)
            # 转换为: (frames, channels, height, width)
            video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2).float()
            # 归一化到[0, 1]
            video_tensor = video_tensor / 255.0
        elif isinstance(video_frames, list):
            # 如果是列表，尝试转换为numpy数组
            self.logger.info(f"列表长度: {len(video_frames)}, 第一个元素类型: {type(video_frames[0]) if video_frames else 'empty'}")
            if video_frames and isinstance(video_frames[0], np.ndarray):
                # 列表中包含numpy数组，合并它们
                video_array = np.stack(video_frames, axis=0)
                video_tensor = torch.from_numpy(video_array).float()
                # 检查维度并调整
                if video_tensor.dim() == 4:  # (frames, height, width, channels)
                    video_tensor = video_tensor.permute(0, 3, 1, 2)
                # 归一化到[0, 1]
                if video_tensor.max() > 1.0:
                    video_tensor = video_tensor / 255.0
            elif video_frames and hasattr(video_frames[0], 'convert'):
                # PIL.Image 对象列表
                from PIL import Image
                self.logger.info("检测到PIL图像列表，转换为tensor")
                # 转换PIL图像为numpy数组
                frames = []
                for img in video_frames:
                    if isinstance(img, Image.Image):
                        # 确保图像是RGB格式
                        img_rgb = img.convert('RGB')
                        # 转换为numpy数组 (H, W, C)
                        frame_array = np.array(img_rgb)
                        frames.append(frame_array)
                
                # 堆叠所有帧 (frames, height, width, channels)
                video_array = np.stack(frames, axis=0)
                # 转换为tensor并调整维度 (frames, channels, height, width)
                video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).float()
                # 归一化到[0, 1]
                video_tensor = video_tensor / 255.0
            else:
                raise ValueError(f"不支持的列表内容类型: {type(video_frames[0]) if video_frames else 'empty list'}")
        elif torch.is_tensor(video_frames):
            video_tensor = video_frames.float()
            # 检查是否需要归一化
            if video_tensor.max() > 1.0:
                video_tensor = video_tensor / 255.0
        else:
            raise ValueError(f"不支持的video_frames类型: {type(video_frames)}")
        
        self.logger.info(f"最终tensor形状: {video_tensor.shape}")
        return video_tensor
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """获取管道信息"""
        info = {
            "device": self.device,
            "pipeline_loaded": self.pipeline is not None,
            "diffusers_available": DIFFUSERS_AVAILABLE
        }
        
        if self.pipeline is not None:
            info.update({
                "dtype": str(self.pipeline.dtype) if hasattr(self.pipeline, 'dtype') else 'unknown',
                "components": list(self.pipeline.components.keys()) if hasattr(self.pipeline, 'components') else []
            })
        
        return info
    
    def clear_pipeline(self):
        """清理管道以释放内存"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("HunyuanVideo管道已清理")


# 方便的工具函数
def create_hunyuan_generator(
    cache_dir: Optional[str] = None,
    device: Optional[str] = None
) -> HunyuanVideoGenerator:
    """
    创建HunyuanVideo生成器的快捷函数
    
    Args:
        cache_dir: 模型缓存目录
        device: 计算设备
        
    Returns:
        HunyuanVideoGenerator: 生成器实例
    """
    model_manager = ModelManager(cache_dir) if cache_dir else ModelManager()
    return HunyuanVideoGenerator(model_manager, device)


if __name__ == "__main__":
    # 测试代码
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("测试HunyuanVideoGenerator...")
    
    try:
        generator = create_hunyuan_generator()
        
        # 显示生成器信息
        info = generator.get_pipeline_info()
        print("生成器信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 如果命令行参数包含test，进行实际生成测试
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            print("\n开始生成测试视频...")
            
            # 生成一个简短的测试视频
            test_prompt = "一只可爱的小猫在草地上玩耍"
            
            video_tensor = generator.generate_video_tensor(
                prompt=test_prompt,
                num_frames=25,  # 较短的视频用于测试
                height=480,     # 较低分辨率用于测试
                width=640,
                num_inference_steps=20,  # 较少步数用于快速测试
                seed=42
            )
            
            print(f"✅ 测试视频生成完成: {video_tensor.shape}")
            
            # 也可以保存为文件
            output_path = "test_hunyuan_output.mp4"
            generator.generate_video(
                prompt=test_prompt,
                num_frames=25,
                height=480,
                width=640,
                num_inference_steps=20,
                seed=42,
                output_path=output_path
            )
            
            print(f"✅ 测试视频已保存: {output_path}")
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()