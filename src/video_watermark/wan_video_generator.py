"""
Wan2.1文生视频生成器
基于Wan-AI/Wan2.1-T2V-1.3B-Diffusers模型的文本到视频生成功能
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
    from diffusers import WanPipeline, AutoencoderKLWan
    from diffusers.utils import export_to_video
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("diffusers not available. Please install with: pip install diffusers")


class WanVideoGenerator:
    """Wan2.1文生视频生成器"""

    def __init__(self, model_manager: ModelManager, device: Optional[str] = None):
        """
        初始化Wan2.1生成器

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
                "diffusers is required for Wan2.1 generation. "
                "Install with: pip install diffusers torch torchvision"
            )

    def _load_pipeline(self, allow_download: bool = False):
        """
        延迟加载Wan2.1管道

        关键特性：
        - 使用trust_remote_code=True（必需）
        - VAE使用float32数据类型
        - Pipeline使用bfloat16（CUDA）或float32（CPU）
        - 仅加载本地模型（local_files_only=True）
        """
        if self.pipeline is not None:
            return

        self.logger.info("正在加载Wan2.1管道...")

        try:
            # 仅使用本地快照路径
            try:
                local_model_path = self.model_manager.ensure_wan_model(allow_download=allow_download)
            except Exception as e:
                raise RuntimeError(
                    f"未找到本地Wan2.1模型，请确保模型已下载到: "
                    f"{self.model_manager.cache_dir}/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers\n"
                    f"错误详情: {e}"
                )

            self.logger.info(f"从本地快照加载Wan2.1: {local_model_path}")

            # 步骤1: 加载VAE（必须使用float32，这是Wan2.1的要求）
            self.logger.info("加载AutoencoderKLWan (VAE)...")
            vae = AutoencoderKLWan.from_pretrained(
                local_model_path,
                subfolder="vae",
                torch_dtype=torch.float32,  # VAE必须使用float32
                local_files_only=True,
                trust_remote_code=True  # Wan2.1必需参数
            )
            self.logger.info("✓ VAE加载完成 (torch.float32)")

            # 步骤2: 加载Pipeline
            # CUDA使用bfloat16以节省显存，CPU使用float32
            torch_dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
            self.logger.info(f"加载WanPipeline (torch_dtype={torch_dtype})...")

            self.pipeline = WanPipeline.from_pretrained(
                local_model_path,
                vae=vae,  # 传入已加载的VAE
                torch_dtype=torch_dtype,
                local_files_only=True,
                trust_remote_code=True  # Wan2.1必需参数
            )
            self.logger.info("✓ Pipeline加载完成")

            # 步骤3: 移动到指定设备
            self.pipeline.to(self.device)
            self.logger.info(f"✓ Pipeline已移动到设备: {self.device}")

            # 步骤4: 内存优化（复用HunyuanVideo的优化策略）
            if hasattr(self.pipeline, 'vae') and hasattr(self.pipeline.vae, 'enable_tiling'):
                self.pipeline.vae.enable_tiling()
                self.logger.info("✓ 启用VAE tiling（降低显存占用）")

            # CPU offload优化（仅CUDA）
            if self.device == 'cuda' and hasattr(self.pipeline, 'enable_model_cpu_offload'):
                # 检查是否已有device_map，避免冲突
                using_device_map = hasattr(self.pipeline, 'hf_device_map') and self.pipeline.hf_device_map is not None
                if not using_device_map:
                    self.pipeline.enable_model_cpu_offload()
                    self.logger.info("✓ 启用模型CPU offload")
                else:
                    self.logger.info("⚠ 检测到device_map，跳过CPU offload以避免冲突")

            self.logger.info(f"🎉 Wan2.1管道加载完成！设备: {self.device}, 数据类型: {torch_dtype}")
            self.logger.info("⚠ 安全提示: trust_remote_code=True 仅在本地模式下使用")

        except Exception as e:
            self.logger.error(f"❌ 加载Wan2.1管道失败: {e}")
            import traceback
            self.logger.error(f"详细错误信息:\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to load Wan2.1 pipeline: {e}")

    def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: int = 81,  # Wan2.1推荐：81帧（5秒@15fps）
        height: int = 480,      # Wan2.1推荐：480p
        width: int = 832,       # Wan2.1推荐：832（16:9比例）
        num_inference_steps: int = 50,  # Wan2.1推荐：50步
        guidance_scale: float = 5.0,    # Wan2.1推荐：5.0
        seed: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Union[torch.Tensor, str]:
        """
        生成视频

        Args:
            prompt: 文本提示词
            negative_prompt: 负向提示词（推荐使用以提升质量）
            num_frames: 视频帧数 (推荐81帧=5秒@15fps，Wan2.1无严格格式限制)
            height: 视频高度 (推荐480p，支持720p但不稳定)
            width: 视频宽度 (推荐832，16:9比例)
            num_inference_steps: 推理步数 (推荐50，越高质量越好但越慢)
            guidance_scale: 引导强度 (推荐5.0)
            seed: 随机种子
            output_path: 输出视频路径，如果None则返回tensor

        Returns:
            torch.Tensor or str: 视频tensor或输出文件路径
        """
        self._load_pipeline(allow_download=False)  # 仅使用本地模型

        self.logger.info(f"开始生成视频: '{prompt[:50]}...'")
        self.logger.info(f"参数: {num_frames}帧, {height}x{width}, {num_inference_steps}步, CFG={guidance_scale}")

        # 设置推荐的负向提示词（如果用户未提供）
        if negative_prompt is None:
            negative_prompt = (
                "Bright tones, overexposed, static, blurred details, "
                "unclear limbs, worst quality, low quality"
            )
            self.logger.info(f"使用默认负向提示词: {negative_prompt[:50]}...")

        # 设置随机种子
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            self.logger.info(f"设置随机种子: {seed}")

        try:
            # 生成视频（带OOM自适应重试，复用HunyuanVideo的逻辑）
            attempt = 0
            max_attempts = 3
            current_params = {
                'num_frames': num_frames,
                'height': height,
                'width': width,
                'num_inference_steps': num_inference_steps
            }

            while attempt < max_attempts:
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
                            generator=generator
                        )
                    break  # 生成成功，跳出重试循环

                except RuntimeError as re:
                    # 捕获CUDA OOM并自适应降低参数重试
                    message = str(re)
                    if ('CUDA out of memory' in message or 'out of memory' in message) and attempt < max_attempts - 1:
                        self.logger.warning(f"⚠ 检测到CUDA OOM，进行自适应重试 (attempt {attempt + 1}/{max_attempts})")
                        self.logger.warning(f"OOM详情: {message[:200]}...")

                        attempt += 1

                        # 清理GPU缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # 降低参数策略
                        current_params['height'] = max(256, current_params['height'] // 2)
                        current_params['width'] = max(256, current_params['width'] // 2)
                        current_params['num_frames'] = max(25, current_params['num_frames'] // 2)
                        current_params['num_inference_steps'] = max(20, current_params['num_inference_steps'] - 10)

                        self.logger.info(
                            f"重试参数 -> frames: {current_params['num_frames']}, "
                            f"size: {current_params['height']}x{current_params['width']}, "
                            f"steps: {current_params['num_inference_steps']}"
                        )
                        continue
                    raise  # 其他错误或达到最大重试次数，直接抛出

            # 提取视频帧（Wan2.1标准输出格式: result.frames[0]）
            self.logger.info(f"管道输出类型: {type(result)}")

            if hasattr(result, 'frames') and result.frames is not None:
                video_frames = result.frames[0]
                self.logger.info(f"✓ 从result.frames[0]提取视频帧")
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                video_frames = result[0]
                self.logger.info(f"✓ 从result[0]提取视频帧")
            else:
                raise ValueError(f"不支持的输出格式: {type(result)}")

            # 验证输出类型
            if isinstance(video_frames, list) and len(video_frames) > 0:
                self.logger.info(f"✓ 视频帧类型: list, 长度: {len(video_frames)}")
                if hasattr(video_frames[0], 'size'):
                    self.logger.info(f"✓ 第一帧尺寸: {video_frames[0].size}")
            else:
                self.logger.warning(f"⚠ 意外的视频帧类型: {type(video_frames)}")

            self.logger.info(f"🎬 视频生成完成！")

            # 如果指定输出路径，保存视频文件
            if output_path:
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

                # 使用diffusers的export_to_video保存 (官方推荐fps=15)
                export_to_video(video_frames, output_path, fps=15)

                self.logger.info(f"✓ 视频已保存到: {output_path}")
                return output_path
            else:
                # 返回video_frames（PIL.Image列表）
                return video_frames

        except Exception as e:
            # CPU回退策略（复用HunyuanVideo的逻辑）
            message = str(e)
            if self.device != 'cpu' and ('CUDA out of memory' in message or 'out of memory' in message):
                try:
                    self.logger.warning("⚠ 持续OOM，尝试切换到CPU并以更小参数重试")

                    # 切换到CPU
                    self.pipeline = self.pipeline.to('cpu')
                    self.device = 'cpu'

                    # 进一步降低参数
                    retry_frames = max(25, num_frames // 2)
                    retry_height = max(256, height // 2)
                    retry_width = max(256, width // 2)
                    retry_steps = max(20, num_inference_steps - 15)

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

                    # 提取输出
                    if hasattr(result, 'frames') and result.frames is not None:
                        video_frames = result.frames[0]
                    elif isinstance(result, (list, tuple)) and len(result) > 0:
                        video_frames = result[0]
                    else:
                        video_frames = result

                    self.logger.info(f"✓ CPU回退成功: frames={retry_frames}, size={retry_height}x{retry_width}, steps={retry_steps}")

                    if output_path:
                        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
                        export_to_video(video_frames, output_path, fps=15)
                        self.logger.info(f"✓ 视频已保存到: {output_path}")
                        return output_path
                    return video_frames

                except Exception as e_cpu:
                    self.logger.error(f"❌ CPU回退仍失败: {e_cpu}")
                    # 继续抛出原始错误

            self.logger.error(f"❌ 视频生成失败: {e}")
            import traceback
            self.logger.error(f"详细错误信息:\n{traceback.format_exc()}")
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
            torch.Tensor: 视频tensor，形状为 (frames, channels, height, width)，值域[0,1]
        """
        # 强制不保存文件，只返回tensor
        kwargs['output_path'] = None
        video_frames = self.generate_video(prompt, **kwargs)

        # 转换为torch tensor
        self.logger.info(f"generate_video_tensor 收到数据类型: {type(video_frames)}")

        # Wan2.1输出格式：PIL.Image列表
        if isinstance(video_frames, list) and video_frames and hasattr(video_frames[0], 'convert'):
            from PIL import Image
            self.logger.info(f"✓ 检测到PIL图像列表，长度: {len(video_frames)}")

            # 转换PIL图像为numpy数组
            frames = []
            for i, img in enumerate(video_frames):
                if isinstance(img, Image.Image):
                    # 确保图像是RGB格式
                    img_rgb = img.convert('RGB')
                    # 转换为numpy数组 (H, W, C)
                    frame_array = np.array(img_rgb)
                    frames.append(frame_array)

                    # 记录第一帧信息
                    if i == 0:
                        self.logger.info(f"  第一帧尺寸: {img.size}, 数组shape: {frame_array.shape}, 值域: [{frame_array.min()}, {frame_array.max()}]")

            # 堆叠所有帧 (frames, height, width, channels)
            video_array = np.stack(frames, axis=0)
            self.logger.info(f"  堆叠后数组shape: {video_array.shape}")

            # 转换为tensor并调整维度 (frames, channels, height, width)
            video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).float()

            # 归一化到[0, 1]
            video_tensor = video_tensor / 255.0
            self.logger.info(f"✓ 转换完成: tensor shape={video_tensor.shape}, 值域=[{video_tensor.min():.3f}, {video_tensor.max():.3f}]")

        elif isinstance(video_frames, np.ndarray):
            # 直接是numpy数组（不太可能，但保留兼容性）
            self.logger.info(f"收到numpy数组: shape={video_frames.shape}")
            video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2).float()
            if video_tensor.max() > 1.0:
                video_tensor = video_tensor / 255.0

        elif torch.is_tensor(video_frames):
            # 直接是tensor（不太可能，但保留兼容性）
            self.logger.info(f"收到torch tensor: shape={video_frames.shape}")
            video_tensor = video_frames.float()
            if video_tensor.max() > 1.0:
                video_tensor = video_tensor / 255.0
        else:
            raise ValueError(f"不支持的video_frames类型: {type(video_frames)}")

        self.logger.info(f"🎬 最终tensor形状: {video_tensor.shape}, 值域: [{video_tensor.min():.3f}, {video_tensor.max():.3f}]")
        return video_tensor

    def get_pipeline_info(self) -> Dict[str, Any]:
        """获取管道信息"""
        info = {
            "model": "Wan2.1-T2V-1.3B-Diffusers",
            "device": self.device,
            "pipeline_loaded": self.pipeline is not None,
            "diffusers_available": DIFFUSERS_AVAILABLE,
            "trust_remote_code": True,  # 标记使用了trust_remote_code
            "recommended_params": {
                "num_frames": 81,
                "height": 480,
                "width": 832,
                "num_inference_steps": 50,
                "guidance_scale": 5.0
            }
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

            self.logger.info("Wan2.1管道已清理")


# 方便的工具函数
def create_wan_generator(
    cache_dir: Optional[str] = None,
    device: Optional[str] = None
) -> WanVideoGenerator:
    """
    创建Wan2.1生成器的快捷函数

    Args:
        cache_dir: 模型缓存目录
        device: 计算设备

    Returns:
        WanVideoGenerator: 生成器实例
    """
    model_manager = ModelManager(cache_dir) if cache_dir else ModelManager()
    return WanVideoGenerator(model_manager, device)


if __name__ == "__main__":
    # 测试代码
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("Wan2.1 Video Generator 测试")
    print("=" * 60)

    try:
        generator = create_wan_generator()

        # 显示生成器信息
        info = generator.get_pipeline_info()
        print("\n生成器信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # 如果命令行参数包含test，进行实际生成测试
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            print("\n" + "=" * 60)
            print("开始生成测试视频...")
            print("=" * 60)

            # 生成一个短视频用于测试
            test_prompt = "A cat walks on the grass, realistic"

            video_tensor = generator.generate_video_tensor(
                prompt=test_prompt,
                num_frames=81,   # Wan2.1推荐
                height=480,      # Wan2.1推荐
                width=832,       # Wan2.1推荐
                num_inference_steps=50,
                guidance_scale=5.0,
                seed=42
            )

            print(f"\n✅ 测试视频生成完成!")
            print(f"   Tensor shape: {video_tensor.shape}")
            print(f"   值域: [{video_tensor.min():.3f}, {video_tensor.max():.3f}]")

            # 也可以保存为文件
            output_path = "test_wan_output.mp4"
            generator.generate_video(
                prompt=test_prompt,
                num_frames=81,
                height=480,
                width=832,
                num_inference_steps=50,
                guidance_scale=5.0,
                seed=42,
                output_path=output_path
            )

            print(f"\n✅ 测试视频已保存: {output_path}")
            print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
