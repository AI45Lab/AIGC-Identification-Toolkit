"""
统一视频水印接口
整合Wan2.1文生视频和VideoSeal水印技术
"""

import os
import logging
import torch
from typing import Optional, Dict, Any, Union
from pathlib import Path

from .model_manager import ModelManager
from .wan_video_generator import WanVideoGenerator
from .videoseal_wrapper import VideoSealWrapper
from .utils import VideoIOUtils, PerformanceTimer, FileUtils, MemoryMonitor, VideoTranscoder
from src.utils.path_manager import path_manager


class VideoWatermark:
    """统一视频水印接口类"""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        初始化视频水印工具

        Args:
            cache_dir: HuggingFace模型缓存目录（None则使用环境变量或默认路径）
            device: 计算设备 ('cuda', 'cpu', 或None自动选择)
            config: 配置字典，可包含VideoSeal等参数
        """
        # Resolve cache directory using path_manager
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = str(path_manager.get_hf_hub_dir())
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 保存配置
        self.config = config or {}

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 初始化组件（延迟加载）
        self.model_manager = None
        self.video_generator = None
        self.watermark_wrapper = None

        # 创建缓存目录（使用解析后的 self.cache_dir，而不是可能为 None 的参数）
        FileUtils.ensure_dir(self.cache_dir)
        
        self.logger.info(f"VideoWatermark初始化完成，设备: {self.device}")
    
    def _transcode_for_browser(self, video_path: str) -> str:
        """
        将视频转码为浏览器兼容格式
        
        Args:
            video_path: 输入视频路径
            
        Returns:
            str: 转码后的视频路径
        """
        self.logger.info(f"开始转码视频为浏览器兼容格式: {video_path}")
        
        try:
            # 检查是否已经兼容
            if VideoTranscoder.is_web_compatible(video_path):
                self.logger.info("视频已经是浏览器兼容格式，无需转码")
                return video_path
            
            # 生成转码后的文件路径
            path = Path(video_path)
            transcoded_path = path.parent / f"{path.stem}_web_compatible.mp4"
            transcoded_path = FileUtils.get_unique_filename(str(transcoded_path))
            
            # 执行转码
            with PerformanceTimer("视频转码", self.logger):
                result_path = VideoTranscoder.transcode_for_browser(
                    input_path=video_path,
                    output_path=transcoded_path,
                    target_fps=15,  # 匹配保存时的帧率
                    quality='medium'
                )
            
            # 获取转码后文件大小
            original_size = FileUtils.get_file_size_mb(video_path)
            transcoded_size = FileUtils.get_file_size_mb(result_path)
            
            self.logger.info(f"转码完成: {result_path}")
            self.logger.info(f"文件大小: {original_size:.1f} MB -> {transcoded_size:.1f} MB")
            
            # 可选: 删除原始文件以节省空间
            try:
                os.remove(video_path)
                self.logger.info(f"已删除原始文件: {video_path}")
            except Exception as e:
                self.logger.warning(f"删除原始文件失败: {e}")
            
            return result_path
            
        except Exception as e:
            self.logger.error(f"视频转码失败: {e}")
            # 转码失败时返回原始文件
            return video_path
    
    def _ensure_model_manager(self) -> ModelManager:
        """确保模型管理器已初始化"""
        if self.model_manager is None:
            self.model_manager = ModelManager(self.cache_dir)
        return self.model_manager
    
    def _ensure_video_generator(self) -> WanVideoGenerator:
        """确保视频生成器已初始化（使用Wan2.1模型）"""
        if self.video_generator is None:
            model_manager = self._ensure_model_manager()
            self.video_generator = WanVideoGenerator(model_manager, self.device)
        return self.video_generator
    
    def _ensure_watermark_wrapper(self) -> VideoSealWrapper:
        """确保水印包装器已初始化"""
        if self.watermark_wrapper is None:
            model_card = self.config.get('videoseal', {}).get('model_card', 'videoseal_1.0')
            self.watermark_wrapper = VideoSealWrapper(self.device, model_card=model_card)
        return self.watermark_wrapper
    
    def generate_video_with_watermark(
        self,
        prompt: str,
        message: str,
        output_path: Optional[str] = None,
        # Wan2.1视频生成参数
        negative_prompt: Optional[str] = None,
        num_frames: int = 81,  # Wan2.1推荐：81帧（5秒@15fps）
        # 使用16的倍数作为默认分辨率，避免后续VideoSeal对齐报错
        height: int = 480,  # Wan2.1推荐：480p
        width: int = 832,   # Wan2.1推荐：832（16:9比例）
        num_inference_steps: int = 50,  # Wan2.1推荐：50步
        guidance_scale: float = 5.0,    # Wan2.1推荐：5.0
        seed: Optional[int] = None,
        # VideoSeal参数
        lowres_attenuation: bool = True,
        # 🆕 原始视频保存选项
        return_original: bool = False
    ) -> Union[str, Dict[str, str]]:
        """
        文生视频+水印嵌入一体化功能
        
        Args:
            prompt: 文本提示词
            message: 要嵌入的水印消息
            output_path: 输出文件路径，如果None则自动生成
            negative_prompt: 负向提示词
            num_frames: 视频帧数
            height: 视频高度
            width: 视频宽度
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            seed: 随机种子
            lowres_attenuation: VideoSeal低分辨率衰减
            return_original: 是否同时返回原始视频路径
            
        Returns:
            str: 输出视频文件路径（当return_original=False时）
            Dict[str, str]: 包含'original'和'watermarked'键的字典（当return_original=True时）
        """
        self.logger.info("开始文生视频+水印嵌入流程")
        self.logger.info(f"提示词: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
        self.logger.info(f"水印消息: '{message}'")
        
        with PerformanceTimer("文生视频+水印嵌入", self.logger):
            # 1. 生成视频tensor（确保分辨率为16的倍数）
            self.logger.info("步骤1: 生成视频tensor")
            generator = self._ensure_video_generator()
            # 对齐到16的倍数
            def _align16(x: int) -> int:
                return max(16, (x // 16) * 16)
            height_aligned = _align16(height)
            width_aligned = _align16(width)
            if height_aligned != height or width_aligned != width:
                self.logger.info(f"分辨率自动对齐: {height}x{width} -> {height_aligned}x{width_aligned}")
            
            with PerformanceTimer("视频生成", self.logger):
                video_tensor = generator.generate_video_tensor(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    height=height_aligned,
                    width=width_aligned,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
            
            self.logger.info(f"视频生成完成: {video_tensor.shape}")
            
            # 2. 嵌入水印
            self.logger.info("步骤2: 嵌入水印")
            wrapper = self._ensure_watermark_wrapper()
            
            with PerformanceTimer("水印嵌入", self.logger):
                watermarked_tensor = wrapper.embed_watermark(
                    video_tensor=video_tensor,
                    message=message,
                    is_video=True,
                    lowres_attenuation=lowres_attenuation
                )
            
            # 3. 保存视频文件
            self.logger.info("步骤3: 保存视频文件")
            
            # 生成输出路径
            if output_path is None:
                # 创建安全的文件名
                safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_message = "".join(c for c in message[:20] if c.isalnum() or c in ('-', '_')).rstrip()
                filename = f"wan_{safe_prompt}_{safe_message}.mp4".replace(' ', '_')
                output_path = os.path.join("tests/test_results", filename)
            
            # 确保输出目录存在
            FileUtils.ensure_dir(os.path.dirname(output_path))
            
            # 避免文件名冲突
            watermarked_path = FileUtils.get_unique_filename(output_path)
            
            # 🆕 如果需要保存原始视频，先保存原始版本
            original_path = None
            if return_original:
                # 生成原始视频文件路径
                base_name = os.path.splitext(watermarked_path)[0]
                original_temp_path = f"{base_name}_original_temp.mp4"
                original_temp_path = FileUtils.get_unique_filename(original_temp_path)
                
                # 保存原始视频（临时）
                with PerformanceTimer("原始视频保存", self.logger):
                    VideoIOUtils.save_video_tensor(video_tensor, original_temp_path, fps=15)
                
                # 转码为浏览器兼容格式
                original_path = self._transcode_for_browser(original_temp_path)
                
                original_size = FileUtils.get_file_size_mb(original_path)
                self.logger.info(f"原始视频已保存: {original_path} ({original_size:.1f} MB)")
            
            # 保存水印视频（临时）
            watermarked_temp_path = f"{os.path.splitext(watermarked_path)[0]}_temp.mp4"
            watermarked_temp_path = FileUtils.get_unique_filename(watermarked_temp_path)
            
            with PerformanceTimer("水印视频保存", self.logger):
                VideoIOUtils.save_video_tensor(watermarked_tensor, watermarked_temp_path, fps=15)
            
            # 转码为浏览器兼容格式
            final_watermarked_path = self._transcode_for_browser(watermarked_temp_path)
            
            watermarked_size = FileUtils.get_file_size_mb(final_watermarked_path)
            self.logger.info(f"水印视频已保存: {final_watermarked_path} ({watermarked_size:.1f} MB)")
            
            # 🆕 根据return_original参数决定返回格式
            if return_original:
                return {
                    'original': original_path,
                    'watermarked': final_watermarked_path
                }
            else:
                return final_watermarked_path
    
    def embed_watermark(
        self,
        video_path: str,
        message: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        lowres_attenuation: bool = True
    ) -> str:
        """
        在现有视频文件中嵌入水印
        
        Args:
            video_path: 输入视频文件路径
            message: 要嵌入的水印消息
            output_path: 输出文件路径，如果None则自动生成
            max_frames: 最大处理帧数限制
            lowres_attenuation: VideoSeal低分辨率衰减
            
        Returns:
            str: 输出视频文件路径
        """
        self.logger.info(f"开始在现有视频中嵌入水印: {video_path}")
        self.logger.info(f"水印消息: '{message}'")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"输入视频文件不存在: {video_path}")
        
        with PerformanceTimer("视频水印嵌入", self.logger):
            # 1. 读取视频
            self.logger.info("步骤1: 读取视频文件")
            with PerformanceTimer("视频读取", self.logger):
                video_tensor = VideoIOUtils.read_video_frames(video_path, max_frames)
            
            self.logger.info(f"视频读取完成: {video_tensor.shape}")
            
            # 2. 嵌入水印
            self.logger.info("步骤2: 嵌入水印")
            wrapper = self._ensure_watermark_wrapper()
            
            with PerformanceTimer("水印嵌入", self.logger):
                watermarked_tensor = wrapper.embed_watermark(
                    video_tensor=video_tensor,
                    message=message,
                    is_video=True,
                    lowres_attenuation=lowres_attenuation
                )
            
            # 3. 保存视频
            self.logger.info("步骤3: 保存带水印视频")
            
            # 生成输出路径
            if output_path is None:
                input_path = Path(video_path)
                safe_message = "".join(c for c in message[:20] if c.isalnum() or c in ('-', '_')).rstrip()
                output_name = f"{input_path.stem}_watermarked_{safe_message}{input_path.suffix}"
                output_path = os.path.join("tests/test_results", output_name)
            
            # 确保输出目录存在
            FileUtils.ensure_dir(os.path.dirname(output_path))
            
            # 避免文件名冲突
            temp_output_path = FileUtils.get_unique_filename(f"{os.path.splitext(output_path)[0]}_temp.mp4")
            
            with PerformanceTimer("视频保存", self.logger):
                VideoIOUtils.save_video_tensor(watermarked_tensor, temp_output_path, fps=15)
            
            # 转码为浏览器兼容格式
            final_output_path = self._transcode_for_browser(temp_output_path)
            
            file_size = FileUtils.get_file_size_mb(final_output_path)
            self.logger.info(f"带水印视频已保存: {final_output_path} ({file_size:.1f} MB)")
            
            return final_output_path
    
    def extract_watermark(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        从视频中提取水印
        
        Args:
            video_path: 带水印的视频文件路径
            max_frames: 最大处理帧数限制
            chunk_size: 分块大小，如果None则从配置读取
            
        Returns:
            Dict[str, Any]: 提取结果，包含detected、message、confidence等字段
        """
        self.logger.info(f"开始从视频中提取水印: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        with PerformanceTimer("水印提取", self.logger):
            # 1. 读取视频
            self.logger.info("步骤1: 读取视频文件")
            with PerformanceTimer("视频读取", self.logger):
                video_tensor = VideoIOUtils.read_video_frames(video_path, max_frames)
            
            self.logger.info(f"视频读取完成: {video_tensor.shape}")
            
            # 2. 提取水印
            self.logger.info("步骤2: 提取水印")
            wrapper = self._ensure_watermark_wrapper()
            
            with PerformanceTimer("水印检测", self.logger):
                # 从配置获取chunk_size，如果没有则使用参数或默认值
                if chunk_size is None:
                    videoseal_config = self.config.get('videoseal', {})
                    watermark_params = videoseal_config.get('watermark_params', {})
                    chunk_size = watermark_params.get('chunk_size', 16)
                
                result = wrapper.extract_watermark(
                    watermarked_video=video_tensor,
                    is_video=True,
                    chunk_size=chunk_size
                )
            
            # 添加额外信息
            result.update({
                "video_path": video_path,
                "video_shape": video_tensor.shape,
                "processing_device": self.device
            })
            
            self.logger.info(
                f"水印提取完成 - 检测: {result['detected']}, "
                f"置信度: {result['confidence']:.3f}, "
                f"消息: '{result['message']}'"
            )
            
            return result
    
    def batch_process_videos(
        self,
        video_paths: list,
        messages: list,
        operation: str = "embed",
        output_dir: str = "tests/test_results",
        **kwargs
    ) -> list:
        """
        批量处理视频
        
        Args:
            video_paths: 视频文件路径列表
            messages: 消息列表（embed操作时使用）
            operation: 操作类型 ('embed' 或 'extract')
            output_dir: 输出目录
            **kwargs: 其他参数
            
        Returns:
            list: 处理结果列表
        """
        self.logger.info(f"开始批量{operation}操作，处理{len(video_paths)}个视频")
        
        FileUtils.ensure_dir(output_dir)
        results = []
        
        for i, video_path in enumerate(video_paths):
            try:
                self.logger.info(f"处理 {i+1}/{len(video_paths)}: {os.path.basename(video_path)}")
                
                if operation == "embed":
                    message = messages[i] if i < len(messages) else f"batch_message_{i+1}"
                    output_path = os.path.join(output_dir, f"batch_watermarked_{i+1}.mp4")
                    
                    result_path = self.embed_watermark(
                        video_path=video_path,
                        message=message,
                        output_path=output_path,
                        **kwargs
                    )
                    
                    results.append({
                        "index": i,
                        "input_path": video_path,
                        "output_path": result_path,
                        "message": message,
                        "success": True
                    })
                
                elif operation == "extract":
                    extract_result = self.extract_watermark(video_path, **kwargs)
                    
                    results.append({
                        "index": i,
                        "input_path": video_path,
                        "extract_result": extract_result,
                        "success": True
                    })
                
                else:
                    raise ValueError(f"不支持的操作类型: {operation}")
            
            except Exception as e:
                self.logger.error(f"处理视频{i+1}失败: {e}")
                results.append({
                    "index": i,
                    "input_path": video_path,
                    "success": False,
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r["success"])
        self.logger.info(f"批量处理完成: {success_count}/{len(video_paths)} 成功")
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            "device": self.device,
            "cache_dir": self.cache_dir,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        # GPU内存信息
        if torch.cuda.is_available():
            info["gpu_memory"] = MemoryMonitor.get_gpu_memory_info()
        
        # 模型信息
        if self.model_manager:
            info["wan_model"] = self.model_manager.get_wan_model_info()
        
        if self.video_generator:
            info["video_generator"] = self.video_generator.get_pipeline_info()
        
        if self.watermark_wrapper:
            info["videoseal"] = self.watermark_wrapper.get_model_info()
        
        return info
    
    def clear_cache(self):
        """清理所有缓存以释放内存"""
        self.logger.info("清理所有缓存...")
        
        if self.video_generator:
            self.video_generator.clear_pipeline()
        
        if self.watermark_wrapper:
            self.watermark_wrapper.clear_model()
        
        # 清理GPU缓存
        MemoryMonitor.clear_gpu_cache()
        
        self.logger.info("缓存清理完成")


# 方便的工厂函数
def create_video_watermark(
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    config_path: Optional[str] = None
) -> VideoWatermark:
    """
    创建视频水印工具的快捷函数

    Args:
        cache_dir: 模型缓存目录（None则使用环境变量或默认路径）
        device: 计算设备
        config_path: 配置文件路径，如果为None则使用默认配置

    Returns:
        VideoWatermark: 视频水印工具实例
    """
    config = {}
    if config_path:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
        config = full_config.get('video_watermark', {})
        if not cache_dir:
            cache_dir = config.get('cache_dir')
        if not device:
            device = config.get('device')
    return VideoWatermark(cache_dir=cache_dir, device=device, config=config)


if __name__ == "__main__":
    # 测试代码
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("测试VideoWatermark统一接口...")
    
    try:
        # 创建视频水印工具
        watermark_tool = create_video_watermark()
        
        # 显示系统信息
        system_info = watermark_tool.get_system_info()
        print("系统信息:")
        for key, value in system_info.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # 如果命令行参数包含test，进行简化测试
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            print("\n开始简化功能测试...")
            
            # 测试1: 文生视频+水印（使用较小参数）
            print("测试1: 文生视频+水印")
            try:
                output_path = watermark_tool.generate_video_with_watermark(
                    prompt="一朵红色的花",
                    message="test_2025",
                    num_frames=16,      # 较少帧数
                    height=320,         # 较小分辨率
                    width=320,
                    num_inference_steps=10,  # 较少步数
                    seed=42
                )
                print(f"✅ 文生视频+水印完成: {output_path}")
                
                # 测试2: 水印提取
                print("测试2: 水印提取")
                extract_result = watermark_tool.extract_watermark(output_path)
                print(f"提取结果: {extract_result}")
                
                # 验证
                success = (extract_result["detected"] and 
                          extract_result["message"] == "test_2025")
                print(f"验证结果: {'✅ 成功' if success else '❌ 失败'}")
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()