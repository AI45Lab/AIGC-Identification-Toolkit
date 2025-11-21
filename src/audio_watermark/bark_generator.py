"""
Bark音频生成器
集成Bark模型进行文本转音频生成
"""

import torch
import numpy as np
import logging
import os
import warnings
from typing import Optional, Dict, Any, Union
from pathlib import Path
from src.utils.path_manager import path_manager

# ===== PyTorch 2.6兼容性补丁 =====
# 修复Bark模型加载的torch.load兼容性问题
# PyTorch 2.6默认weights_only=True，但Bark旧模型需要weights_only=False
_original_torch_load = torch.load

def _patched_torch_load(f, *args, **kwargs):
    """torch.load兼容性补丁，用于加载Bark旧模型"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_torch_load
# ===== 补丁结束 =====

# 尝试导入Bark相关依赖
HAS_BARK = False
set_seed = None  # 初始化为None

try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    HAS_BARK = True
    
    # 尝试导入set_seed函数，如果不存在也没关系
    try:
        from bark.generation import set_seed
    except ImportError:
        # 某些版本的Bark可能没有set_seed函数
        # 我们可以使用torch.manual_seed作为替代
        import torch
        def set_seed(seed):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
except ImportError:
    HAS_BARK = False
    warnings.warn(
        "Bark模块未安装。请手动安装:\n"
        "pip install git+https://github.com/suno-ai/bark.git"
    )

try:
    import scipy.io.wavfile as wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# 导入本地工具
from .utils import AudioProcessingUtils, FileUtils


class BarkGenerator:
    """Bark文本转音频生成器"""
    
    # Bark支持的语音预设
    VOICE_PRESETS = [
        # 英文预设
        "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", 
        "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
        "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9",
        # 中文预设
        "v2/zh_speaker_0", "v2/zh_speaker_1", "v2/zh_speaker_2", 
        "v2/zh_speaker_3", "v2/zh_speaker_4", "v2/zh_speaker_5",
        "v2/zh_speaker_6", "v2/zh_speaker_7", "v2/zh_speaker_8", "v2/zh_speaker_9",
        # 其他语言
        "v2/fr_speaker_0", "v2/fr_speaker_1", "v2/de_speaker_0", 
        "v2/hi_speaker_0", "v2/it_speaker_0", "v2/ja_speaker_0",
        "v2/ko_speaker_0", "v2/pl_speaker_0", "v2/pt_speaker_0",
        "v2/ru_speaker_0", "v2/es_speaker_0", "v2/tr_speaker_0"
    ]
    
    def __init__(self, 
                 device: Optional[str] = None,
                 model_size: str = "large",
                 use_gpu: bool = True,
                 target_sample_rate: int = 16000):
        """
        初始化Bark生成器
        
        Args:
            device: 计算设备 ('cuda', 'cpu' 或 None 自动选择)
            model_size: 模型大小 ('small', 'large')
            use_gpu: 是否使用GPU
            target_sample_rate: 目标采样率 (AudioSeal需要16kHz)
        """
        if not HAS_BARK:
            raise ImportError(
                "Bark模块未安装。请安装: pip install git+https://github.com/suno-ai/bark.git"
            )
        
        self.device = device or ('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.model_size = model_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.target_sample_rate = target_sample_rate
        self.bark_sample_rate = SAMPLE_RATE  # Bark原生采样率 (通常是24kHz)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 模型加载状态
        self._models_loaded = False
        self._default_voice = "v2/en_speaker_6"  # 默认语音
        
        self.logger.info(
            f"BarkGenerator初始化完成 - 设备: {self.device}, "
            f"模型: {model_size}, 目标采样率: {target_sample_rate}Hz"
        )
    
    def _setup_bark_cache_dir(self):
        """设置Bark模型缓存目录（离线优先，支持环境变量覆盖）"""
        # 保存原始环境变量，便于恢复
        original_cache_home = os.environ.get('XDG_CACHE_HOME')
        original_home = os.environ.get('HOME')

        # 项目缓存根目录（使用跨平台路径管理）
        cache_base = os.environ.get('XDG_CACHE_HOME') or str(path_manager.get_cache_root())

        # 允许通过环境变量直接指定 Bark 模型目录（可指向 bark_v0 或 HF hub 的 models--suno--bark）
        env_bark_dir = os.environ.get('BARK_CACHE_DIR')

        # HF 缓存根目录（使用跨平台路径管理）
        hf_home = os.environ.get('HF_HOME') or str(path_manager.get_hf_home())

        # 备选的本地 Bark 模型目录（按优先级）
        candidate_targets = []
        if env_bark_dir:
            candidate_targets.append(env_bark_dir)
        # 兼容新旧 HF Hub 目录结构
        candidate_targets.append(os.path.join(hf_home, 'hub', 'models--suno--bark'))
        candidate_targets.append(os.path.join(hf_home, 'models--suno--bark'))
        # 直接使用项目缓存下的 bark_v0（若已被手动填充）
        candidate_targets.append(os.path.join(cache_base, 'suno', 'bark_v0'))

        # 选择第一个存在且非空的目录作为目标
        target_cache_dir = None
        for path in candidate_targets:
            if os.path.exists(path) and os.listdir(path):
                target_cache_dir = path
                break

        # 确保 bark_v0 最终位置存在于项目缓存，并指向目标（若目标为空则创建占位目录）
        bark_cache_parent = os.path.join(cache_base, 'suno')
        bark_cache_dir = os.path.join(bark_cache_parent, 'bark_v0')
        os.makedirs(bark_cache_parent, exist_ok=True)

        if target_cache_dir and os.path.exists(target_cache_dir):
            if os.path.islink(bark_cache_dir) or os.path.exists(bark_cache_dir):
                try:
                    if os.path.islink(bark_cache_dir):
                        link_target = os.readlink(bark_cache_dir)
                        if link_target != target_cache_dir:
                            os.unlink(bark_cache_dir)
                            os.symlink(target_cache_dir, bark_cache_dir)
                            self.logger.info(f"更新符号链接: {bark_cache_dir} -> {target_cache_dir}")
                    else:
                        # 目录已存在但非链接，若为空可替换为链接
                        if not os.listdir(bark_cache_dir):
                            os.rmdir(bark_cache_dir)
                            os.symlink(target_cache_dir, bark_cache_dir)
                            self.logger.info(f"创建符号链接: {bark_cache_dir} -> {target_cache_dir}")
                except Exception:
                    # 如有权限或占用问题，忽略并继续使用现有目录
                    pass
            else:
                os.symlink(target_cache_dir, bark_cache_dir)
                self.logger.info(f"创建符号链接: {bark_cache_dir} -> {target_cache_dir}")
        else:
            # 未找到本地 Bark 模型，则仅确保目录存在（若开启离线模式将报错，避免联网）
            os.makedirs(bark_cache_dir, exist_ok=True)

        # 设置缓存目录环境变量，让 Bark/torch 遵循项目缓存
        os.environ['XDG_CACHE_HOME'] = cache_base

        # 根据配置选择小/大模型（若存在）
        os.environ['SUNO_USE_SMALL_MODELS'] = 'False' if self.model_size == 'large' else 'True'

        # 日志提示
        self.logger.info(f"Bark缓存目录设置为: {bark_cache_dir}")
        if target_cache_dir:
            self.logger.info(f"目标存储目录: {target_cache_dir}")
        else:
            self.logger.warning("未发现本地 Bark 模型权重，若开启离线模式将无法自动下载")

        # 返回修改后的环境变量和实际缓存目录
        return bark_cache_dir, original_cache_home, original_home
    
    def _check_local_models_exist(self, cache_dir: str) -> bool:
        """检查本地模型是否存在"""
        # Bark模型文件的基本检查
        # Bark模型通常存储在.cache/huggingface/download目录中
        # 检查是否有任何模型相关文件
        
        if not os.path.exists(cache_dir):
            self.logger.info(f"缓存目录不存在: {cache_dir}")
            return False
        
        # 检查是否有任何.cache文件或模型文件
        cache_files = []
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                if file.endswith(('.pt', '.pth', '.bin', '.safetensors')) or 'model' in file.lower():
                    cache_files.append(os.path.join(root, file))
        
        if cache_files:
            self.logger.info(f"检测到 {len(cache_files)} 个本地模型相关文件")
            for file in cache_files[:3]:  # 只显示前3个文件
                self.logger.debug(f"  模型文件: {file}")
            if len(cache_files) > 3:
                self.logger.debug(f"  ... 还有 {len(cache_files) - 3} 个文件")
            return True
        else:
            self.logger.info("未检测到本地Bark模型文件")
            return False
    
    def _ensure_models_loaded(self):
        """确保Bark模型已加载"""
        if not self._models_loaded:
            self.logger.info("预加载Bark模型...")

            # 设置缓存目录
            cache_dir, original_cache_home, original_home = self._setup_bark_cache_dir()

            try:
                # 设置设备
                if self.use_gpu:
                    os.environ['BARK_FORCE_CPU'] = 'False'
                else:
                    os.environ['BARK_FORCE_CPU'] = 'True'

                # 检查本地模型是否存在
                local_models_exist = self._check_local_models_exist(cache_dir)

                if local_models_exist:
                    self.logger.info("使用本地缓存的Bark模型")
                else:
                    self.logger.info("本地模型不完整，将下载到指定目录")
                    self.logger.warning(f"模型将下载到: {cache_dir}")
                    self.logger.warning("首次下载可能需要较长时间和大量磁盘空间(~5GB)")

                # 预加载模型（会自动从缓存加载或下载）
                preload_models()
                self._models_loaded = True
                self.logger.info("Bark模型预加载完成")

            except Exception as e:
                self.logger.error(f"Bark模型加载失败: {e}")
                # 检查是否是磁盘空间问题
                if "No space left on device" in str(e):
                    raise RuntimeError(
                        f"磁盘空间不足，无法下载Bark模型。需要约5GB空间。\n"
                        f"目标目录: {cache_dir}\n"
                        f"原始错误: {e}"
                    )
                else:
                    raise RuntimeError(f"无法加载Bark模型: {e}")
            finally:
                # 保持XDG_CACHE_HOME以确保后续路径解析一致（离线模式更稳健）
                # 如果需要恢复，可根据需要在外部会话恢复
                pass
    
    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            str: 预处理后的文本
        """
        # 移除过长的文本，Bark有长度限制
        if len(text) > 500:
            self.logger.warning(f"文本过长 ({len(text)} 字符)，将截断到500字符")
            text = text[:500]
        
        # 基本清理
        text = text.strip()
        
        # 检测语言并添加适当的标记
        # 这是一个简化的语言检测，实际可以使用更复杂的方法
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            # 包含中文字符
            if not text.startswith('[zh]'):
                text = f"[zh] {text}"
        elif any(ord(char) > 127 for char in text):
            # 包含非ASCII字符，可能是其他语言
            pass  # 保持原样
        else:
            # 英文文本
            if not text.startswith('[en]'):
                text = f"[en] {text}"
        
        return text
    
    def _postprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        """
        后处理音频
        
        Args:
            audio: Bark生成的音频numpy数组
            
        Returns:
            torch.Tensor: 处理后的音频张量
        """
        # 转换为torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # 确保是2D张量 (1, samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # 重采样到目标采样率
        if self.bark_sample_rate != self.target_sample_rate:
            audio_tensor = AudioProcessingUtils.resample(
                audio_tensor, 
                self.bark_sample_rate, 
                self.target_sample_rate
            )
        
        # 归一化
        audio_tensor = AudioProcessingUtils.normalize(audio_tensor, method='peak')
        
        return audio_tensor
    
    def generate_audio(self,
                      text: str,
                      voice_preset: Optional[str] = None,
                      temperature: float = 0.8,
                      silence_duration: float = 0.25,
                      seed: Optional[int] = None) -> torch.Tensor:
        """
        生成音频
        
        Args:
            text: 要转换的文本
            voice_preset: 语音预设，None使用默认
            temperature: 生成温度 (0.0-1.0, 越高越随机)
            silence_duration: 静音时长(秒)
            seed: 随机种子，None则随机
            
        Returns:
            torch.Tensor: 生成的音频张量 (1, samples)
        """
        self._ensure_models_loaded()
        
        # 预处理文本
        processed_text = self._preprocess_text(text)
        self.logger.info(f"生成音频，文本: '{processed_text[:50]}...'")
        
        # 设置随机种子
        if seed is not None:
            set_seed(seed)
        
        # 选择语音预设
        if voice_preset is None:
            voice_preset = self._default_voice
        elif voice_preset not in self.VOICE_PRESETS:
            self.logger.warning(f"未知语音预设: {voice_preset}，使用默认预设")
            voice_preset = self._default_voice
        
        try:
            # 生成音频
            audio_array = generate_audio(
                processed_text,
                history_prompt=voice_preset,
                text_temp=temperature,
                waveform_temp=temperature,
                silent=True  # 禁用进度条输出
            )
            
            # 后处理
            audio_tensor = self._postprocess_audio(audio_array)
            
            self.logger.info(
                f"音频生成成功 - 形状: {audio_tensor.shape}, "
                f"时长: {audio_tensor.size(-1) / self.target_sample_rate:.2f}秒"
            )
            
            return audio_tensor
            
        except Exception as e:
            self.logger.error(f"音频生成失败: {e}")
            raise RuntimeError(f"Bark音频生成失败: {e}")
    
    def text_to_speech(self,
                      text: str,
                      output_path: Optional[str] = None,
                      voice_preset: Optional[str] = None,
                      **kwargs) -> Union[torch.Tensor, str]:
        """
        文本转语音 (简化接口)
        
        Args:
            text: 输入文本
            output_path: 输出文件路径，None则返回tensor
            voice_preset: 语音预设
            **kwargs: 其他生成参数
            
        Returns:
            Union[torch.Tensor, str]: 音频tensor或输出文件路径
        """
        # 生成音频
        audio_tensor = self.generate_audio(text, voice_preset, **kwargs)
        
        # 保存或返回
        if output_path:
            from .utils import AudioIOUtils
            
            # 确保输出目录存在
            output_path = Path(output_path)
            FileUtils.ensure_dir(output_path.parent)
            
            # 避免文件名冲突
            output_path = FileUtils.get_unique_filename(output_path)
            
            # 保存音频
            AudioIOUtils.save_audio(
                audio_tensor, 
                output_path, 
                self.target_sample_rate
            )
            
            self.logger.info(f"音频已保存: {output_path}")
            return str(output_path)
        else:
            return audio_tensor
    
    def batch_generate(self,
                      texts: list,
                      output_dir: Optional[str] = None,
                      voice_presets: Optional[list] = None,
                      **kwargs) -> list:
        """
        批量生成音频
        
        Args:
            texts: 文本列表
            output_dir: 输出目录，None则返回tensor列表
            voice_presets: 语音预设列表，None则使用默认
            **kwargs: 生成参数
            
        Returns:
            list: 生成结果列表
        """
        self.logger.info(f"开始批量生成，共 {len(texts)} 个文本")
        
        if voice_presets is None:
            voice_presets = [None] * len(texts)
        elif len(voice_presets) != len(texts):
            raise ValueError("语音预设列表长度必须与文本列表长度相同")
        
        results = []
        
        for i, (text, voice_preset) in enumerate(zip(texts, voice_presets)):
            try:
                self.logger.info(f"生成第 {i+1}/{len(texts)} 个音频")
                
                if output_dir:
                    output_path = os.path.join(output_dir, f"generated_audio_{i+1}.wav")
                else:
                    output_path = None
                
                result = self.text_to_speech(
                    text, output_path, voice_preset, **kwargs
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"生成第 {i+1} 个音频失败: {e}")
                results.append(None)
        
        success_count = sum(1 for r in results if r is not None)
        self.logger.info(f"批量生成完成: {success_count}/{len(texts)} 成功")
        
        return results
    
    def get_available_voices(self) -> list:
        """获取可用的语音预设列表"""
        return self.VOICE_PRESETS.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "device": self.device,
            "model_size": self.model_size,
            "use_gpu": self.use_gpu,
            "models_loaded": self._models_loaded,
            "bark_sample_rate": self.bark_sample_rate,
            "target_sample_rate": self.target_sample_rate,
            "default_voice": self._default_voice,
            "available_voices": len(self.VOICE_PRESETS)
        }
    
    def set_default_voice(self, voice_preset: str):
        """设置默认语音预设"""
        if voice_preset in self.VOICE_PRESETS:
            self._default_voice = voice_preset
            self.logger.info(f"默认语音设置为: {voice_preset}")
        else:
            raise ValueError(f"无效的语音预设: {voice_preset}")
    
    def clear_cache(self):
        """清理模型缓存"""
        self.logger.info("清理Bark模型缓存...")
        
        # Bark模型通常会自动管理缓存，这里主要清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._models_loaded = False
        self.logger.info("缓存清理完成")


# 便捷函数
def create_bark_generator(device: str = None, 
                         target_sample_rate: int = 16000) -> BarkGenerator:
    """
    创建Bark生成器的便捷函数
    
    Args:
        device: 计算设备
        target_sample_rate: 目标采样率
        
    Returns:
        BarkGenerator: Bark生成器实例
    """
    return BarkGenerator(device=device, target_sample_rate=target_sample_rate)


if __name__ == "__main__":
    # 简单测试
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("测试BarkGenerator...")
    
    if not HAS_BARK:
        print("❌ Bark未安装，请先安装Bark模块")
        print("安装命令: pip install git+https://github.com/suno-ai/bark.git")
        exit(1)
    
    try:
        # 创建生成器
        generator = create_bark_generator()
        
        # 显示模型信息
        info = generator.get_model_info()
        print("模型信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 显示可用语音
        voices = generator.get_available_voices()
        print(f"\n可用语音预设: {len(voices)} 个")
        print("前10个:", voices[:10])
        
        # 测试文本生成
        test_texts = [
            "Hello, this is a test of Bark text-to-speech generation.",
            "你好，这是Bark文本转语音生成的测试。"
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\n测试 {i+1}: '{text}'")
            
            try:
                # 生成音频
                audio_tensor = generator.generate_audio(
                    text, 
                    temperature=0.7,
                    seed=42  # 使用固定种子确保可重复性
                )
                
                print(f"生成成功 - 形状: {audio_tensor.shape}")
                print(f"时长: {audio_tensor.size(-1) / 16000:.2f}秒")
                
            except Exception as e:
                print(f"生成失败: {e}")
        
        print("\n✅ BarkGenerator测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()