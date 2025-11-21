"""
PRC水印核心封装类 - 实现统一的embed和extract接口
"""

import os
import pickle
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Union, Tuple
from src.utils.path_manager import path_manager

# 导入PRC核心算法
import sys
prc_path = os.path.join(os.path.dirname(__file__), 'PRC-Watermark')
if prc_path not in sys.path:
    sys.path.insert(0, prc_path)

# 测试导入
PRC_AVAILABLE = False
original_cwd = os.getcwd()

try:
    # 先切换工作目录到PRC-Watermark
    os.chdir(prc_path)
    
    # 临时移除我们项目的src路径，避免包名冲突
    temp_path = sys.path.copy()
    sys.path = [p for p in sys.path if not p.endswith('unified_watermark_tool')]
    sys.path.insert(0, prc_path)
    
    from src.prc import KeyGen, Encode, Detect, Decode, str_to_bin, bin_to_str
    import src.pseudogaussians as prc_gaussians
    from inversion import stable_diffusion_pipe, generate, exact_inversion
    from src.optim_utils import transform_img
    
    PRC_AVAILABLE = True
    print("✓ PRC-Watermark核心模块导入成功")
    
except ImportError as e:
    print(f"Warning: PRC-Watermark dependencies not available: {e}")
    print("Please install required packages first")
    
finally:
    # 恢复原始状态
    if 'temp_path' in locals():
        sys.path = temp_path
    os.chdir(original_cwd)

# 如果导入失败，提供占位符函数
if not PRC_AVAILABLE:
    def KeyGen(*args, **kwargs):
        raise RuntimeError("PRC dependencies not installed")
    def Encode(*args, **kwargs):
        raise RuntimeError("PRC dependencies not installed")
    def Detect(*args, **kwargs):
        raise RuntimeError("PRC dependencies not installed")
    def Decode(*args, **kwargs):
        raise RuntimeError("PRC dependencies not installed")
    def str_to_bin(*args, **kwargs):
        raise RuntimeError("PRC dependencies not installed")
    def bin_to_str(*args, **kwargs):
        raise RuntimeError("PRC dependencies not installed")
    def stable_diffusion_pipe(*args, **kwargs):
        raise RuntimeError("PRC dependencies not installed")
    def generate(*args, **kwargs):
        raise RuntimeError("PRC dependencies not installed")
    def exact_inversion(*args, **kwargs):
        raise RuntimeError("PRC dependencies not installed")
    def transform_img(*args, **kwargs):
        raise RuntimeError("PRC dependencies not installed")
    
    class MockPseudogaussians:
        @staticmethod
        def sample(*args, **kwargs):
            raise RuntimeError("PRC dependencies not installed")
        @staticmethod
        def recover_posteriors(*args, **kwargs):
            raise RuntimeError("PRC dependencies not installed")
    
    prc_gaussians = MockPseudogaussians()


class PRCWatermark:
    """PRC水印处理类 - 统一接口"""
    
    def __init__(self, 
                 n: int = 4 * 64 * 64,
                 false_positive_rate: float = 1e-5,
                 t: int = 3,
                 model_id: str = 'stabilityai/stable-diffusion-2-1-base',
                 cache_dir: str = None,
                 keys_dir: str = 'keys',
                 device: str = None):
        """
        初始化PRC水印处理器
        
        Args:
            n: PRC码字长度，默认为4*64*64
            false_positive_rate: 允许的假阳性率
            t: 奇偶校验稀疏度
            model_id: 使用的Stable Diffusion模型ID
            cache_dir: 模型缓存目录
            keys_dir: 密钥存储目录
            device: 计算设备
        """
        self.n = n
        self.false_positive_rate = false_positive_rate
        self.t = t
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.keys_dir = keys_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 确保密钥目录存在
        if not os.path.exists(self.keys_dir):
            os.makedirs(self.keys_dir)
            
        # 初始化Stable Diffusion管道
        self._setup_diffusion_pipe()
        
        # 水印密钥缓存
        self._key_cache = {}
    
    def _setup_diffusion_pipe(self):
        """初始化Stable Diffusion管道"""
        if not PRC_AVAILABLE:
            raise RuntimeError("PRC dependencies not available, cannot setup diffusion pipeline")

        # 设置正确的缓存目录
        cache_dir = self.cache_dir or str(path_manager.get_hf_hub_dir())
        
        # 设置离线模式，避免网络连接
        import os
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['DIFFUSERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        # 构建本地模型路径
        local_model_path = os.path.join(cache_dir, f"models--{self.model_id.replace('/', '--')}")
        
        self.pipe = stable_diffusion_pipe(
            solver_order=1, 
            model_id=local_model_path if os.path.exists(local_model_path) else self.model_id, 
            cache_dir=cache_dir
        )
        self.pipe.set_progress_bar_config(disable=True)
    
    def _get_or_create_keys(self, key_id: str, message_length: int = 512) -> Tuple[Any, Any]:
        """获取或创建PRC密钥对"""
        # 检查缓存
        if key_id in self._key_cache:
            return self._key_cache[key_id]
        
        # 密钥文件路径
        key_file = os.path.join(self.keys_dir, f'{key_id}.pkl')
        
        if os.path.exists(key_file):
            # 从文件加载密钥
            with open(key_file, 'rb') as f:
                encoding_key, decoding_key = pickle.load(f)
        else:
            # 生成新密钥
            encoding_key, decoding_key = KeyGen(
                n=self.n,
                message_length=message_length,
                false_positive_rate=self.false_positive_rate,
                t=self.t
            )
            # 保存密钥到文件
            with open(key_file, 'wb') as f:
                pickle.dump((encoding_key, decoding_key), f)
        
        # 缓存密钥
        self._key_cache[key_id] = (encoding_key, decoding_key)
        return encoding_key, decoding_key
    
    def _seed_everything(self, seed: int):
        """设置所有随机种子"""
        import random
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def embed(self, 
              prompt: str, 
              message: Optional[str] = None,
              key_id: str = "default",
              num_inference_steps: int = 50,
              seed: Optional[int] = None,
              **kwargs) -> Image.Image:
        """
        嵌入水印并生成图像
        
        Args:
            prompt: 图像生成提示词
            message: 要嵌入的消息（字符串），如果为None则生成随机消息
            key_id: 密钥标识符
            num_inference_steps: 推理步数
            seed: 随机种子
            **kwargs: 其他参数
            
        Returns:
            含水印的生成图像
        """
        # 设置随机种子
        if seed is not None:
            self._seed_everything(seed)
        
        # 处理消息
        if message is not None:
            # 将字符串消息转换为二进制
            message_bits = str_to_bin(message)
            message_length = len(message_bits)
        else:
            message_bits = None
            message_length = 512  # 默认消息长度
        
        # 获取密钥
        encoding_key, _ = self._get_or_create_keys(key_id, message_length)
        
        # 生成PRC码字
        if message_bits is not None:
            prc_codeword = Encode(encoding_key, message_bits)
        else:
            prc_codeword = Encode(encoding_key)
        
        # 从PRC码字采样初始潜变量
        init_latents = prc_gaussians.sample(prc_codeword).reshape(1, 4, 64, 64).to(self.device)
        
        # 生成图像
        image, _, _ = generate(
            prompt=prompt,
            init_latents=init_latents,
            num_inference_steps=num_inference_steps,
            solver_order=1,
            pipe=self.pipe
        )
        
        return image
    
    def extract(self, 
                image: Union[str, Image.Image, torch.Tensor],
                key_id: str = "default", 
                mode: str = 'exact',
                prompt: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        """
        从图像中提取水印
        
        Args:
            image: 输入图像（文件路径、PIL Image或torch.Tensor）
            key_id: 密钥标识符
            mode: 逆向模式 ('fast', 'accurate', 'exact')
            prompt: 用于exact模式的提示词（可选）
            **kwargs: 其他参数
            
        Returns:
            提取结果字典，包含：
            - detected: 是否检测到水印
            - message: 解码的消息（如果检测到且解码成功）
            - confidence: 检测置信度
            - mode_used: 实际使用的逆向模式
        """
        # 获取密钥
        _, decoding_key = self._get_or_create_keys(key_id)
        
        # 处理输入图像
        if isinstance(image, str):
            # 从文件路径加载图像
            pil_image = Image.open(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, torch.Tensor):
            # 假设输入是潜变量
            latents = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # 如果输入是PIL图像，需要使用逆向过程获取潜变量
        if not isinstance(image, torch.Tensor):
            latents = self._image_to_latents(pil_image, mode=mode, prompt=prompt)
        
        # 计算后验概率 - 确保tensor在CPU上且分离梯度
        latents_cpu = latents.detach().cpu() if hasattr(latents, 'detach') else latents
        if hasattr(latents_cpu, 'cpu'):
            latents_cpu = latents_cpu.cpu()
        posteriors = prc_gaussians.recover_posteriors(latents_cpu.flatten())
        
        # 检测水印
        detected = Detect(decoding_key, posteriors, self.false_positive_rate)
        
        result = {
            'detected': detected,
            'message': None,
            'confidence': 0.0,
            'mode_used': mode if not isinstance(image, torch.Tensor) else 'tensor_input'
        }
        
        if detected:
            # 如果检测到水印，尝试解码消息
            decoded_bits = Decode(decoding_key, posteriors)
            
            if decoded_bits is not None:
                try:
                    # 将二进制转换为字符串
                    decoded_message = bin_to_str(decoded_bits)
                    result['message'] = decoded_message
                    result['confidence'] = 1.0
                except:
                    # 解码失败，可能是随机消息
                    result['message'] = f"Random message (bits: {decoded_bits.tolist()})"
                    result['confidence'] = 0.8
            else:
                # 检测到但解码失败
                result['confidence'] = 0.6
        
        return result
    
    def _image_to_latents(self, image: Image.Image, mode: str = 'accurate', prompt: Optional[str] = None) -> torch.Tensor:
        """
        将PIL图像转换为潜变量，使用exact_inversion函数
        
        Args:
            image: PIL图像
            mode: 逆向模式 ('fast', 'accurate', 'exact')
            prompt: 提示词（可选，默认为空字符串）
            
        Returns:
            潜变量tensor
        """
        if not PRC_AVAILABLE:
            raise RuntimeError("PRC dependencies not available")
            
        if prompt is None:
            prompt = ""  # 使用空提示词作为默认值
            
        # 根据模式设置不同的参数
        if mode == 'fast':
            # 快速模式：使用较少的推理步数和简单逆向
            decoder_inv = False
            num_inference_steps = 20
            test_num_inference_steps = 20
        elif mode == 'accurate':
            # 精确模式：使用decoder_inv优化求解
            decoder_inv = True
            num_inference_steps = 50
            test_num_inference_steps = 50
        elif mode == 'exact':
            # 完整模式：最高精度设置
            decoder_inv = True
            num_inference_steps = 50
            test_num_inference_steps = 50
        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose from 'fast', 'accurate', 'exact'")
        
        # 使用PRC-Watermark的exact_inversion函数
        reversed_latents = exact_inversion(
            image=image,
            prompt=prompt, 
            guidance_scale=3.0,
            num_inference_steps=num_inference_steps,
            solver_order=1,
            test_num_inference_steps=test_num_inference_steps,
            inv_order=1,
            decoder_inv=decoder_inv,
            model_id=self.model_id,
            pipe=self.pipe
        )
        
        return reversed_latents
    
    def generate_key(self, key_id: str, message_length: int = 512) -> str:
        """
        生成新的密钥对
        
        Args:
            key_id: 密钥标识符
            message_length: 支持的最大消息长度
            
        Returns:
            密钥文件路径
        """
        encoding_key, decoding_key = KeyGen(
            n=self.n,
            message_length=message_length,
            false_positive_rate=self.false_positive_rate,
            t=self.t
        )
        
        key_file = os.path.join(self.keys_dir, f'{key_id}.pkl')
        with open(key_file, 'wb') as f:
            pickle.dump((encoding_key, decoding_key), f)
        
        self._key_cache[key_id] = (encoding_key, decoding_key)
        
        return key_file