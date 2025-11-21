"""
模型管理器模块 - 统一管理AI模型的加载、缓存和内存管理
"""

import gc
import os
import torch
import psutil
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
import logging
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM,
    PreTrainedModel, PreTrainedTokenizer
)
from diffusers import StableDiffusionPipeline, DDIMScheduler

logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器 - 提供统一的模型加载、缓存和内存管理"""
    
    def __init__(self, cache_dir: str = "models", max_memory_usage: float = 0.8):
        """
        初始化模型管理器
        
        Args:
            cache_dir: 模型缓存目录
            max_memory_usage: 最大内存使用率（0-1之间）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_memory_usage = max_memory_usage
        
        # 模型缓存字典
        self.loaded_models = {}  # {model_name: (model, tokenizer, metadata)}
        self.model_metadata = {}  # {model_name: {size, device, type, last_used}}
        
        # 设备管理
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        logger.info(f"ModelManager初始化完成，使用设备: {self.device}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取当前内存使用情况
        
        Returns:
            内存使用统计字典
        """
        memory_info = {}
        
        # 系统内存
        system_memory = psutil.virtual_memory()
        memory_info['system'] = {
            'total': system_memory.total / (1024**3),  # GB
            'used': system_memory.used / (1024**3),
            'percent': system_memory.percent
        }
        
        # GPU内存（如果可用）
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            memory_info['gpu'] = {
                'allocated': torch.cuda.memory_allocated() / (1024**3),
                'cached': torch.cuda.memory_reserved() / (1024**3),
                'max_allocated': torch.cuda.max_memory_allocated() / (1024**3)
            }
        
        return memory_info
    
    def check_memory_pressure(self) -> bool:
        """
        检查是否存在内存压力
        
        Returns:
            True表示内存紧张，需要释放模型
        """
        memory_info = self.get_memory_usage()
        system_usage = memory_info['system']['percent'] / 100
        
        if torch.cuda.is_available():
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_usage = memory_info['gpu']['allocated'] / gpu_total
            return system_usage > self.max_memory_usage or gpu_usage > self.max_memory_usage
        
        return system_usage > self.max_memory_usage
    
    def free_least_used_model(self):
        """释放最少使用的模型以节省内存"""
        if not self.loaded_models:
            return
        
        # 找到最少使用的模型
        least_used_model = min(
            self.model_metadata.keys(),
            key=lambda x: self.model_metadata[x].get('last_used', 0)
        )
        
        logger.info(f"释放最少使用的模型: {least_used_model}")
        self.unload_model(least_used_model)
    
    def load_text_model(self, 
                       model_name: str, 
                       model_class: str = "causal_lm",
                       force_reload: bool = False) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        加载文本模型
        
        Args:
            model_name: 模型名称或路径
            model_class: 模型类型 ('causal_lm', 'base', 'seq2seq')
            force_reload: 是否强制重新加载
            
        Returns:
            (model, tokenizer) 元组
        """
        cache_key = f"text_{model_name}_{model_class}"
        
        # 检查缓存
        if cache_key in self.loaded_models and not force_reload:
            model, tokenizer, _ = self.loaded_models[cache_key]
            self.model_metadata[cache_key]['last_used'] = torch.cuda.Event(enable_timing=True)
            return model, tokenizer
        
        # 检查内存压力
        if self.check_memory_pressure():
            self.free_least_used_model()
        
        logger.info(f"加载文本模型: {model_name} ({model_class})")
        
        try:
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # 确保tokenizer有pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 根据模型类型加载模型
            if model_class == "causal_lm":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
            elif model_class == "base":
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
            else:
                raise ValueError(f"不支持的模型类型: {model_class}")
            
            # 移动到设备
            if not torch.cuda.is_available() or model.device.type == 'cpu':
                model = model.to(self.device)
            
            # 设置评估模式
            model.eval()
            
            # 缓存模型
            metadata = {
                'type': 'text',
                'class': model_class,
                'device': str(model.device),
                'size': sum(p.numel() for p in model.parameters()) / 1e6,  # M parameters
                'last_used': torch.cuda.Event(enable_timing=True)
            }
            
            self.loaded_models[cache_key] = (model, tokenizer, metadata)
            self.model_metadata[cache_key] = metadata
            
            logger.info(f"文本模型加载完成: {model_name}, 参数量: {metadata['size']:.1f}M")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"加载文本模型失败 {model_name}: {str(e)}")
            raise
    
    def load_diffusion_model(self, 
                           model_name: str, 
                           scheduler_type: str = "ddim",
                           force_reload: bool = False) -> StableDiffusionPipeline:
        """
        加载扩散模型
        
        Args:
            model_name: 模型名称或路径
            scheduler_type: 调度器类型
            force_reload: 是否强制重新加载
            
        Returns:
            扩散模型pipeline
        """
        cache_key = f"diffusion_{model_name}_{scheduler_type}"
        
        # 检查缓存
        if cache_key in self.loaded_models and not force_reload:
            pipeline, _, _ = self.loaded_models[cache_key]
            self.model_metadata[cache_key]['last_used'] = torch.cuda.Event(enable_timing=True)
            return pipeline
        
        # 检查内存压力
        if self.check_memory_pressure():
            self.free_least_used_model()
        
        logger.info(f"加载扩散模型: {model_name}")
        
        try:
            # 强制离线模式，禁止联网下载
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['DIFFUSERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'

            def _candidate_hub_dirs() -> list:
                candidates = []
                if os.getenv('HF_HOME'):
                    candidates.append(Path(os.getenv('HF_HOME')) / 'hub')
                if os.getenv('HF_HUB_CACHE'):
                    candidates.append(Path(os.getenv('HF_HUB_CACHE')))
                # 项目内 models 目录
                candidates.append(Path(__file__).resolve().parents[3] / 'models')
                # 用户默认缓存
                candidates.append(Path.home() / '.cache' / 'huggingface' / 'hub')
                # 去重并保序
                ordered = []
                seen = set()
                for p in candidates:
                    if p and str(p) not in seen:
                        seen.add(str(p))
                        ordered.append(p)
                return ordered

            def _resolve_local_model_path(model_name_str: str) -> Path:
                # 1) 直接是本地目录
                p = Path(model_name_str)
                if p.exists() and (p / 'model_index.json').exists():
                    return p
                # 2) 在候选hub缓存中查找 models--org--repo 结构
                if '/' in model_name_str:
                    hub_subdir = f"models--{model_name_str.replace('/', '--')}"
                    for base in _candidate_hub_dirs():
                        hub_dir = base / hub_subdir
                        if hub_dir.exists():
                            # 检查snapshots目录，找到实际的模型文件
                            snapshots_dir = hub_dir / 'snapshots'
                            if snapshots_dir.exists():
                                # 获取最新的snapshot
                                snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                                if snapshot_dirs:
                                    # 使用第一个snapshot（通常只有一个）
                                    actual_model_path = snapshot_dirs[0]
                                    if (actual_model_path / 'model_index.json').exists():
                                        logger.info(f"找到模型路径: {actual_model_path}")
                                        return actual_model_path
                            # 如果没有找到snapshot，尝试直接返回hub_dir让diffusers自己解析
                            return hub_dir
                # 3) 回退原始字符串
                return Path(model_name_str)

            model_path = _resolve_local_model_path(model_name)
            
            # 加载pipeline
            pipeline = StableDiffusionPipeline.from_pretrained(
                str(model_path),
                cache_dir=self.cache_dir,
                torch_dtype=self.torch_dtype,
                safety_checker=None,  # 关闭安全检查以节省内存
                requires_safety_checker=False,
                local_files_only=True
            )
            
            # 设置调度器
            if scheduler_type == "ddim":
                pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            
            # 移动到设备
            pipeline = pipeline.to(self.device)
            
            # 启用内存优化
            if torch.cuda.is_available():
                pipeline.enable_attention_slicing()
                if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                    try:
                        pipeline.enable_xformers_memory_efficient_attention()
                    except:
                        pass
            
            # 缓存模型
            metadata = {
                'type': 'diffusion',
                'scheduler': scheduler_type,
                'device': str(self.device),
                'size': 'large',  # 扩散模型通常较大
                'last_used': torch.cuda.Event(enable_timing=True)
            }
            
            self.loaded_models[cache_key] = (pipeline, None, metadata)
            self.model_metadata[cache_key] = metadata
            
            logger.info(f"扩散模型加载完成: {model_name}")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"加载扩散模型失败 {model_name}: {str(e)}")
            raise
    
    def load_decoder_model(self, 
                          decoder_path: str, 
                          force_reload: bool = False) -> torch.nn.Module:
        """
        加载解码器模型（用于图像水印提取）
        
        Args:
            decoder_path: 解码器模型路径
            force_reload: 是否强制重新加载
            
        Returns:
            解码器模型
        """
        cache_key = f"decoder_{decoder_path}"
        
        # 检查缓存
        if cache_key in self.loaded_models and not force_reload:
            decoder, _, _ = self.loaded_models[cache_key]
            self.model_metadata[cache_key]['last_used'] = torch.cuda.Event(enable_timing=True)
            return decoder
        
        logger.info(f"加载解码器模型: {decoder_path}")
        
        try:
            # 检查文件存在
            decoder_path = Path(decoder_path)
            if not decoder_path.exists():
                raise FileNotFoundError(f"解码器文件不存在: {decoder_path}")
            
            # 加载模型
            if decoder_path.suffix == '.pt' or decoder_path.suffix == '.pth':
                decoder = torch.jit.load(decoder_path, map_location=self.device)
            else:
                decoder = torch.load(decoder_path, map_location=self.device)
            
            decoder.eval()
            
            # 缓存模型
            metadata = {
                'type': 'decoder',
                'device': str(self.device),
                'size': 'small',
                'last_used': torch.cuda.Event(enable_timing=True)
            }
            
            self.loaded_models[cache_key] = (decoder, None, metadata)
            self.model_metadata[cache_key] = metadata
            
            logger.info(f"解码器模型加载完成: {decoder_path}")
            
            return decoder
            
        except Exception as e:
            logger.error(f"加载解码器模型失败 {decoder_path}: {str(e)}")
            raise
    
    def unload_model(self, model_key: str):
        """
        卸载指定模型
        
        Args:
            model_key: 模型缓存键
        """
        if model_key in self.loaded_models:
            logger.info(f"卸载模型: {model_key}")
            
            model_data = self.loaded_models[model_key]
            
            # 清理模型引用
            for item in model_data:
                if hasattr(item, 'cpu'):
                    item.cpu()
                del item
            
            # 从缓存中移除
            del self.loaded_models[model_key]
            del self.model_metadata[model_key]
            
            # 强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def unload_all_models(self):
        """卸载所有模型"""
        logger.info("卸载所有模型")
        
        model_keys = list(self.loaded_models.keys())
        for key in model_keys:
            self.unload_model(key)
    
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """
        获取已加载的模型信息
        
        Returns:
            模型信息字典
        """
        return {key: metadata.copy() for key, metadata in self.model_metadata.items()}
    
    def optimize_memory(self):
        """内存优化：释放不必要的模型"""
        if self.check_memory_pressure():
            # 按最后使用时间排序，释放最老的模型
            sorted_models = sorted(
                self.model_metadata.items(),
                key=lambda x: x[1].get('last_used', 0)
            )
            
            # 释放最老的50%模型
            models_to_remove = len(sorted_models) // 2
            for i in range(models_to_remove):
                model_key = sorted_models[i][0]
                self.unload_model(model_key)
    
    def __del__(self):
        """析构函数：清理所有模型"""
        try:
            self.unload_all_models()
        except:
            pass


# 全局模型管理器实例
_global_manager = None

def get_global_manager() -> ModelManager:
    """获取全局模型管理器实例"""
    global _global_manager
    if _global_manager is None:
        _global_manager = ModelManager()
    return _global_manager 