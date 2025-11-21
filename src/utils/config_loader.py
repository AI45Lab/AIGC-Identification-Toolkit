"""
配置加载器模块 - 统一管理配置文件的加载、验证和合并
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path


class ConfigLoader:
    """配置加载器类 - 提供灵活的配置管理功能"""
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.loaded_configs = {}  # 缓存已加载的配置
        
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载单个配置文件
        
        Args:
            config_path: 配置文件路径（支持相对路径和绝对路径）
            
        Returns:
            配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML格式错误
        """
        # 处理路径
        if not Path(config_path).is_absolute():
            config_path = self.config_dir / config_path
        else:
            config_path = Path(config_path)
            
        # 检查缓存
        cache_key = str(config_path.absolute())
        if cache_key in self.loaded_configs:
            return self.loaded_configs[cache_key].copy()
            
        # 检查文件存在
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 根据文件扩展名选择加载方式
        try:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
                
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"配置文件格式错误 {config_path}: {str(e)}")
        
        # 缓存配置
        self.loaded_configs[cache_key] = config.copy()
        
        return config
    
    def load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return self.load_config("default_config.yaml")
    
    def merge_configs(self, *config_paths: str) -> Dict[str, Any]:
        """
        合并多个配置文件
        
        Args:
            *config_paths: 配置文件路径列表
            
        Returns:
            合并后的配置字典
        """
        merged_config = {}
        
        for config_path in config_paths:
            config = self.load_config(config_path)
            merged_config = self._deep_merge(merged_config, config)
            
        return merged_config
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """
        深度合并字典
        
        Args:
            base_dict: 基础字典
            update_dict: 更新字典
            
        Returns:
            合并后的字典
        """
        result = base_dict.copy()
        
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def validate_config(self, config: Dict[str, Any], required_keys: list) -> bool:
        """
        验证配置是否包含必需的键
        
        Args:
            config: 配置字典
            required_keys: 必需的键列表（支持嵌套键，如 'text_watermark.algorithm'）
            
        Returns:
            验证是否通过
            
        Raises:
            ValueError: 缺少必需的配置项
        """
        missing_keys = []
        
        for key in required_keys:
            if not self._check_nested_key(config, key):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"配置缺少必需项: {missing_keys}")
            
        return True
    
    def _check_nested_key(self, config: Dict, key: str) -> bool:
        """检查嵌套键是否存在"""
        keys = key.split('.')
        current = config
        
        for k in keys:
            if not isinstance(current, dict) or k not in current:
                return False
            current = current[k]
            
        return True
    
    def get_nested_value(self, config: Dict[str, Any], key: str, default: Any = None) -> Any:
        """
        获取嵌套键的值
        
        Args:
            config: 配置字典
            key: 嵌套键（如 'text_watermark.credid.watermark_key'）
            default: 默认值
            
        Returns:
            键对应的值或默认值
        """
        keys = key.split('.')
        current = config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
        """
        设置嵌套键的值
        
        Args:
            config: 配置字典
            key: 嵌套键
            value: 要设置的值
            
        Returns:
            更新后的配置字典
        """
        keys = key.split('.')
        current = config
        
        # 导航到倒数第二个键
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        # 设置最后一个键的值
        current[keys[-1]] = value
        
        return config
    
    def clear_cache(self):
        """清除配置缓存"""
        self.loaded_configs.clear()


# 全局配置加载器实例
_global_loader = None

def get_global_loader() -> ConfigLoader:
    """获取全局配置加载器实例"""
    global _global_loader
    if _global_loader is None:
        _global_loader = ConfigLoader()
    return _global_loader

def load_config(config_path: str) -> Dict[str, Any]:
    """
    便捷函数：加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    loader = get_global_loader()
    return loader.load_config(config_path) 