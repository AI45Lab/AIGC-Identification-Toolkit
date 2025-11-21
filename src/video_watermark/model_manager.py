"""
视频生成模型下载和缓存管理器
自动处理模型的下载、缓存和加载
支持HuggingFace镜像站点
支持的模型: HunyuanVideo, Wan2.1
"""

import os
import logging
from typing import Optional
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logging.warning("huggingface_hub not available. Model downloading will be disabled.")

from src.utils.path_manager import path_manager


class ModelManager:
    """视频生成模型管理器（支持HunyuanVideo和Wan2.1）"""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化模型管理器

        Args:
            cache_dir: HuggingFace模型缓存目录（None则使用环境变量或默认路径）
        """
        # Use path_manager to resolve cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = path_manager.get_hf_hub_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # HunyuanVideo模型配置（支持多个仓库源）
        self.hunyuan_repos = [
            "hunyuanvideo-community/HunyuanVideo",  # 社区diffusers兼容版本
            "tencent/HunyuanVideo"  # 官方版本
        ]
        self.hunyuan_repo = self.hunyuan_repos[0]  # 默认使用社区版本
        self.hunyuan_model_dir = self.cache_dir / "models--hunyuanvideo-community--HunyuanVideo"

        # Wan2.1模型配置
        self.wan_repo = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        self.wan_model_dir = self.cache_dir / "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 设置镜像站点环境变量（如果未设置）
        if not os.environ.get('HF_ENDPOINT'):
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            self.logger.info("设置HF_ENDPOINT为镜像站点: https://hf-mirror.com")
        
    def _check_local_model_exists(self) -> bool:
        """检查本地是否存在HunyuanVideo模型"""
        # 检查多个可能的路径格式（包括社区版本和官方版本）
        possible_paths = [
            self.cache_dir / "models--hunyuanvideo-community--HunyuanVideo",
            self.cache_dir / "models--tencent--HunyuanVideo",
            self.cache_dir / "tencent--HunyuanVideo",
            self.cache_dir / "hub" / "models--tencent--HunyuanVideo",
            self.cache_dir / "hub" / "models--hunyuanvideo-community--HunyuanVideo"
        ]
        for path in possible_paths:
            if path.exists() and any(path.iterdir()):
                self.logger.info(f"发现本地HunyuanVideo模型: {path}")
                # 更新实际路径
                self.hunyuan_model_dir = path
                return True
        
        return False
    
    def _find_actual_model_path(self) -> Optional[Path]:
        """查找实际的模型文件路径"""
        if not self.hunyuan_model_dir.exists():
            return None
            
        # 查找模型文件（递归搜索）
        for root, dirs, files in os.walk(self.hunyuan_model_dir):
            # 查找关键文件（如config.json, pytorch_model.bin等）
            key_files = ['config.json', 'model_index.json', 'scheduler']
            if any(f in files or f in dirs for f in key_files):
                return Path(root)
        
        # 如果没有找到关键文件，返回根目录
        if any(self.hunyuan_model_dir.iterdir()):
            return self.hunyuan_model_dir
            
        return None
    
    def ensure_hunyuan_model(self, allow_download: bool = True) -> str:
        """
        确保HunyuanVideo模型可用，如果不存在则下载（可选）
        
        Args:
            allow_download: 是否允许下载模型（默认True）
        
        Returns:
            str: 本地模型路径
            
        Raises:
            RuntimeError: 模型不存在且不允许下载，或下载失败
        """
        # 检查本地模型
        if self._check_local_model_exists():
            actual_path = self._find_actual_model_path()
            if actual_path:
                self.logger.info(f"使用本地HunyuanVideo模型: {actual_path}")
                return str(actual_path)
        
        # 模型不存在
        if not allow_download:
            raise RuntimeError(
                f"HunyuanVideo模型不存在于: {self.cache_dir}\n"
                "请手动下载模型或设置allow_download=True启用自动下载"
            )
        
        # 需要下载模型
        if not HF_HUB_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub not available. Please install with: pip install huggingface_hub"
            )
        
        self.logger.info(f"开始从镜像站点下载HunyuanVideo模型到: {self.cache_dir}")
        self.logger.info(f"使用HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'default')}")
        
        # 尝试多个仓库源
        for repo_id in self.hunyuan_repos:
            try:
                self.logger.info(f"尝试下载仓库: {repo_id}")
                downloaded_path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=str(self.cache_dir),
                    resume_download=True,  # 支持断点续传
                    local_files_only=False,
                    # force_download=False,  # 不强制重新下载
                )
                
                self.logger.info(f"HunyuanVideo模型下载完成: {downloaded_path}")
                # 更新当前使用的仓库
                self.hunyuan_repo = repo_id
                return downloaded_path
                
            except Exception as e:
                self.logger.warning(f"从 {repo_id} 下载失败: {e}")
                continue
        
        # 如果所有仓库都失败
        raise RuntimeError("所有HunyuanVideo仓库都下载失败")
    
    def get_model_path(self) -> str:
        """
        获取HunyuanVideo模型路径（不触发下载）
        
        Returns:
            str: 模型路径，如果不存在返回空字符串
        """
        if self._check_local_model_exists():
            actual_path = self._find_actual_model_path()
            if actual_path:
                return str(actual_path)
        return ""
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            dict: 包含模型信息的字典
        """
        model_path = self.get_model_path()
        
        info = {
            "repo_id": self.hunyuan_repo,
            "cache_dir": str(self.cache_dir),
            "local_path": model_path,
            "exists": bool(model_path),
            "huggingface_hub_available": HF_HUB_AVAILABLE
        }
        
        # 如果模型存在，获取更多信息
        if model_path:
            model_path_obj = Path(model_path)
            info.update({
                "size_mb": sum(f.stat().st_size for f in model_path_obj.rglob('*') if f.is_file()) / (1024*1024),
                "num_files": len([f for f in model_path_obj.rglob('*') if f.is_file()])
            })
        
        return info
    
    def _check_wan_model_exists(self) -> bool:
        """检查本地是否存在Wan2.1模型"""
        # 检查多个可能的路径格式
        possible_paths = [
            self.cache_dir / "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers",
            self.cache_dir / "Wan-AI--Wan2.1-T2V-1.3B-Diffusers",
            self.cache_dir / "hub" / "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
        ]
        for path in possible_paths:
            if path.exists() and any(path.iterdir()):
                self.logger.info(f"发现本地Wan2.1模型: {path}")
                # 更新实际路径
                self.wan_model_dir = path
                return True

        return False

    def _find_actual_model_path_wan(self) -> Optional[Path]:
        """查找实际的Wan2.1模型文件路径"""
        if not self.wan_model_dir.exists():
            return None

        # 查找模型文件（递归搜索）
        for root, dirs, files in os.walk(self.wan_model_dir):
            # 查找关键文件（如config.json, model_index.json等）
            key_files = ['config.json', 'model_index.json', 'scheduler', 'vae']
            if any(f in files or f in dirs for f in key_files):
                return Path(root)

        # 如果没有找到关键文件，返回根目录
        if any(self.wan_model_dir.iterdir()):
            return self.wan_model_dir

        return None

    def ensure_wan_model(self, allow_download: bool = False) -> str:
        """
        确保Wan2.1模型可用，如果不存在则抛出错误

        Args:
            allow_download: 是否允许下载模型（默认False，仅使用本地模型）

        Returns:
            str: 本地模型路径

        Raises:
            RuntimeError: 模型不存在且不允许下载，或下载失败
        """
        # 检查本地模型
        if self._check_wan_model_exists():
            actual_path = self._find_actual_model_path_wan()
            if actual_path:
                self.logger.info(f"使用本地Wan2.1模型: {actual_path}")
                return str(actual_path)

        # 模型不存在
        if not allow_download:
            raise RuntimeError(
                f"Wan2.1模型不存在于: {self.cache_dir}\n"
                f"预期路径: {self.wan_model_dir}\n"
                "请手动下载模型到该目录，或从HuggingFace下载:\n"
                f"  huggingface-cli download {self.wan_repo} --local-dir {self.wan_model_dir}"
            )

        # 需要下载模型
        if not HF_HUB_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub not available. Please install with: pip install huggingface_hub"
            )

        self.logger.info(f"开始从镜像站点下载Wan2.1模型到: {self.cache_dir}")
        self.logger.info(f"使用HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'default')}")

        try:
            self.logger.info(f"下载仓库: {self.wan_repo}")
            downloaded_path = snapshot_download(
                repo_id=self.wan_repo,
                cache_dir=str(self.cache_dir),
                resume_download=True,  # 支持断点续传
                local_files_only=False,
            )

            self.logger.info(f"Wan2.1模型下载完成: {downloaded_path}")
            return downloaded_path

        except Exception as e:
            self.logger.error(f"从 {self.wan_repo} 下载失败: {e}")
            raise RuntimeError(f"Wan2.1模型下载失败: {e}")

    def get_wan_model_path(self) -> str:
        """
        获取Wan2.1模型路径（不触发下载）

        Returns:
            str: 模型路径，如果不存在返回空字符串
        """
        if self._check_wan_model_exists():
            actual_path = self._find_actual_model_path_wan()
            if actual_path:
                return str(actual_path)
        return ""

    def get_wan_model_info(self) -> dict:
        """
        获取Wan2.1模型信息

        Returns:
            dict: 包含模型信息的字典
        """
        model_path = self.get_wan_model_path()

        info = {
            "repo_id": self.wan_repo,
            "cache_dir": str(self.cache_dir),
            "local_path": model_path,
            "exists": bool(model_path),
            "model_size": "1.3B parameters",
            "vram_requirement": "~8GB",
            "huggingface_hub_available": HF_HUB_AVAILABLE
        }

        # 如果模型存在，获取更多信息
        if model_path:
            model_path_obj = Path(model_path)
            info.update({
                "size_mb": sum(f.stat().st_size for f in model_path_obj.rglob('*') if f.is_file()) / (1024*1024),
                "num_files": len([f for f in model_path_obj.rglob('*') if f.is_file()])
            })

        return info

    def clear_cache(self):
        """清理模型缓存（HunyuanVideo和Wan2.1）"""
        if self.hunyuan_model_dir.exists():
            import shutil
            shutil.rmtree(self.hunyuan_model_dir)
            self.logger.info(f"已清理HunyuanVideo模型缓存: {self.hunyuan_model_dir}")

        if self.wan_model_dir.exists():
            import shutil
            shutil.rmtree(self.wan_model_dir)
            self.logger.info(f"已清理Wan2.1模型缓存: {self.wan_model_dir}")


# 方便的工具函数
def get_default_model_manager() -> ModelManager:
    """获取默认的模型管理器实例"""
    return ModelManager()


def ensure_hunyuan_model_available(cache_dir: Optional[str] = None) -> str:
    """
    确保HunyuanVideo模型可用的快捷函数
    
    Args:
        cache_dir: 可选的缓存目录
        
    Returns:
        str: 模型路径
    """
    manager = ModelManager(cache_dir) if cache_dir else get_default_model_manager()
    return manager.ensure_hunyuan_model()


if __name__ == "__main__":
    # 测试代码
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("测试ModelManager...")
    
    manager = ModelManager()
    
    # 显示模型信息
    info = manager.get_model_info()
    print("模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试模型确保功能
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        print("\n开始确保模型可用...")
        try:
            model_path = manager.ensure_hunyuan_model()
            print(f"✅ 模型就绪: {model_path}")
        except Exception as e:
            print(f"❌ 错误: {e}")