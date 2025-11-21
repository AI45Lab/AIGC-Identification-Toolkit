"""
Unified Path Manager for AIGC Identification Toolkit

This module provides cross-platform path resolution for model caches and data directories.
It follows the priority order:
1. Environment variables (user-specified)
2. Configuration files (project-level)
3. Intelligent defaults (cross-platform)

Environment Variables Supported:
- HF_HOME: HuggingFace cache root directory
- HF_HUB_CACHE: HuggingFace Hub models directory
- TRANSFORMERS_CACHE: Transformers-specific cache
- BARK_CACHE_DIR: Bark TTS model cache
- XDG_CACHE_HOME: Standard cache directory (Linux/macOS)
"""

import os
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class PathManager:
    """Centralized path management for AIGC toolkit"""

    def __init__(self):
        """Initialize path manager with default configurations"""
        self._cache_root = None
        self._hf_home = None
        self._hf_hub_cache = None
        self._bark_cache = None

    def get_cache_root(self) -> Path:
        """
        Get the root cache directory for the project.

        Priority:
        1. XDG_CACHE_HOME environment variable
        2. ~/.cache (Linux/macOS default)
        3. %LOCALAPPDATA% (Windows default)

        Returns:
            Path: Root cache directory path
        """
        if self._cache_root is not None:
            return self._cache_root

        if os.getenv('XDG_CACHE_HOME'):
            cache_root = Path(os.getenv('XDG_CACHE_HOME'))
        elif os.name == 'nt':  # Windows
            cache_root = Path(os.getenv('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
        else:  # Linux/macOS
            cache_root = Path.home() / '.cache'

        cache_root.mkdir(parents=True, exist_ok=True)
        self._cache_root = cache_root
        logger.debug(f"Cache root directory: {cache_root}")
        return cache_root

    def get_hf_home(self) -> Path:
        """
        Get HuggingFace home directory.

        Priority:
        1. HF_HOME environment variable
        2. {cache_root}/huggingface

        Returns:
            Path: HuggingFace home directory
        """
        if self._hf_home is not None:
            return self._hf_home

        if os.getenv('HF_HOME'):
            hf_home = Path(os.getenv('HF_HOME'))
        else:
            hf_home = self.get_cache_root() / 'huggingface'

        hf_home.mkdir(parents=True, exist_ok=True)
        self._hf_home = hf_home
        logger.debug(f"HuggingFace home directory: {hf_home}")
        return hf_home

    def get_hf_hub_dir(self) -> Path:
        """
        Get HuggingFace Hub models directory.

        Priority:
        1. HF_HUB_CACHE environment variable
        2. HF_HOME/hub
        3. {cache_root}/huggingface/hub

        Returns:
            Path: HuggingFace Hub directory
        """
        if self._hf_hub_cache is not None:
            return self._hf_hub_cache

        if os.getenv('HF_HUB_CACHE'):
            hub_dir = Path(os.getenv('HF_HUB_CACHE'))
        else:
            hub_dir = self.get_hf_home() / 'hub'

        hub_dir.mkdir(parents=True, exist_ok=True)
        self._hf_hub_cache = hub_dir
        logger.debug(f"HuggingFace Hub directory: {hub_dir}")
        return hub_dir

    def get_transformers_cache(self) -> Path:
        """
        Get Transformers cache directory.

        Priority:
        1. TRANSFORMERS_CACHE environment variable
        2. HF_HOME/transformers

        Returns:
            Path: Transformers cache directory
        """
        if os.getenv('TRANSFORMERS_CACHE'):
            cache_dir = Path(os.getenv('TRANSFORMERS_CACHE'))
        else:
            cache_dir = self.get_hf_home() / 'transformers'

        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Transformers cache directory: {cache_dir}")
        return cache_dir

    def get_bark_cache_dir(self) -> Path:
        """
        Get Bark TTS model cache directory.

        Priority:
        1. BARK_CACHE_DIR environment variable
        2. HF_HOME/hub/models--suno--bark
        3. {cache_root}/bark

        Returns:
            Path: Bark cache directory
        """
        if self._bark_cache is not None:
            return self._bark_cache

        if os.getenv('BARK_CACHE_DIR'):
            bark_cache = Path(os.getenv('BARK_CACHE_DIR'))
        else:
            # Check if Bark model exists in HF Hub cache
            hf_bark_path = self.get_hf_hub_dir() / 'models--suno--bark'
            if hf_bark_path.exists():
                bark_cache = hf_bark_path
            else:
                bark_cache = self.get_cache_root() / 'bark'

        bark_cache.mkdir(parents=True, exist_ok=True)
        self._bark_cache = bark_cache
        logger.debug(f"Bark cache directory: {bark_cache}")
        return bark_cache

    def find_model_in_hub(self, model_id: str) -> Optional[Path]:
        """
        Find a model in HuggingFace Hub cache.

        Args:
            model_id: Model identifier (e.g., "stabilityai/stable-diffusion-2-1-base")

        Returns:
            Path to the model directory if found, None otherwise
        """
        # Convert model_id to HF Hub directory name format
        hub_model_name = f"models--{model_id.replace('/', '--')}"

        # Search in HF Hub cache
        hub_dir = self.get_hf_hub_dir()
        model_dir = hub_dir / hub_model_name

        if model_dir.exists():
            # Check for snapshots subdirectory
            snapshots_dir = model_dir / 'snapshots'
            if snapshots_dir.exists():
                # Return the first snapshot (usually the latest)
                snapshots = list(snapshots_dir.iterdir())
                if snapshots:
                    logger.debug(f"Found model {model_id} at {snapshots[0]}")
                    return snapshots[0]
            logger.debug(f"Found model {model_id} at {model_dir}")
            return model_dir

        logger.debug(f"Model {model_id} not found in cache")
        return None

    def get_candidate_paths(self, env_var: str, default_subpath: str) -> List[Path]:
        """
        Get a list of candidate paths for a resource.

        Args:
            env_var: Environment variable name to check first
            default_subpath: Default subpath under cache root

        Returns:
            List of candidate paths in priority order
        """
        candidates = []

        # Priority 1: Environment variable
        if os.getenv(env_var):
            candidates.append(Path(os.getenv(env_var)))

        # Priority 2: Default path under cache root
        candidates.append(self.get_cache_root() / default_subpath)

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for path in candidates:
            if path not in seen:
                seen.add(path)
                unique_candidates.append(path)

        return unique_candidates

    def resolve_cache_dir(self, cache_dir: Optional[str] = None,
                         env_var: Optional[str] = None,
                         default_subpath: str = 'huggingface/hub') -> Path:
        """
        Resolve a cache directory with flexible priority.

        Priority:
        1. Explicit cache_dir parameter
        2. Environment variable (if provided)
        3. Default subpath under cache root

        Args:
            cache_dir: Explicit cache directory path
            env_var: Environment variable name to check
            default_subpath: Default subpath under cache root

        Returns:
            Resolved cache directory path
        """
        if cache_dir:
            resolved = Path(cache_dir)
        elif env_var and os.getenv(env_var):
            resolved = Path(os.getenv(env_var))
        else:
            resolved = self.get_cache_root() / default_subpath

        resolved.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Resolved cache directory: {resolved}")
        return resolved

    def get_project_output_dir(self, subdir: str = 'outputs') -> Path:
        """
        Get project output directory (relative to project root).

        Args:
            subdir: Subdirectory name under project root

        Returns:
            Path to output directory
        """
        # This should be relative to the project root, not cache
        # Find project root (where .git or pyproject.toml exists)
        current = Path.cwd()
        while current != current.parent:
            if (current / '.git').exists() or (current / 'pyproject.toml').exists():
                output_dir = current / subdir
                output_dir.mkdir(parents=True, exist_ok=True)
                return output_dir
            current = current.parent

        # Fallback to current directory
        output_dir = Path.cwd() / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


# Global singleton instance
path_manager = PathManager()


# Convenience functions for backward compatibility
def get_hf_hub_cache() -> Path:
    """Get HuggingFace Hub cache directory"""
    return path_manager.get_hf_hub_dir()


def get_hf_home() -> Path:
    """Get HuggingFace home directory"""
    return path_manager.get_hf_home()


def get_bark_cache() -> Path:
    """Get Bark TTS cache directory"""
    return path_manager.get_bark_cache_dir()


def resolve_model_path(model_id: str) -> Optional[Path]:
    """Find a model in HuggingFace Hub cache"""
    return path_manager.find_model_in_hub(model_id)
