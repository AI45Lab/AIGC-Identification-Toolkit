"""
Video I/O Utilities for Video-Bench

Provides functions for loading and saving videos in both .pt tensor and .mp4 formats.
Prioritizes .pt format for efficiency and losslessness during benchmarking.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional

try:
    import torchvision.io as tvio
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: torchvision not available. Video loading from MP4 will be limited.")


def load_video_tensor(
    video_path: Union[str, Path],
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Load video as PyTorch tensor

    Supports two formats:
        - .pt: PyTorch tensor file (preferred, lossless)
        - .mp4/.avi/.mov: Video files (requires torchvision)

    Args:
        video_path: path to video file
        device: device to load tensor to ('cpu' or 'cuda')

    Returns:
        torch.Tensor: Video tensor, shape (T, C, H, W), range [0, 1]
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Load from .pt tensor file
    if video_path.suffix == '.pt':
        video = torch.load(video_path, map_location=device)

        # Ensure shape is (T, C, H, W)
        if video.dim() == 4:
            pass  # Already correct shape
        elif video.dim() == 5 and video.shape[0] == 1:
            video = video.squeeze(0)  # Remove batch dimension
        else:
            raise ValueError(f"Unexpected video tensor shape: {video.shape}")

        # Ensure range [0, 1]
        if video.max() > 1.0:
            video = video / 255.0

        return video

    # Load from video file (.mp4, .avi, .mov, etc.)
    elif video_path.suffix in ['.mp4', '.avi', '.mov', '.mkv']:
        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required to load video files. "
                "Install with: pip install torchvision"
            )

        # Load video using torchvision (always to CPU first to avoid GPU OOM)
        video, audio, info = tvio.read_video(str(video_path), pts_unit='sec')

        # Ensure video is on CPU (torchvision may decode to GPU)
        if video.is_cuda:
            video = video.cpu()

        # Convert from (T, H, W, C) to (T, C, H, W)
        video = video.permute(0, 3, 1, 2)

        # Normalize to [0, 1]
        video = video.float() / 255.0

        # Only move to target device if explicitly requested (not for batch loading)
        return video.to(device)

    else:
        raise ValueError(
            f"Unsupported video format: {video_path.suffix}. "
            f"Supported: .pt, .mp4, .avi, .mov, .mkv"
        )


def save_video_tensor(
    video_tensor: torch.Tensor,
    output_path: Union[str, Path],
    fps: int = 30,
    codec: str = 'libx264'
):
    """
    Save video tensor to file

    Supports two formats:
        - .pt: PyTorch tensor file (preferred for benchmarking)
        - .mp4: Video file (for visualization)

    Args:
        video_tensor: torch.Tensor, shape (T, C, H, W), range [0, 1]
        output_path: path to save video
        fps: frames per second (for .mp4 only)
        codec: video codec (for .mp4 only, e.g., 'libx264', 'h264')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as .pt tensor file
    if output_path.suffix == '.pt':
        torch.save(video_tensor.cpu(), output_path)

    # Save as .mp4 video file
    elif output_path.suffix == '.mp4':
        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required to save video files. "
                "Install with: pip install torchvision"
            )

        # Convert to uint8 [0, 255]
        video_uint8 = (video_tensor * 255).clamp(0, 255).byte()

        # Convert from (T, C, H, W) to (T, H, W, C)
        video_uint8 = video_uint8.permute(0, 2, 3, 1)

        # Save video
        tvio.write_video(
            str(output_path),
            video_uint8.cpu(),
            fps=fps,
            video_codec=codec,
            options={'crf': '18'}  # High quality
        )

    else:
        raise ValueError(
            f"Unsupported output format: {output_path.suffix}. "
            f"Supported: .pt, .mp4"
        )


def load_dataset_videos(
    dataset_path: Union[str, Path],
    max_videos: Optional[int] = None,
    device: str = 'cpu'
) -> Tuple[list, list]:
    """
    Load all videos from a dataset directory (recursively scans subdirectories)

    Supports nested directory structures like:
        dataset/
        ├── model1/
        │   ├── category1/*.mp4
        │   └── category2/*.mp4
        └── model2/
            └── category/*.mp4

    Args:
        dataset_path: path to dataset directory containing .pt or video files
        max_videos: maximum number of videos to load (None for all)
        device: device to load tensors to

    Returns:
        tuple: (video_paths, video_tensors)
            - video_paths: list of Path objects
            - video_tensors: list of torch.Tensor, each shape (T, C, H, W)
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Find all video files recursively
    video_extensions = ['.pt', '.mp4', '.avi', '.mov', '.mkv']
    video_files = []

    for ext in video_extensions:
        # Use rglob for recursive search (** matches any number of subdirectories)
        video_files.extend(dataset_path.rglob(f'**/*{ext}'))

    # Sort for reproducibility
    video_files = sorted(video_files)

    # Limit number of videos
    if max_videos is not None:
        video_files = video_files[:max_videos]

    print(f"Found {len(video_files)} videos in {dataset_path}")

    # Load videos
    video_paths = []
    video_tensors = []

    for video_file in video_files:
        try:
            video = load_video_tensor(video_file, device=device)
            video_paths.append(video_file)
            video_tensors.append(video)
        except Exception as e:
            print(f"Warning: Failed to load {video_file}: {e}")
            continue

    print(f"Successfully loaded {len(video_tensors)} videos")

    return video_paths, video_tensors


def tensor_to_numpy(video_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert video tensor to numpy array

    Args:
        video_tensor: torch.Tensor, shape (T, C, H, W), range [0, 1]

    Returns:
        np.ndarray: shape (T, H, W, C), range [0, 255], dtype uint8
    """
    # Convert to numpy
    video_np = video_tensor.cpu().numpy()

    # Convert from (T, C, H, W) to (T, H, W, C)
    video_np = np.transpose(video_np, (0, 2, 3, 1))

    # Convert to uint8 [0, 255]
    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)

    return video_np


def numpy_to_tensor(video_np: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    Convert numpy array to video tensor

    Args:
        video_np: np.ndarray, shape (T, H, W, C), range [0, 255], dtype uint8
        device: device to place tensor on

    Returns:
        torch.Tensor: shape (T, C, H, W), range [0, 1]
    """
    # Convert to float and normalize
    video_np = video_np.astype(np.float32) / 255.0

    # Convert from (T, H, W, C) to (T, C, H, W)
    video_np = np.transpose(video_np, (0, 3, 1, 2))

    # Convert to tensor
    video_tensor = torch.from_numpy(video_np).to(device)

    return video_tensor


def get_video_info(video_tensor: torch.Tensor) -> dict:
    """
    Get information about a video tensor

    Args:
        video_tensor: torch.Tensor, shape (T, C, H, W)

    Returns:
        dict: Video information
    """
    T, C, H, W = video_tensor.shape

    info = {
        'num_frames': T,
        'num_channels': C,
        'height': H,
        'width': W,
        'shape': (T, C, H, W),
        'dtype': str(video_tensor.dtype),
        'device': str(video_tensor.device),
        'min_value': video_tensor.min().item(),
        'max_value': video_tensor.max().item(),
        'mean_value': video_tensor.mean().item(),
    }

    return info


if __name__ == "__main__":
    # Test video I/O utilities
    print("Testing Video I/O utilities...")

    # Create synthetic video
    print("\n[1/3] Creating synthetic video...")
    T, C, H, W = 10, 3, 64, 64
    video = torch.rand(T, C, H, W)
    print(f"Created video: {video.shape}")

    # Test saving and loading .pt format
    print("\n[2/3] Testing .pt format I/O...")
    temp_pt_path = Path('/tmp/test_video.pt')

    save_video_tensor(video, temp_pt_path)
    print(f"  Saved to {temp_pt_path}")

    loaded_video = load_video_tensor(temp_pt_path)
    print(f"  Loaded: {loaded_video.shape}")

    assert torch.allclose(video, loaded_video), "Video mismatch after save/load!"
    print("  ✓ .pt format I/O successful")

    # Clean up
    temp_pt_path.unlink()

    # Test video info
    print("\n[3/3] Testing video info...")
    info = get_video_info(video)
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n✅ All video I/O tests passed!")
