"""
Video Attack Executor for VideoSeal Watermark Robustness Testing

Wrapper around VideoMarkBench Common Perturbations (8 attack types).
Provides unified interface for applying frame-level and video-level attacks.
"""

import sys
import torch
from pathlib import Path
from typing import Union, List

# Add VideoMarkBench common_perturbations to path
VIDEOMARKBENCH_PATH = Path(__file__).parent.parent.parent / 'VideoMarkBench' / 'common_perturabtions'
if str(VIDEOMARKBENCH_PATH) not in sys.path:
    sys.path.insert(0, str(VIDEOMARKBENCH_PATH))

# Import VideoMarkBench attack modules
try:
    from jpeg import JPEG
    from gaussian import Gaussian
    from gaussian_blur import GaussianBlur
    from crop import Crop
    from frame_average import FrameAverage
    from frame_switch import FrameSwitch
    from frame_remove import FrameRemove
    from mpeg4 import MPEG4
except ImportError as e:
    raise ImportError(
        f"Failed to import VideoMarkBench attacks. "
        f"Please ensure benchmarks/VideoMarkBench/common_perturabtions/ exists. "
        f"Error: {e}"
    )


class VideoAttackExecutor:
    """
    Unified video attack executor

    Wraps VideoMarkBench Common Perturbations for robustness evaluation.
    Supports 8 attack types from VideoMarkBench paper:
        - Frame-level: gaussian_noise, gaussian_blur, jpeg, crop
        - Video-level: mpeg4, frame_average, frame_swap, frame_remove
    """

    # Attack type categorization
    FRAME_LEVEL_ATTACKS = ['gaussian_noise', 'gaussian_blur', 'jpeg', 'crop']
    VIDEO_LEVEL_ATTACKS = ['mpeg4', 'frame_average', 'frame_swap', 'frame_remove']

    def __init__(self):
        """Initialize attack executor with attack class mapping"""
        self.attack_class_map = {
            'gaussian_noise': Gaussian,
            'gaussian_blur': GaussianBlur,
            'jpeg': JPEG,
            'crop': Crop,
            'mpeg4': MPEG4,
            'frame_average': FrameAverage,
            'frame_swap': FrameSwitch,
            'frame_remove': FrameRemove,
        }

    def apply_attack(
        self,
        video_tensor: torch.Tensor,
        attack_type: str,
        strength: Union[float, int, List]
    ) -> torch.Tensor:
        """
        Apply specified attack to video tensor

        Args:
            video_tensor: torch.Tensor, shape (T, C, H, W), range [0, 1]
            attack_type: str, one of 8 supported attack types
            strength: attack strength parameter
                - gaussian_noise: std (float, e.g., 0.01-0.20)
                - gaussian_blur: kernel std (float, e.g., 0.1-1.5)
                - jpeg: quality factor (int, e.g., 20-90)
                - crop: retain ratio (float, e.g., 0.90-0.98)
                - mpeg4: CRF value (int, e.g., 1-40)
                - frame_average: window size (int, e.g., 1-5)
                - frame_swap: probability (float, e.g., 0.0-0.20)
                - frame_remove: probability (float, e.g., 0.0-0.20)

        Returns:
            torch.Tensor: Attacked video, shape (T', C, H, W)
                         Note: T' may differ from T for frame_remove attack
        """
        if attack_type not in self.attack_class_map:
            raise ValueError(
                f"Unknown attack type: {attack_type}. "
                f"Supported: {list(self.attack_class_map.keys())}"
            )

        # Get attack class
        attack_cls = self.attack_class_map[attack_type]

        # Instantiate attack module with strength parameter
        attack_module = attack_cls(strength)

        # Apply attack based on type
        if attack_type in self.FRAME_LEVEL_ATTACKS:
            return self._apply_frame_level_attack(video_tensor, attack_module)
        else:
            return self._apply_video_level_attack(video_tensor, attack_module)

    def _apply_frame_level_attack(
        self,
        video_tensor: torch.Tensor,
        attack_module: torch.nn.Module
    ) -> torch.Tensor:
        """
        Apply attack to each frame independently

        Args:
            video_tensor: shape (T, C, H, W)
            attack_module: VideoMarkBench attack module

        Returns:
            torch.Tensor: shape (T, C, H, W), attacked video
        """
        T = video_tensor.shape[0]
        attacked_frames = []

        for t in range(T):
            # Extract single frame and add batch dimension
            frame = video_tensor[t].unsqueeze(0)  # (1, C, H, W)

            # Apply attack
            attacked_frame = attack_module(frame)  # (1, C, H, W)

            # Remove batch dimension
            attacked_frames.append(attacked_frame.squeeze(0))

        # Stack frames back into video
        attacked_video = torch.stack(attacked_frames, dim=0)  # (T, C, H, W)

        return attacked_video

    def _apply_video_level_attack(
        self,
        video_tensor: torch.Tensor,
        attack_module: torch.nn.Module
    ) -> torch.Tensor:
        """
        Apply attack to entire video sequence

        Args:
            video_tensor: shape (T, C, H, W)
            attack_module: VideoMarkBench attack module

        Returns:
            torch.Tensor: shape (T', C, H, W), attacked video
                         Note: T' may differ for frame_remove
        """
        # VideoMarkBench video-level attacks expect (T, C, H, W)
        attacked_video = attack_module(video_tensor)
        return attacked_video

    def get_supported_attacks(self) -> List[str]:
        """Get list of supported attack types"""
        return list(self.attack_class_map.keys())

    def is_frame_level(self, attack_type: str) -> bool:
        """Check if attack is frame-level"""
        return attack_type in self.FRAME_LEVEL_ATTACKS

    def is_video_level(self, attack_type: str) -> bool:
        """Check if attack is video-level"""
        return attack_type in self.VIDEO_LEVEL_ATTACKS


def apply_multiple_attacks(
    video_tensor: torch.Tensor,
    attack_configs: List[dict],
    executor: VideoAttackExecutor = None
) -> dict:
    """
    Apply multiple attacks to a single video

    Args:
        video_tensor: torch.Tensor, shape (T, C, H, W)
        attack_configs: list of dicts, each containing:
            {'type': str, 'strength': float/int, 'name': str}
        executor: VideoAttackExecutor instance (optional)

    Returns:
        dict: {attack_name: attacked_video_tensor}
    """
    if executor is None:
        executor = VideoAttackExecutor()

    results = {}

    for config in attack_configs:
        attack_type = config['type']
        strength = config['strength']
        attack_name = config.get('name', f"{attack_type}_{strength}")

        # Apply attack
        attacked_video = executor.apply_attack(video_tensor, attack_type, strength)

        results[attack_name] = attacked_video

    return results


if __name__ == "__main__":
    # Test attack executor
    print("Testing VideoAttackExecutor...")

    # Create synthetic video (10 frames, 3 channels, 64x64)
    print("\nCreating synthetic video (10 frames, 3x64x64)...")
    T, C, H, W = 10, 3, 64, 64
    video = torch.rand(T, C, H, W)
    print(f"Original video shape: {video.shape}")

    # Initialize executor
    executor = VideoAttackExecutor()
    print(f"\nSupported attacks: {executor.get_supported_attacks()}")

    # Test frame-level attack (JPEG)
    print("\n[1/8] Testing JPEG compression (quality=50)...")
    attacked = executor.apply_attack(video.clone(), 'jpeg', strength=50)
    print(f"  Result shape: {attacked.shape}")
    assert attacked.shape == video.shape, "Shape mismatch!"

    # Test frame-level attack (Gaussian noise)
    print("[2/8] Testing Gaussian noise (std=0.05)...")
    attacked = executor.apply_attack(video.clone(), 'gaussian_noise', strength=0.05)
    print(f"  Result shape: {attacked.shape}")

    # Test frame-level attack (Gaussian blur)
    print("[3/8] Testing Gaussian blur (kernel_std=0.5)...")
    attacked = executor.apply_attack(video.clone(), 'gaussian_blur', strength=0.5)
    print(f"  Result shape: {attacked.shape}")

    # Test frame-level attack (Crop)
    print("[4/8] Testing Crop (retain_ratio=0.95)...")
    attacked = executor.apply_attack(video.clone(), 'crop', strength=0.95)
    print(f"  Result shape: {attacked.shape}")

    # Test video-level attack (MPEG-4)
    print("[5/8] Testing MPEG-4 compression (CRF=20)...")
    attacked = executor.apply_attack(video.clone(), 'mpeg4', strength=20)
    print(f"  Result shape: {attacked.shape}")

    # Test video-level attack (Frame average)
    print("[6/8] Testing Frame average (window=2)...")
    attacked = executor.apply_attack(video.clone(), 'frame_average', strength=2)
    print(f"  Result shape: {attacked.shape}")

    # Test video-level attack (Frame swap)
    print("[7/8] Testing Frame swap (prob=0.1)...")
    attacked = executor.apply_attack(video.clone(), 'frame_swap', strength=0.1)
    print(f"  Result shape: {attacked.shape}")

    # Test video-level attack (Frame remove)
    print("[8/8] Testing Frame remove (prob=0.1)...")
    attacked = executor.apply_attack(video.clone(), 'frame_remove', strength=0.1)
    print(f"  Result shape: {attacked.shape}")
    print(f"  Note: Frames may be removed (T' <= T)")

    print("\n✅ All attacks tested successfully!")
