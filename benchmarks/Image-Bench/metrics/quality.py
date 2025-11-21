"""
Image Quality Metrics: PSNR, SSIM, LPIPS

Adapted from VINE's quality_metrics.py for evaluating watermark imperceptibility.
"""

import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from typing import Union, Dict

try:
    import lpips
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("⚠️  Warning: Quality metrics dependencies not installed.")
    print("   Install with: pip install scikit-image lpips")


def compute_quality_metrics(
    original_img: Union[Image.Image, np.ndarray, str],
    watermarked_img: Union[Image.Image, np.ndarray, str],
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute PSNR, SSIM, LPIPS between original and watermarked images.

    Args:
        original_img: Original image (PIL.Image, numpy array, or file path)
        watermarked_img: Watermarked image (PIL.Image, numpy array, or file path)
        device: Device for LPIPS computation ('cuda' or 'cpu')

    Returns:
        dict: {
            'psnr': float,      # Peak Signal-to-Noise Ratio (higher is better)
            'ssim': float,      # Structural Similarity Index (0-1, higher is better)
            'lpips': float      # Learned Perceptual Image Patch Similarity (lower is better)
        }

    Example:
        >>> from PIL import Image
        >>> orig = Image.open('original.png')
        >>> wm = Image.open('watermarked.png')
        >>> metrics = compute_quality_metrics(orig, wm, device='cuda')
        >>> print(f"PSNR: {metrics['psnr']:.2f} dB")
    """
    if not METRICS_AVAILABLE:
        raise ImportError(
            "Quality metrics dependencies not installed. "
            "Install with: pip install scikit-image lpips"
        )

    # Convert inputs to numpy arrays
    orig_arr = _to_numpy(original_img)
    wm_arr = _to_numpy(watermarked_img)

    # Ensure RGB format
    if orig_arr.shape[-1] != 3:
        orig_arr = cv2.cvtColor(orig_arr, cv2.COLOR_BGR2RGB)
    if wm_arr.shape[-1] != 3:
        wm_arr = cv2.cvtColor(wm_arr, cv2.COLOR_BGR2RGB)

    # Compute PSNR & SSIM
    psnr_value = psnr(orig_arr, wm_arr)
    ssim_value, _ = ssim(orig_arr, wm_arr, full=True, channel_axis=2)

    # Compute LPIPS
    lpips_value = _compute_lpips(orig_arr, wm_arr, device)

    return {
        'psnr': float(psnr_value),
        'ssim': float(ssim_value),
        'lpips': float(lpips_value)
    }


def _to_numpy(img: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
    """Convert various image formats to numpy array."""
    if isinstance(img, str):
        # Load from file path
        img = Image.open(img).convert('RGB')
        return np.array(img)
    elif isinstance(img, Image.Image):
        # PIL Image
        return np.array(img.convert('RGB'))
    elif isinstance(img, np.ndarray):
        # Already numpy
        return img
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


def _compute_lpips(
    orig_arr: np.ndarray,
    wm_arr: np.ndarray,
    device: str = 'cuda'
) -> float:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        orig_arr: Original image as numpy array (H, W, C)
        wm_arr: Watermarked image as numpy array (H, W, C)
        device: Device for computation

    Returns:
        float: LPIPS score (lower is better, 0 = identical)
    """
    # Initialize LPIPS model (AlexNet backend)
    loss_fn = lpips.LPIPS(net='alex').to(device)

    # Convert to tensors and normalize to [-1, 1]
    orig_tensor = torchvision.transforms.ToTensor()(orig_arr) * 2 - 1
    wm_tensor = torchvision.transforms.ToTensor()(wm_arr) * 2 - 1

    # Move to device and compute
    orig_tensor = orig_tensor.to(device)
    wm_tensor = wm_tensor.to(device)

    with torch.no_grad():
        lpips_score = loss_fn(orig_tensor, wm_tensor)

    return float(lpips_score.item())


def compute_batch_quality_metrics(
    original_images: list,
    watermarked_images: list,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute average quality metrics over a batch of images.

    Args:
        original_images: List of original images
        watermarked_images: List of watermarked images
        device: Device for LPIPS computation

    Returns:
        dict: Average metrics {
            'psnr': float,
            'ssim': float,
            'lpips': float
        }

    Example:
        >>> orig_list = [Image.open(f'orig_{i}.png') for i in range(10)]
        >>> wm_list = [Image.open(f'wm_{i}.png') for i in range(10)]
        >>> avg_metrics = compute_batch_quality_metrics(orig_list, wm_list)
        >>> print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
    """
    if len(original_images) != len(watermarked_images):
        raise ValueError("Number of original and watermarked images must match")

    metrics_list = []
    for orig, wm in zip(original_images, watermarked_images):
        metrics = compute_quality_metrics(orig, wm, device)
        metrics_list.append(metrics)

    # Compute averages
    avg_metrics = {
        'psnr': sum(m['psnr'] for m in metrics_list) / len(metrics_list),
        'ssim': sum(m['ssim'] for m in metrics_list) / len(metrics_list),
        'lpips': sum(m['lpips'] for m in metrics_list) / len(metrics_list)
    }

    return avg_metrics
