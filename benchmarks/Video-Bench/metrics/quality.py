"""
Video Quality Metrics for VideoSeal Watermark Evaluation

Implements PSNR, SSIM, and tLP (temporal LPIPS) metrics
following VideoMarkBench methodology.
"""

import math
import torch
import numpy as np
import lpips

try:
    from pytorch_msssim import ssim as pytorch_ssim
except ImportError:
    pytorch_ssim = None
    print("Warning: pytorch_msssim not installed. SSIM metric will not be available.")


def compute_psnr(video_watermarked, video_original, is_video=True):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR)

    Implementation strictly follows:
    src/video_watermark/videoseal/videoseal/evals/metrics.py

    Args:
        video_watermarked: torch.Tensor, shape (T, C, H, W), range [0, 1]
        video_original: torch.Tensor, shape (T, C, H, W), range [0, 1]
        is_video: bool, if True aggregate over entire sequence, else per-frame

    Returns:
        float: PSNR value in dB
    """
    # Convert to pixel space [0, 255]
    delta = 255 * (video_watermarked - video_original)

    # Reshape to BxCxHxW format
    delta = delta.reshape(-1, video_watermarked.shape[-3],
                         video_watermarked.shape[-2],
                         video_watermarked.shape[-1])

    # Peak value for 8-bit images: 20 * log10(255) ≈ 48.13 dB
    peak = 20 * math.log10(255.0)

    # Aggregate dimensions
    avg_dims = (0, 1, 2, 3) if is_video else (1, 2, 3)
    noise = torch.mean(delta ** 2, dim=avg_dims)

    # PSNR formula: peak - 10 * log10(MSE)
    psnr = peak - 10 * torch.log10(noise)

    return psnr.item()


def compute_ssim(video_watermarked, video_original, data_range=1.0):
    """
    Compute Structural Similarity Index (SSIM)

    Calculates SSIM per frame and returns the average.
    Uses pytorch_msssim for consistency with VideoSeal evaluation.

    Args:
        video_watermarked: torch.Tensor, shape (T, C, H, W), range [0, 1]
        video_original: torch.Tensor, shape (T, C, H, W), range [0, 1]
        data_range: float, data range of input (default: 1.0)

    Returns:
        float: Average SSIM value across all frames, range [0, 1]
    """
    if pytorch_ssim is None:
        raise ImportError("pytorch_msssim is required for SSIM computation. "
                         "Install with: pip install pytorch-msssim")

    ssim_values = []
    T = video_watermarked.shape[0]

    for t in range(T):
        # Extract single frame and add batch dimension
        frame_wm = video_watermarked[t].unsqueeze(0)    # (1, C, H, W)
        frame_orig = video_original[t].unsqueeze(0)     # (1, C, H, W)

        # Compute SSIM for this frame
        ssim_val = pytorch_ssim(frame_wm, frame_orig,
                               data_range=data_range,
                               size_average=False)
        ssim_values.append(ssim_val.item())

    # Return average across all frames
    return np.mean(ssim_values)


def init_lpips_model(device='cuda'):
    """
    Initialize LPIPS model once for reuse

    Optimization: Avoids repeated model initialization in loops

    Args:
        device: str, computation device ('cuda' or 'cpu')

    Returns:
        lpips.LPIPS: Initialized LPIPS model
    """
    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()
    return loss_fn


def compute_tLP(video_watermarked, video_original, device='cuda', lpips_model=None):
    """
    Compute temporal LPIPS (tLP) - temporal consistency metric

    Measures perceptual consistency between consecutive frames.
    Lower tLP indicates better temporal coherence (watermark preserves video smoothness).

    Implementation inferred from VideoMarkBench methodology:
    1. Compute LPIPS between adjacent frames for original video
    2. Compute LPIPS between adjacent frames for watermarked video
    3. Return average absolute difference

    Args:
        video_watermarked: torch.Tensor, shape (T, C, H, W), range [0, 1]
        video_original: torch.Tensor, shape (T, C, H, W), range [0, 1]
        device: str, computation device ('cuda' or 'cpu')
        lpips_model: lpips.LPIPS, optional pre-initialized model (avoids repeated init)

    Returns:
        float: tLP value (lower is better, indicates temporal coherence)
    """
    # Use provided model or initialize new one
    if lpips_model is None:
        loss_fn = init_lpips_model(device)
    else:
        loss_fn = lpips_model

    T = video_original.shape[0]

    if T < 2:
        # Need at least 2 frames to compute temporal metric
        return 0.0

    # Compute LPIPS for consecutive frames in original video
    lpips_original = []
    for t in range(T - 1):
        frame_t = video_original[t].unsqueeze(0).to(device)
        frame_t1 = video_original[t+1].unsqueeze(0).to(device)

        # LPIPS expects range [-1, 1]
        frame_t = frame_t * 2 - 1
        frame_t1 = frame_t1 * 2 - 1

        with torch.no_grad():
            lpips_val = loss_fn(frame_t, frame_t1).item()
        lpips_original.append(lpips_val)

    # Compute LPIPS for consecutive frames in watermarked video
    lpips_watermarked = []
    for t in range(T - 1):
        frame_t = video_watermarked[t].unsqueeze(0).to(device)
        frame_t1 = video_watermarked[t+1].unsqueeze(0).to(device)

        # LPIPS expects range [-1, 1]
        frame_t = frame_t * 2 - 1
        frame_t1 = frame_t1 * 2 - 1

        with torch.no_grad():
            lpips_val = loss_fn(frame_t, frame_t1).item()
        lpips_watermarked.append(lpips_val)

    # Compute tLP as average absolute difference
    # between original and watermarked temporal LPIPS sequences
    tLP = sum([abs(w - o) for w, o in zip(lpips_watermarked, lpips_original)]) / (T - 1)

    return tLP


def compute_all_quality_metrics(video_watermarked, video_original, device='cuda', lpips_model=None):
    """
    Compute all quality metrics for a video pair

    Args:
        video_watermarked: torch.Tensor, shape (T, C, H, W), range [0, 1]
        video_original: torch.Tensor, shape (T, C, H, W), range [0, 1]
        device: str, computation device
        lpips_model: lpips.LPIPS, optional pre-initialized model (optimization)

    Returns:
        dict: Dictionary containing all quality metrics
            {
                'psnr': float,    # Higher is better
                'ssim': float,    # Higher is better (0-1)
                'tLP': float      # Lower is better
            }
    """
    metrics = {}

    # Compute PSNR
    metrics['psnr'] = compute_psnr(video_watermarked, video_original, is_video=True)

    # Compute SSIM
    if pytorch_ssim is not None:
        metrics['ssim'] = compute_ssim(video_watermarked, video_original)
    else:
        metrics['ssim'] = None

    # Compute tLP (with optional pre-initialized model)
    metrics['tLP'] = compute_tLP(video_watermarked, video_original, device=device, lpips_model=lpips_model)

    return metrics


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing quality metrics with synthetic video...")

    # Create synthetic video (10 frames, 3 channels, 64x64)
    T, C, H, W = 10, 3, 64, 64
    video_orig = torch.rand(T, C, H, W)
    video_wm = video_orig + 0.01 * torch.randn(T, C, H, W)
    video_wm = torch.clamp(video_wm, 0, 1)

    # Compute metrics
    print("\nComputing PSNR...")
    psnr_val = compute_psnr(video_wm, video_orig, is_video=True)
    print(f"PSNR: {psnr_val:.2f} dB")

    if pytorch_ssim is not None:
        print("\nComputing SSIM...")
        ssim_val = compute_ssim(video_wm, video_orig)
        print(f"SSIM: {ssim_val:.4f}")

    print("\nComputing tLP...")
    tlp_val = compute_tLP(video_wm, video_orig, device='cpu')
    print(f"tLP: {tlp_val:.6f}")

    print("\nAll quality metrics computed successfully!")
