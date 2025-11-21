"""
W-Bench Evaluation Metrics

Provides quality metrics (PSNR, SSIM, LPIPS) and detection metrics
for evaluating watermark robustness.
"""

from .quality import compute_quality_metrics
from .detection import compute_detection_metrics

__all__ = [
    'compute_quality_metrics',
    'compute_detection_metrics'
]
