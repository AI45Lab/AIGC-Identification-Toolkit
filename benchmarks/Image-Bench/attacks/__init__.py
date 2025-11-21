"""
W-Bench Traditional Distortion Attacks

Directly imports VINE's distortion utilities without modification.
This module provides a thin wrapper around VINE's battle-tested
distortion implementations for evaluating watermark robustness.
"""

# Import VINE's distortion functions directly
from benchmarks.VINE.vine.w_bench_utils.distortion.distortions import (
    apply_single_distortion,
    distortion_strength_paras
)

# Export the 5 traditional distortion types
DISTORTION_TYPES = [
    'brightness',   # Factor multiplier (1.0-2.0)
    'contrast',     # Factor multiplier (1.0-2.0)
    'blurring',     # Gaussian kernel size (0-20)
    'noise',        # Gaussian noise std (0.0-0.1)
    'compression'   # JPEG quality (90-10, lower=more compression)
]

# Export strength ranges (directly from VINE)
DISTORTION_RANGES = distortion_strength_paras

# Convenience mapping for descriptions
DISTORTION_DESCRIPTIONS = {
    'brightness': 'Adjust image brightness by multiplying pixel values',
    'contrast': 'Adjust image contrast by scaling pixel value differences',
    'blurring': 'Apply Gaussian blur with specified kernel size',
    'noise': 'Add Gaussian noise with specified standard deviation',
    'compression': 'Apply JPEG compression with specified quality level'
}

__all__ = [
    'apply_single_distortion',
    'DISTORTION_TYPES',
    'DISTORTION_RANGES',
    'DISTORTION_DESCRIPTIONS'
]
