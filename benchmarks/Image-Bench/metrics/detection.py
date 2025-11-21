"""
Watermark Detection Metrics

Computes TPR (True Positive Rate), Bit Accuracy, and confidence statistics
from watermark extraction results.
"""

import numpy as np
from typing import List, Dict, Any, Optional


def compute_bit_accuracy(
    all_raw_preds: np.ndarray,
    message_bits: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute bit accuracy (percentage of correctly extracted bits).

    Strategy: Compare all images' raw_preds against message_bits globally.
    Suitable for scenarios where all images share the same embedded message.

    Implementation follows VideoSeal's bit_accuracy logic:
    1. Binarize predicted bits (> threshold)
    2. Binarize target bits (> 0.5)
    3. Compare bit-wise and compute accuracy

    Args:
        all_raw_preds: Concatenated raw_preds from all images, shape (N*255,) or (N*256,)
        message_bits: Original message bits, shape (256,)
        threshold: Binarization threshold (default 0.5)

    Returns:
        float: Bit accuracy [0.0, 1.0]

    Examples:
        >>> # 1000 images, 255 bits each
        >>> raw_preds = np.random.rand(1000 * 255)
        >>> msg_bits = np.array([0, 1, 0, 1, ...])  # 256 bits
        >>> bit_acc = compute_bit_accuracy(raw_preds, msg_bits)
        >>> print(f"Bit Accuracy: {bit_acc:.2%}")
    """
    if len(all_raw_preds) == 0 or len(message_bits) == 0:
        return 0.0

    # Calculate bits per image
    n_total = len(all_raw_preds)
    n_msg_bits = len(message_bits)

    # Determine if raw_preds are 255-bit or 256-bit
    # raw_preds are usually 255 (excluding first detection bit)
    # message_bits are full 256-bit
    if n_total % 255 == 0:
        # raw_preds are 255-bit, need to align to message_bits[1:]
        msg_bits_aligned = message_bits[1:]  # Skip first bit
        n_images = n_total // 255
    elif n_total % 256 == 0:
        # raw_preds are 256-bit (full)
        msg_bits_aligned = message_bits
        n_images = n_total // 256
    else:
        # Length mismatch, try to find common divisor
        bits_per_image = n_total // max(1, n_total // n_msg_bits)
        msg_bits_aligned = message_bits[:bits_per_image]
        n_images = n_total // bits_per_image

    # Repeat message_bits to match all images
    # E.g., 1000 images × 255 bits = 255000 total bits
    msg_bits_repeated = np.tile(msg_bits_aligned, n_images)[:n_total]

    # Binarize
    preds_binary = (all_raw_preds > threshold).astype(int)
    targets_binary = (msg_bits_repeated > 0.5).astype(int)

    # Compute accuracy
    correct = (preds_binary == targets_binary).astype(float)
    bit_acc = np.mean(correct)

    return float(bit_acc)


def compute_detection_metrics(
    results: List[Dict[str, Any]],
    message_bits: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute detection statistics: TPR, Bit Accuracy, and confidence.

    Args:
        results: List of extraction result dicts, each containing:
            - 'detected': bool, whether watermark was detected
            - 'confidence': float, confidence score
            - 'raw_preds': np.ndarray (optional), raw bit predictions for bit accuracy
        message_bits: Original message bits (256,), used to compute bit accuracy

    Returns:
        dict: {
            'tpr': float,               # True Positive Rate (0.0-1.0)
            'avg_confidence': float,    # Average confidence for detected watermarks
            'bit_accuracy': float,      # Bit accuracy (0.0-1.0)
            'total_images': int,        # Total number of images evaluated
            'detected_count': int,      # Number of images with watermark detected
            'failed_count': int         # Number of images where watermark was not detected
        }

    Example:
        >>> results = [
        ...     {'detected': True, 'confidence': 20.96, 'raw_preds': np.array([...])},
        ...     {'detected': True, 'confidence': 19.27, 'raw_preds': np.array([...])},
        ...     {'detected': False, 'confidence': 0.12},
        ... ]
        >>> msg_bits = np.array([0, 1, 0, 1, ...])  # 256 bits
        >>> metrics = compute_detection_metrics(results, message_bits=msg_bits)
        >>> print(f"TPR: {metrics['tpr']:.2%}, Bit Acc: {metrics['bit_accuracy']:.2%}")
        TPR: 66.67%, Bit Acc: 89.34%
    """
    if not results:
        return {
            'tpr': 0.0,
            'detection_rate': 0.0,
            'avg_confidence': 0.0,
            'bit_accuracy': 0.0,
            'total_images': 0,
            'detected_count': 0,
            'failed_count': 0
        }

    total = len(results)
    detected = [r for r in results if r.get('detected', False)]
    detected_count = len(detected)
    failed_count = total - detected_count

    # 1. Compute average confidence (only for detected watermarks)
    avg_conf = 0.0
    if detected:
        avg_conf = sum(r.get('confidence', 0.0) for r in detected) / detected_count

    # 2. Compute TPR (True Positive Rate)
    # In Image-Bench all samples are positive (watermarked), so TPR = detection success rate
    tpr = detected_count / total if total > 0 else 0.0

    # 3. Compute Bit Accuracy (new!)
    bit_acc = 0.0
    if message_bits is not None:
        # Collect all raw_preds from results
        all_preds = []
        for r in results:
            if 'raw_preds' in r and r['raw_preds'] is not None:
                preds = r['raw_preds']
                if isinstance(preds, np.ndarray):
                    all_preds.append(preds.flatten())

        # Concatenate and compute overall bit accuracy
        if all_preds:
            all_preds_concat = np.concatenate(all_preds)
            bit_acc = compute_bit_accuracy(all_preds_concat, message_bits)

    return {
        'tpr': tpr,                     # True positive rate (a.k.a detection rate)
        'detection_rate': tpr,          # Backward compatibility for legacy code
        'avg_confidence': avg_conf,
        'bit_accuracy': bit_acc,        # New metric!
        'total_images': total,
        'detected_count': detected_count,
        'failed_count': failed_count
    }


def compute_detection_metrics_by_strength(
    results_by_strength: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute detection metrics for multiple attack strengths.

    Args:
        results_by_strength: Dict mapping strength values to extraction results
            Example: {
                '1.2': [{'detected': True, 'confidence': 0.95}, ...],
                '1.4': [{'detected': True, 'confidence': 0.87}, ...],
            }

    Returns:
        dict: Metrics for each strength level

    Example:
        >>> results = {
        ...     '1.2': [{'detected': True, 'confidence': 0.95}] * 100,
        ...     '1.4': [{'detected': True, 'confidence': 0.85}] * 100,
        ... }
        >>> metrics = compute_detection_metrics_by_strength(results)
        >>> print(metrics['1.2']['detection_rate'])
        1.0
    """
    metrics_by_strength = {}

    for strength, results in results_by_strength.items():
        metrics_by_strength[str(strength)] = compute_detection_metrics(results)

    return metrics_by_strength


def print_detection_summary(metrics: Dict[str, float], indent: int = 0) -> None:
    """
    Pretty print detection metrics summary.

    Args:
        metrics: Detection metrics dict from compute_detection_metrics()
        indent: Number of spaces for indentation

    Example:
        >>> metrics = compute_detection_metrics(results, message_bits=msg_bits)
        >>> print_detection_summary(metrics)
        TPR: 95.00%
        Bit Accuracy: 89.34%
        Avg Confidence: 20.87
        Total Images: 1000
        Detected: 950
        Failed: 50
    """
    prefix = ' ' * indent
    print(f"{prefix}TPR: {metrics['tpr']:.2%}")
    print(f"{prefix}Bit Accuracy: {metrics['bit_accuracy']:.2%}")
    print(f"{prefix}Avg Confidence: {metrics['avg_confidence']:.4f}")
    print(f"{prefix}Total Images: {metrics['total_images']}")
    print(f"{prefix}Detected: {metrics['detected_count']}")
    print(f"{prefix}Failed: {metrics['failed_count']}")
