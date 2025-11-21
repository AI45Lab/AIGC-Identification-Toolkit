"""
Audio Detection Metrics

Extended from Image-Bench detection metrics with AudioSeal-specific dual-threshold detection.
Supports both tau_prob (detection probability) and tau_ba (bit accuracy) thresholds.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional


def compute_audio_bit_accuracy(
    decoded_bits: torch.Tensor,
    original_msg: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute bit accuracy for audio watermark

    Args:
        decoded_bits: Decoded bits (nbits,), values in [0, 1]
        original_msg: Original message bits (nbits,), values 0 or 1
        threshold: Binarization threshold

    Returns:
        float: Bit accuracy [0.0, 1.0]
    """
    # Binarize
    decoded_binary = (decoded_bits > threshold).int()
    original_binary = (original_msg > 0.5).int()

    # Compute matching bits
    correct = (decoded_binary == original_binary).sum().item()
    total = len(original_msg)

    return correct / total if total > 0 else 0.0


def compute_audio_detection_metrics(
    extraction_results: List[Dict],
    original_msg_bits: Optional[np.ndarray] = None,
    tau_prob: float = 0.15,
    tau_ba: float = 0.1  # User-requested adjustment from 0.875 to 0.1
) -> Dict[str, float]:
    """
    Compute AudioSeal-specific detection metrics

    Extended from Image-Bench's compute_detection_metrics with dual-threshold detection.

    Args:
        extraction_results: List of extraction result dictionaries
        original_msg_bits: Original message bits (16-bit for AudioSeal)
        tau_prob: Detection probability threshold (default 0.15)
        tau_ba: Bit accuracy threshold (default 0.1, user-adjusted)

    Returns:
        Dict containing detection metrics:
        - tpr_prob: TPR based on detection probability
        - tpr_ba: TPR based on bit accuracy
        - avg_confidence: Average confidence score
        - avg_bit_accuracy: Average bit accuracy
        - fnr_prob: False negative rate (prob-based)
        - fnr_ba: False negative rate (BA-based)
        - total_samples: Total number of samples
        - detected_count_prob: Count detected by prob threshold
        - detected_count_ba: Count detected by BA threshold
    """
    total = len(extraction_results)

    if total == 0:
        return {
            'tpr_prob': 0.0,
            'tpr_ba': 0.0,
            'avg_confidence': 0.0,
            'avg_bit_accuracy': 0.0,
            'fnr_prob': 1.0,
            'fnr_ba': 1.0,
            'total_samples': 0,
            'detected_count_prob': 0,
            'detected_count_ba': 0
        }

    # TPR based on detection probability
    detected_prob = [
        r for r in extraction_results
        if r.get('detection_prob', r.get('confidence', 0.0)) > tau_prob
    ]
    tpr_prob = len(detected_prob) / total

    # TPR based on bit accuracy
    bit_accuracies = []
    for r in extraction_results:
        if 'bit_accuracy' in r:
            bit_accuracies.append(r['bit_accuracy'])
        elif ('raw_preds' in r or 'raw_bits' in r) and original_msg_bits is not None:
            # Compute bit accuracy if not provided
            # BUGFIX: Support both 'raw_preds' (Image-Bench) and 'raw_bits' (AudioSeal)
            # AudioSealWrapper.extract() returns 'raw_bits', not 'raw_preds'
            raw_preds = r.get('raw_preds', r.get('raw_bits'))

            if isinstance(raw_preds, np.ndarray):
                raw_preds = torch.from_numpy(raw_preds)
            if isinstance(original_msg_bits, np.ndarray):
                original_msg_bits_tensor = torch.from_numpy(original_msg_bits)
            else:
                original_msg_bits_tensor = original_msg_bits

            ba = compute_audio_bit_accuracy(
                raw_preds.flatten(),
                original_msg_bits_tensor.flatten()
            )
            bit_accuracies.append(ba)
        else:
            bit_accuracies.append(0.0)

    # Add bit_accuracy to extraction_results for consistency
    for i, r in enumerate(extraction_results):
        if 'bit_accuracy' not in r and i < len(bit_accuracies):
            r['bit_accuracy'] = bit_accuracies[i]

    detected_ba = [
        r for r, ba in zip(extraction_results, bit_accuracies)
        if ba > tau_ba
    ]
    tpr_ba = len(detected_ba) / total

    # Average confidence (detection probability)
    confidences = [
        r.get('detection_prob', r.get('confidence', 0.0))
        for r in extraction_results
    ]
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    # Average bit accuracy
    avg_bit_accuracy = float(np.mean(bit_accuracies)) if bit_accuracies else 0.0

    return {
        'tpr_prob': float(tpr_prob),
        'tpr_ba': float(tpr_ba),
        'avg_confidence': avg_confidence,
        'avg_bit_accuracy': avg_bit_accuracy,
        'fnr_prob': 1.0 - tpr_prob,
        'fnr_ba': 1.0 - tpr_ba,
        'total_samples': total,
        'detected_count_prob': len(detected_prob),
        'detected_count_ba': len(detected_ba)
    }


def aggregate_detection_metrics_by_attack(
    extraction_results_by_attack: Dict[str, Dict[Any, List[Dict]]],
    original_msg_bits: Optional[np.ndarray] = None,
    tau_prob: float = 0.15,
    tau_ba: float = 0.1
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate detection metrics by attack type and strength

    Args:
        extraction_results_by_attack: Nested dict {attack_type: {strength: [results]}}
        original_msg_bits: Original message bits
        tau_prob: Detection probability threshold
        tau_ba: Bit accuracy threshold

    Returns:
        Dict: Aggregated metrics by attack type and strength
    """
    aggregated = {}

    for attack_type, strength_dict in extraction_results_by_attack.items():
        aggregated[attack_type] = {}

        for strength, results in strength_dict.items():
            metrics = compute_audio_detection_metrics(
                results,
                original_msg_bits=original_msg_bits,
                tau_prob=tau_prob,
                tau_ba=tau_ba
            )
            # Convert strength to string (handle tuples for echo attack)
            if isinstance(strength, tuple):
                strength_str = '_'.join(map(str, strength))
            else:
                strength_str = str(strength)

            aggregated[attack_type][strength_str] = metrics

    return aggregated


def print_detection_summary(metrics_by_attack: Dict[str, Dict[str, Any]]):
    """
    Print detection metrics summary

    Args:
        metrics_by_attack: Metrics dictionary organized by attack type and strength
    """
    print("\n" + "=" * 80)
    print("DETECTION METRICS SUMMARY")
    print("=" * 80)

    for attack_type, strength_dict in metrics_by_attack.items():
        print(f"\n{attack_type.upper()}:")
        print(f"{'Strength':<15} {'TPR(prob)':<12} {'TPR(BA)':<12} {'Avg Conf':<12} {'Avg BA':<12}")
        print("-" * 80)

        for strength, metrics in strength_dict.items():
            print(
                f"{str(strength):<15} "
                f"{metrics['tpr_prob']:<12.2%} "
                f"{metrics['tpr_ba']:<12.2%} "
                f"{metrics['avg_confidence']:<12.4f} "
                f"{metrics['avg_bit_accuracy']:<12.4f}"
            )

    print("\n" + "=" * 80)
