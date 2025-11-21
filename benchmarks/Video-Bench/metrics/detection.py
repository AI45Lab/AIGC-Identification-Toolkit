"""
Video Watermark Detection Metrics for VideoSeal Evaluation

Implements FNR (False Negative Rate), Bit Accuracy, and detection aggregation
following VideoMarkBench methodology.
"""

import numpy as np
from scipy.stats import binom
from typing import List, Dict, Any


def message_to_bits(message: str, target_bits: int = 256) -> np.ndarray:
    """
    Convert message string to bit array (matching VideoSeal wrapper logic)

    This function replicates the VideoSealWrapper._string_to_bits() method
    to ensure ground truth bits match the actual embedded watermark.

    Args:
        message: Input message string (e.g., "VideoMarkBench-2025")
        target_bits: Target bit length (default 256 for VideoSeal)

    Returns:
        np.ndarray: Bit array, shape (target_bits,), values in {0, 1}
    """
    # Convert string to bytes (UTF-8 encoding)
    message_bytes = message.encode('utf-8')

    # Convert to bit array (LSB first to match VideoSeal)
    bit_array = []
    for byte in message_bytes:
        # Convert each byte to 8 bits (LSB first)
        for i in range(8):
            bit_array.append((byte >> i) & 1)

    # Pad to minimum length
    min_bits = 64
    while len(bit_array) < min_bits:
        bit_array.append(0)

    # Truncate or pad to target length
    if len(bit_array) > target_bits:
        bit_array = bit_array[:target_bits]
    else:
        while len(bit_array) < target_bits:
            bit_array.append(0)

    return np.array(bit_array, dtype=int)


def detection_threshold(decode, watermark_gt, tau_ba=179/256, binom_p=0.0000661, binom_thresh=1e-4):
    """
    VideoSeal official detection logic with binomial test

    Directly copied from: benchmarks/VideoMarkBench/aggregation.py
    Modified to accept actual watermark ground truth instead of fixed pattern.

    Args:
        decode: np.ndarray, per-frame logits, shape (n_frames, 256)
        watermark_gt: np.ndarray, ground truth watermark bits, shape (256,)
        tau_ba: float, frame-level BA threshold, default 179/256 ≈ 0.699 (equivalent to 67/96)
        binom_p: float, random guess probability, default 0.0000661
        binom_thresh: float, significance level, default 1e-4

    Returns:
        bool: True if watermark detected, False otherwise

    Detection Logic:
        Step 1: Compute per-frame Bit Accuracy (BA)
        Step 2: Count frames with BA >= tau_ba
        Step 3: Binomial test to determine if detected_frames >= k
                where k is computed such that binom.sf(k-1, n, p) <= threshold
    """
    # Ground truth watermark is now passed as parameter
    # REMOVED: watermark_gt = np.array([0, 1] * 128)

    # Step 1: Compute per-frame Bit Accuracy
    # Convert logits to binary predictions: logit >= 0 → 1, logit < 0 → 0
    bitacc = np.mean((decode >= 0) == watermark_gt, axis=1)  # shape (n_frames,)

    # Step 2: Count frames with BA >= tau_ba
    detected_frames = np.sum(bitacc >= tau_ba)

    # Step 3: Compute binomial test threshold k
    n = decode.shape[0]  # Total number of frames

    # Find minimum k such that P(X >= k) <= threshold
    # where X ~ Binomial(n, p) under null hypothesis (random guess)
    for i in range(n + 1):
        if binom.sf(i - 1, n, binom_p) <= binom_thresh:
            k = i
            break
    else:
        # If no k found (should not happen), set k = n + 1 (no detection)
        k = n + 1

    # Detection decision
    return detected_frames >= k


def compute_bit_accuracy(predicted_bits, original_bits):
    """
    Compute Bit Accuracy (BA) - proportion of correctly predicted bits

    Args:
        predicted_bits: np.ndarray, predicted watermark bits (0 or 1), shape (96,)
        original_bits: np.ndarray, original watermark bits (0 or 1), shape (96,)

    Returns:
        float: Bit Accuracy in range [0, 1]
    """
    return np.mean(predicted_bits == original_bits)


def compute_fnr(all_video_logits, config, watermark_message: str):
    """
    Compute False Negative Rate (FNR) - proportion of missed watermarks

    FNR = (Number of watermarked videos NOT detected) / (Total watermarked videos)

    Args:
        all_video_logits: list of np.ndarray, each shape (n_frames_i, 256)
                         List of per-frame logits for all test videos
        config: dict, configuration containing detection parameters:
                - 'tau_ba': frame-level BA threshold
                - 'binom_p': random guess probability
                - 'binom_threshold': significance level
        watermark_message: str, original watermark message (e.g., "VideoMarkBench-2025")

    Returns:
        float: FNR value in range [0, 1], lower is better
    """
    # Convert message to ground truth bits
    watermark_gt = message_to_bits(watermark_message, target_bits=256)

    detected_count = 0
    total_count = len(all_video_logits)

    for video_logits in all_video_logits:
        # Apply VideoSeal detection logic with actual ground truth
        if detection_threshold(
            video_logits,
            watermark_gt,  # Pass actual ground truth
            tau_ba=config.get('tau_ba', 179/256),
            binom_p=config.get('binom_p', 0.0000661),
            binom_thresh=config.get('binom_threshold', 1e-4)
        ):
            detected_count += 1

    # FNR = miss rate
    fnr = 1 - (detected_count / total_count)
    return fnr


def compute_fpr(all_video_logits, config, watermark_message: str):
    """
    Compute False Positive Rate (FPR) - proportion of false detections

    Note: This requires a dataset of non-watermarked videos.
    FPR = (Number of non-watermarked videos DETECTED as watermarked) / (Total non-watermarked videos)

    Args:
        all_video_logits: list of np.ndarray, each shape (n_frames_i, 256)
                         List of per-frame logits for NON-WATERMARKED videos
        config: dict, configuration containing detection parameters
        watermark_message: str, watermark message used for ground truth

    Returns:
        float: FPR value in range [0, 1], lower is better
    """
    # Convert message to ground truth bits
    watermark_gt = message_to_bits(watermark_message, target_bits=256)

    false_positive_count = 0
    total_count = len(all_video_logits)

    for video_logits in all_video_logits:
        # For non-watermarked videos, detection should be False
        if detection_threshold(
            video_logits,
            watermark_gt,  # Pass actual ground truth
            tau_ba=config.get('tau_ba', 179/256),
            binom_p=config.get('binom_p', 0.0000661),
            binom_thresh=config.get('binom_threshold', 1e-4)
        ):
            false_positive_count += 1

    fpr = false_positive_count / total_count if total_count > 0 else 0.0
    return fpr


def compute_avg_bit_accuracy(all_video_logits, watermark_message: str):
    """
    Compute average Bit Accuracy across all videos and frames

    Args:
        all_video_logits: list of np.ndarray, each shape (n_frames_i, 256)
        watermark_message: str, original watermark message (e.g., "VideoMarkBench-2025")

    Returns:
        float: Average BA across all videos and all frames
    """
    # Convert message to ground truth bits
    watermark_gt = message_to_bits(watermark_message, target_bits=256)

    all_ba_values = []

    for video_logits in all_video_logits:
        # Convert logits to binary predictions
        predicted_bits = (video_logits >= 0).astype(int)  # shape (n_frames, 256)

        # Compute BA for each frame
        for frame_pred in predicted_bits:
            ba = compute_bit_accuracy(frame_pred, watermark_gt)
            all_ba_values.append(ba)

    return np.mean(all_ba_values) if all_ba_values else 0.0


def compute_avg_confidence(all_video_logits):
    """
    Compute average detection confidence across all videos

    Confidence is defined as the mean absolute deviation from 0.5
    after sigmoid transformation of logits.

    Args:
        all_video_logits: list of np.ndarray, each shape (n_frames_i, 96)

    Returns:
        float: Average confidence in range [0, 1]
    """
    all_confidences = []

    for video_logits in all_video_logits:
        # Convert logits to probabilities using sigmoid
        # sigmoid(x) = 1 / (1 + exp(-x))
        probs = 1 / (1 + np.exp(-video_logits))  # shape (n_frames, 96)

        # Confidence: mean absolute deviation from 0.5
        # Higher deviation indicates stronger signal
        confidence = np.mean(np.abs(probs - 0.5)) * 2  # Scale to [0, 1]
        all_confidences.append(confidence)

    return np.mean(all_confidences) if all_confidences else 0.0


def compute_all_detection_metrics(
    watermarked_video_logits: List[np.ndarray],
    config: Dict[str, Any],
    watermark_message: str,
    non_watermarked_video_logits: List[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all detection metrics for watermarked (and optionally non-watermarked) videos

    Args:
        watermarked_video_logits: list of np.ndarray, logits from watermarked videos
        config: dict, detection configuration parameters
        watermark_message: str, original watermark message (e.g., "VideoMarkBench-2025")
        non_watermarked_video_logits: optional list of np.ndarray, logits from non-watermarked videos

    Returns:
        dict: Dictionary containing all detection metrics
            {
                'fnr': float,              # Lower is better
                'bit_accuracy': float,     # Higher is better (0-1)
                'avg_confidence': float,   # Higher is better (0-1)
                'fpr': float (optional)    # Lower is better, only if non-watermarked data provided
            }
    """
    metrics = {}

    # Ground truth is now derived from actual watermark message
    # REMOVED: watermark_gt = np.array([0, 1] * 128)

    # Compute FNR with actual message
    metrics['fnr'] = compute_fnr(watermarked_video_logits, config, watermark_message)

    # Compute average Bit Accuracy with actual message
    metrics['bit_accuracy'] = compute_avg_bit_accuracy(watermarked_video_logits, watermark_message)

    # Compute average confidence (unchanged)
    metrics['avg_confidence'] = compute_avg_confidence(watermarked_video_logits)

    # Compute FPR if non-watermarked data provided
    if non_watermarked_video_logits is not None:
        metrics['fpr'] = compute_fpr(non_watermarked_video_logits, config, watermark_message)

    return metrics


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing detection metrics with synthetic data...")

    # Simulate watermarked video logits (strong signal)
    n_videos = 10
    n_frames_per_video = 30
    watermark_gt = np.array([0, 1] * 48)

    print(f"\nGenerating {n_videos} synthetic watermarked videos...")
    watermarked_logits = []
    for _ in range(n_videos):
        # Strong signal: logits with correct sign matching watermark
        logits = np.random.randn(n_frames_per_video, 96) + 2.0  # Positive bias
        # Adjust signs to match watermark
        for i in range(96):
            if watermark_gt[i] == 0:
                logits[:, i] = -np.abs(logits[:, i])  # Force negative
            else:
                logits[:, i] = np.abs(logits[:, i])   # Force positive
        watermarked_logits.append(logits)

    # Test detection config
    config = {
        'tau_ba': 67/96,
        'binom_p': 0.0000661,
        'binom_threshold': 1e-4
    }

    print("\nComputing detection metrics...")
    metrics = compute_all_detection_metrics(watermarked_logits, config)

    print("\nResults:")
    print(f"FNR (False Negative Rate): {metrics['fnr']:.4f}")
    print(f"Bit Accuracy: {metrics['bit_accuracy']:.4f}")
    print(f"Avg Confidence: {metrics['avg_confidence']:.4f}")

    print("\nDetection metrics computed successfully!")
