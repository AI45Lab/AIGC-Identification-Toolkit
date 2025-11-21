#!/usr/bin/env python3
"""
Radar Chart Visualization for AudioSeal Watermark Robustness

Generates three radar charts visualizing watermark robustness metrics:
- TPR (prob): True Positive Rate based on detection probability (tau_prob=0.15)
- TPR (BA): True Positive Rate based on bit accuracy (tau_ba=0.1)
- Average Confidence Score

Adapted from Image-Bench radar chart implementation for audio watermarking.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import sys


# ============================================================================
# Configuration
# ============================================================================

# Audio attack types in fixed order for consistent visualization
# Note: time_stretch is excluded due to bidirectional strength semantics (1.0 is baseline)
ATTACK_TYPES = [
    'gaussian_noise',
    'background_noise',
    'quantization',
    'lowpass_filter',
    'highpass_filter',
    'smooth',
    'echo',
    'mp3_compression'
]

ATTACK_LABELS = [
    'Gaussian\nNoise',
    'Background\nNoise',
    'Quantization',
    'Lowpass\nFilter',
    'Highpass\nFilter',
    'Smooth',
    'Echo',
    'MP3'
]

# Attack strength direction configuration
# Controls how attack parameters are sorted to determine strength ordering
# - 'ascending': higher parameter value = stronger attack (e.g., noise std, smooth window)
# - 'descending': lower parameter value = stronger attack (e.g., SNR, quantization bits, MP3 bitrate)
# Note: time_stretch is excluded from radar chart (bidirectional strength: 1.0 is baseline)
ATTACK_STRENGTH_DIRECTION = {
    'gaussian_noise': 'descending',      # SNR 40dB > 5dB: weaker > stronger ⚠️ reversed
    'background_noise': 'descending',    # SNR 40dB > 5dB: weaker > stronger ⚠️ reversed
    'quantization': 'descending',        # Bits 64 > 4: weaker > stronger ⚠️ reversed
    'lowpass_filter': 'ascending',       # Cutoff 0.1 < 0.5: weaker < stronger
    'highpass_filter': 'ascending',      # Cutoff 0.1 < 0.5: weaker < stronger
    'smooth': 'ascending',               # Window 6 < 22: weaker < stronger
    'echo': 'ascending',                 # Duration 0.1 < 0.5: weaker < stronger
    'mp3_compression': 'descending',     # Bitrate 256kbps > 64kbps: weaker > stronger ⚠️ reversed
}

# Metric configurations
METRICS_CONFIG = {
    'tpr_prob': {
        'name': 'TPR (Detection Probability > 0.15)',
        'filename': 'audioseal_tpr_prob_radar.png',
        'ylim': (0, 1.0),
        'yticks': [0.2, 0.4, 0.6, 0.8, 1.0],
        'normalize': False
    },
    'tpr_ba': {
        'name': 'TPR (Bit Accuracy > 0.1)',
        'filename': 'audioseal_tpr_ba_radar.png',
        'ylim': (0, 1.0),
        'yticks': [0.2, 0.4, 0.6, 0.8, 1.0],
        'normalize': False
    },
    'avg_bit_accuracy': {
        'name': 'Average Bit Accuracy',
        'filename': 'audioseal_bit_accuracy_radar.png',
        'ylim': (0, 1.0),
        'yticks': [0.2, 0.4, 0.6, 0.8, 1.0],
        'normalize': False
    },
    'avg_confidence': {
        'name': 'Average Confidence Score',
        'filename': 'audioseal_avg_confidence_radar.png',
        'ylim': (0, 1.0),
        'yticks': [0.2, 0.4, 0.6, 0.8, 1.0],
        'normalize': False
    }
}

# Color schemes for intensity levels (5 levels from weakest to strongest)
COLOR_SCHEMES = {
    'blue': ['#0D47A1', '#1976D2', '#42A5F5', '#90CAF9', '#BBDEFB'],
    'green': ['#1B5E20', '#388E3C', '#66BB6A', '#A5D6A7', '#C8E6C9'],
    'orange': ['#BF360C', '#E64A19', '#FF7043', '#FFAB91', '#FFCCBC'],
    'purple': ['#4A148C', '#7B1FA2', '#AB47BC', '#CE93D8', '#E1BEE7'],
    'teal': ['#004D40', '#00796B', '#26A69A', '#80CBC4', '#B2DFDB']
}

# Default color scheme
DEFAULT_COLORS = COLOR_SCHEMES['blue']


# ============================================================================
# Data Extraction Functions
# ============================================================================

def load_metrics(json_path: str) -> Dict:
    """
    Load metrics from JSON file.

    Args:
        json_path: Path to metrics.json

    Returns:
        Parsed JSON data
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_robustness_data(metrics: Dict, metric_key: str) -> Dict[str, List[float]]:
    """
    Extract robustness data for a specific metric across all attacks and intensities.

    Args:
        metrics: Full metrics dictionary
        metric_key: One of 'tpr_prob', 'tpr_ba', 'avg_bit_accuracy', 'avg_confidence'

    Returns:
        Dictionary mapping attack type to list of 5 intensity values

    Example:
        {
            'gaussian_noise': [1.0, 1.0, 0.99, 0.98, 0.97],
            'time_stretch': [1.0, 1.0, 0.98, 0.96, 0.94],
            ...
        }
    """
    robustness = metrics.get('robustness_by_attack', {})
    data = {}

    for attack_type in ATTACK_TYPES:
        attack_data = robustness.get(attack_type, {})

        if not attack_data:
            # Skip if attack type not found
            continue

        # Determine sorting direction based on attack type
        # Some attacks (e.g., SNR, quantization bits, MP3 bitrate) have reversed strength semantics
        direction = ATTACK_STRENGTH_DIRECTION.get(attack_type, 'ascending')
        reverse = (direction == 'descending')

        # Sort by strength parameter (weak to strong based on direction)
        try:
            strengths = sorted(attack_data.keys(), key=lambda x: float(x), reverse=reverse)
        except ValueError:
            # Handle non-numeric strength keys (fallback to lexicographic sorting)
            strengths = sorted(attack_data.keys(), reverse=reverse)

        # Take first 5 strengths (weakest to strongest)
        strengths = strengths[:5]

        # Extract metric values
        values = []
        for strength in strengths:
            strength_metrics = attack_data[strength]
            value = strength_metrics.get(metric_key, 0.0)
            values.append(value)

        if values:
            data[attack_type] = values

    return data


# ============================================================================
# Radar Chart Plotting
# ============================================================================

def plot_radar_chart(
    data: Dict[str, List[float]],
    metric_config: Dict,
    colors: List[str] = None,
    output_path: str = None,
    dpi: int = 300
):
    """
    Create a radar chart for a single metric.

    Args:
        data: Dictionary mapping attack type to list of 5 intensity values
        metric_config: Configuration dict from METRICS_CONFIG
        colors: List of 5 colors for intensity levels (default: blue scheme)
        output_path: Path to save the figure (default: current dir)
        dpi: Resolution in dots per inch
    """
    if colors is None:
        colors = DEFAULT_COLORS

    # Filter out attacks with missing data
    valid_attacks = [attack for attack in ATTACK_TYPES if attack in data]
    valid_labels = [ATTACK_LABELS[ATTACK_TYPES.index(attack)] for attack in valid_attacks]

    if not valid_attacks:
        print(f"⚠️  Warning: No valid data for metric, skipping chart generation")
        return

    # Number of variables (attack types)
    num_vars = len(valid_attacks)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Close the plot (connect back to first point)
    angles += angles[:1]

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

    # Get normalization settings
    normalize = metric_config.get('normalize', False)
    normalize_factor = metric_config.get('normalize_factor', 1.0)

    # Determine max number of intensity levels
    max_intensities = max(len(values) for values in data.values())

    # Plot data for each intensity level
    for intensity_idx in range(max_intensities):
        # Extract values for this intensity across all attacks
        values = []
        for attack in valid_attacks:
            attack_values = data[attack]
            if intensity_idx < len(attack_values):
                values.append(attack_values[intensity_idx])
            else:
                values.append(0.0)  # Fallback for missing data

        # Normalize if needed
        if normalize:
            values = [v / normalize_factor for v in values]

        # Close the plot
        values += values[:1]

        # Plot line with fill
        color = colors[intensity_idx] if intensity_idx < len(colors) else colors[-1]
        label = f'Intensity {intensity_idx + 1} {"(weakest)" if intensity_idx == 0 else "(strongest)" if intensity_idx == max_intensities - 1 else ""}'

        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=label, markersize=6)
        ax.fill(angles, values, alpha=0.15, color=color)

    # Customize plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(valid_labels, fontsize=11)

    # Set y-axis limits and ticks
    ylim = metric_config['ylim']
    yticks = metric_config['yticks']
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)

    # Use custom tick labels if provided (for normalized metrics)
    if 'ytick_labels' in metric_config:
        ax.set_yticklabels(metric_config['ytick_labels'], fontsize=10)
    else:
        ax.set_yticklabels([f'{y:.1f}' for y in yticks], fontsize=10)

    # Grid style
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)

    # Title
    plt.title(
        f"{metric_config['name']}\nAudioSeal Watermark Robustness",
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Legend
    plt.legend(
        loc='upper right',
        bbox_to_anchor=(1.3, 1.1),
        fontsize=10,
        framealpha=0.9
    )

    # Tight layout
    plt.tight_layout()

    # Save figure
    if output_path:
        output_file = Path(output_path) / metric_config['filename']
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
    else:
        output_file = metric_config['filename']
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")

    plt.close()


# ============================================================================
# Main Function
# ============================================================================

def generate_all_radars(
    metrics_path: str,
    output_dir: str = None,
    color_scheme: str = 'blue',
    dpi: int = 300
):
    """
    Generate all radar charts from metrics JSON.

    Args:
        metrics_path: Path to metrics.json file
        output_dir: Output directory for charts (default: same as metrics.json)
        color_scheme: Color scheme name from COLOR_SCHEMES
        dpi: Resolution in dots per inch
    """
    print("=" * 70)
    print("🎨 Generating Radar Charts for AudioSeal Robustness")
    print("=" * 70)
    print(f"\n📂 Loading metrics from: {metrics_path}")

    # Load data
    metrics = load_metrics(metrics_path)

    # Determine output directory
    if output_dir is None:
        output_dir = Path(metrics_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {output_dir}\n")

    # Get colors
    colors = COLOR_SCHEMES.get(color_scheme, DEFAULT_COLORS)

    # Generate charts for each metric
    for metric_key, metric_config in METRICS_CONFIG.items():
        print(f"🎯 Generating {metric_config['name']}...")

        # Extract data
        data = extract_robustness_data(metrics, metric_key)

        # Validate data
        if not data:
            print(f"⚠️  Warning: No data for {metric_key}, skipping...")
            continue

        # Plot radar chart
        plot_radar_chart(
            data=data,
            metric_config=metric_config,
            colors=colors,
            output_path=output_dir,
            dpi=dpi
        )

    print("\n" + "=" * 70)
    print("✅ All radar charts generated successfully!")
    print("=" * 70)
    print(f"\n📊 Generated files:")
    for metric_config in METRICS_CONFIG.values():
        print(f"   - {output_dir / metric_config['filename']}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate radar charts for AudioSeal watermark robustness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from default path
  python plot_radar.py benchmarks/Audio-Bench/results/audioseal_robustness/metrics.json

  # Specify custom output directory
  python plot_radar.py metrics.json --output ./charts

  # Use different color scheme
  python plot_radar.py metrics.json --colors green

  # High resolution output
  python plot_radar.py metrics.json --dpi 600
        """
    )

    parser.add_argument(
        'metrics_path',
        type=str,
        help='Path to metrics.json file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: same as metrics.json)'
    )

    parser.add_argument(
        '--colors', '-c',
        type=str,
        default='blue',
        choices=list(COLOR_SCHEMES.keys()),
        help='Color scheme (default: blue)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Output resolution in DPI (default: 300)'
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.metrics_path).exists():
        print(f"❌ Error: File not found: {args.metrics_path}")
        sys.exit(1)

    # Generate charts
    try:
        generate_all_radars(
            metrics_path=args.metrics_path,
            output_dir=args.output,
            color_scheme=args.colors,
            dpi=args.dpi
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
