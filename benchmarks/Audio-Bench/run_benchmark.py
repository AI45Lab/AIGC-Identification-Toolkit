#!/usr/bin/env python3
"""
Audio-Bench Evaluation Entry Point

Runs AudioSeal watermark robustness evaluation against 9 audio attack types.

Usage:
    # Full evaluation (1000 audios × 9 attacks × 5 strengths)
    python run_benchmark.py

    # Quick test (10 audios)
    python run_benchmark.py --max_audios 10

    # Custom config
    python run_benchmark.py --config ./my_config.yaml

    # CPU-only mode
    python run_benchmark.py --device cpu
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from evaluators.benchmark_runner import AudioBenchmarkRunner


def main():
    """Command-line interface for Audio-Bench evaluation."""

    parser = argparse.ArgumentParser(
        description='Audio-Bench: AudioSeal Watermark Robustness Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 10 audios
  python run_benchmark.py --max_audios 10

  # Full evaluation with default config
  python run_benchmark.py

  # Use custom configuration file
  python run_benchmark.py --config my_config.yaml

  # CPU-only evaluation
  python run_benchmark.py --device cpu

  # High verbosity for debugging
  python run_benchmark.py --max_audios 5 -v

Output:
  Results are saved to: benchmarks/Audio-Bench/results/audioseal_robustness/
  - metrics.json: Complete evaluation metrics
  - audioseal_tpr_prob_radar.png: TPR (detection probability) radar chart
  - audioseal_tpr_ba_radar.png: TPR (bit accuracy) radar chart
  - audioseal_bit_accuracy_radar.png: Bit accuracy radar chart
  - watermarked/: Watermarked audio files
  - attacked/: Attacked audio files (organized by attack type and strength)

References:
  - AudioSeal Paper: https://arxiv.org/abs/2401.17264
  - AudioMarkBench: https://github.com/moyangkuo/AudioMarkBench
  - Image-Bench: benchmarks/Image-Bench/
        """
    )

    parser.add_argument(
        '--config',
        '-c',
        type=str,
        default='benchmarks/Audio-Bench/configs/audioseal_robustness.yaml',
        help='Path to YAML configuration file (default: configs/audioseal_robustness.yaml)'
    )

    parser.add_argument(
        '--max_audios',
        '-n',
        type=int,
        default=None,
        help='Maximum number of audios to evaluate (default: all audios in dataset). '
             'Use --max_audios 10 for quick testing.'
    )

    parser.add_argument(
        '--device',
        '-d',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device for watermarking (default: auto-detect). '
             'Override config file setting if specified.'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output (shows detailed progress)'
    )

    args = parser.parse_args()

    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Configuration file not found: {args.config}")
        print(f"\nExpected location: {config_path.absolute()}")
        print("\nPlease create a config file or specify a valid path with --config")
        sys.exit(1)

    # Print header
    print("\n" + "=" * 70)
    print("🎵 Audio-Bench: AudioSeal Robustness Evaluation")
    print("=" * 70)
    print(f"\nConfiguration: {config_path.name}")

    if args.max_audios:
        print(f"Mode: Quick test ({args.max_audios} audios)")
    else:
        print("Mode: Full evaluation (all audios in dataset)")

    if args.device:
        print(f"Device: {args.device} (override)")
    else:
        print("Device: Auto-detect from config")

    print("\n" + "-" * 70)

    # Create runner and execute
    try:
        runner = AudioBenchmarkRunner(str(config_path))

        # Override device if specified
        if args.device:
            runner.watermarker.device = args.device
            print(f"\n✓ Device override applied: {args.device}\n")

        # Run evaluation
        metrics = runner.run(max_audios=args.max_audios)

        # Print success summary
        if metrics:
            print("\n" + "=" * 70)
            print("✅ Evaluation completed successfully!")
            print("=" * 70)
            print(f"\n📂 Results directory: {runner.output_dir}")
            print(f"📊 Metrics file: {runner.output_dir / 'metrics.json'}")
            print(f"\n🎨 Radar charts:")
            print(f"   - {runner.output_dir / 'audioseal_tpr_prob_radar.png'}")
            print(f"   - {runner.output_dir / 'audioseal_tpr_ba_radar.png'}")
            print(f"   - {runner.output_dir / 'audioseal_bit_accuracy_radar.png'}")
            print("\n" + "=" * 70)
            print("\n✨ Next steps:")
            print("1. Review metrics.json for detailed statistics")
            print("2. Examine radar charts for visual robustness analysis")
            print("3. Compare results with Image-Bench (VideoSeal) if available")
            print("\n" + "=" * 70 + "\n")
        else:
            print("\n❌ Evaluation failed. Please check error messages above.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user (Ctrl+C)")
        print("Partial results may be available in the output directory.")
        sys.exit(130)

    except Exception as e:
        print(f"\n❌ Evaluation failed with error:")
        print(f"\n{type(e).__name__}: {e}")

        if args.verbose:
            print("\nFull traceback:")
            import traceback
            traceback.print_exc()

        print("\n💡 Troubleshooting:")
        print("1. Verify dataset path in config file")
        print("2. Check that audio files (.wav) exist in dataset directory")
        print("3. Ensure dependencies are installed: torch, torchaudio, librosa, julius, audiomentations")
        print("4. Try running with --max_audios 1 to test with a single audio")
        print("5. Use --verbose flag for detailed error messages")

        sys.exit(1)


if __name__ == "__main__":
    main()
