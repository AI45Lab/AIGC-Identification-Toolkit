#!/usr/bin/env python
"""
Video-Bench: VideoSeal Robustness Evaluation

CLI entry point for running complete VideoSeal watermark robustness evaluation
following VideoMarkBench methodology.

Usage:
    python run_benchmark.py --config configs/videoseal_robustness.yaml
    python run_benchmark.py --config configs/videoseal_robustness.yaml --max_videos 10
    python run_benchmark.py --config configs/videoseal_robustness.yaml --device cpu
"""

import sys
import argparse
from pathlib import Path

# Add benchmark directory to path for relative imports
BENCHMARK_ROOT = Path(__file__).parent
PROJECT_ROOT = BENCHMARK_ROOT.parent.parent
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from evaluators.benchmark_runner import VideoBenchmarkRunner


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Video-Bench: VideoSeal Robustness Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--max_videos',
        type=int,
        default=None,
        help='Maximum number of videos to evaluate (default: all)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Computation device'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--skip-to-extraction',
        action='store_true',
        help='Skip stages 1-3 (load/embed/attack) and start from stage 4 (extraction). '
             'Requires watermarked/ and attacked/ directories with .mp4 files to exist.'
    )

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    # Print banner
    print("\n" + "="*70)
    print(" " * 15 + "Video-Bench: VideoSeal Robustness Evaluation")
    print("="*70)
    print(f"\nConfiguration: {config_path}")
    print(f"Device:        {args.device}")
    if args.max_videos:
        print(f"Max Videos:    {args.max_videos}")
    print()

    try:
        # Initialize benchmark runner
        runner = VideoBenchmarkRunner(
            config_path=str(config_path),
            device=args.device,
            skip_to_extraction=args.skip_to_extraction
        )

        # Run evaluation
        results = runner.run(max_videos=args.max_videos)

        # Print summary
        print("\n" + "="*70)
        print("Evaluation Summary")
        print("="*70)

        print("\nQuality Metrics:")
        print(f"  PSNR:  {results['quality']['psnr']:.2f} dB")
        print(f"  SSIM:  {results['quality']['ssim']:.4f}")
        print(f"  tLP:   {results['quality']['tLP']:.6f}")

        print("\nDetection Metrics:")
        print(f"  FNR (False Negative Rate):  {results['detection']['fnr']:.4f}")
        print(f"  Bit Accuracy:               {results['detection']['bit_accuracy']:.4f}")
        print(f"  Avg Confidence:             {results['detection']['avg_confidence']:.4f}")

        print(f"\nPer-Attack Results: {len(results['per_attack'])} configurations")

        print("\n" + "="*70)
        print(f"✅ Evaluation completed successfully!")
        print(f"📊 Results saved to: {runner.output_dir}")
        print("="*70 + "\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        return 130

    except Exception as e:
        print(f"\n\n❌ Error during evaluation:")
        print(f"   {type(e).__name__}: {e}")

        if args.verbose:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())
