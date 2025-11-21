"""
Image-Bench Benchmark Runner for VideoSeal

Orchestrates the complete evaluation pipeline:
1. Load test images
2. Embed watermarks
3. Apply distortion attacks
4. Extract watermarks
5. Compute metrics
6. Save results
"""

import json
import yaml
import sys
from pathlib import Path
from PIL import Image
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add paths for imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add VINE to path (we only need the distortion scripts, not the full package)
vine_path = project_root / 'benchmarks' / 'VINE'
if vine_path.exists():
    sys.path.insert(0, str(vine_path))

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not available, progress bars disabled")

from src.image_watermark.image_watermark import ImageWatermark

# Import VINE distortions (only requires PIL, numpy, torch - no full installation needed)
try:
    from vine.w_bench_utils.distortion.distortions import apply_single_distortion
    VINE_AVAILABLE = True
except ImportError as e:
    VINE_AVAILABLE = False
    print("⚠️  VINE distortion module not found.")
    print(f"   Error: {e}")
    print("   Solution: Ensure benchmarks/VINE directory exists with distortion scripts.")

# Import local metrics using importlib (handle hyphenated directory names)
import importlib.util
import os

# Get the Image-Bench directory path
image_bench_dir = Path(__file__).resolve().parent.parent

# Import quality metrics module
quality_spec = importlib.util.spec_from_file_location(
    "quality",
    image_bench_dir / "metrics" / "quality.py"
)
quality_module = importlib.util.module_from_spec(quality_spec)
quality_spec.loader.exec_module(quality_module)
compute_quality_metrics = quality_module.compute_quality_metrics

# Import detection metrics module
detection_spec = importlib.util.spec_from_file_location(
    "detection",
    image_bench_dir / "metrics" / "detection.py"
)
detection_module = importlib.util.module_from_spec(detection_spec)
detection_spec.loader.exec_module(detection_module)
compute_detection_metrics = detection_module.compute_detection_metrics

# Import message encoding utilities
encoding_spec = importlib.util.spec_from_file_location(
    "message_encoding",
    image_bench_dir / "utils" / "message_encoding.py"
)
encoding_module = importlib.util.module_from_spec(encoding_spec)
encoding_spec.loader.exec_module(encoding_module)
string_to_bits = encoding_module.string_to_bits


class BenchmarkRunner:
    """
    Evaluates VideoSeal watermark robustness against traditional distortions.

    Workflow:
        1. Load images from Image-Bench DISTORTION_1K
        2. Embed watermarks using ImageWatermark (VideoSeal backend)
        3. Apply 5 distortion types at various strengths
        4. Extract watermarks from attacked images
        5. Compute quality metrics (PSNR, SSIM, LPIPS)
        6. Compute detection metrics (TPR, bit accuracy, confidence)
        7. Save results to JSON
    """

    def __init__(self, config_path: str):
        """
        Initialize runner with YAML config.

        Args:
            config_path: Path to videoseal_distortion.yaml
        """
        print("=" * 70)
        print("🚀 Image-Bench VideoSeal Robustness Evaluation")
        print("=" * 70)
        print(f"\n📂 Loading config from: {config_path}\n")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        if not VINE_AVAILABLE:
            raise ImportError(
                "VINE package not installed. Please install it first:\n"
                "  cd benchmarks/VINE && pip install -e ."
            )

        # Initialize ImageWatermark with VideoSeal backend
        print("🔧 Initializing VideoSeal watermarker...")
        self.watermarker = ImageWatermark()
        self.watermarker.algorithm = 'videoseal'

        # Setup paths
        self.dataset_path = Path(self.config['dataset']['path'])
        self.output_dir = Path(self.config['output']['base_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"✓ Dataset path: {self.dataset_path}")
        print(f"✓ Output directory: {self.output_dir}\n")

    def load_images(self, max_images: Optional[int] = None) -> List[Path]:
        """
        Load test images from Image-Bench DISTORTION_1K.

        Args:
            max_images: Optional limit for testing (default: None = all images)

        Returns:
            List of image file paths
        """
        print("=" * 70)
        print("📂 Loading test images")
        print("=" * 70)

        image_paths = sorted(self.dataset_path.glob('*.png'))

        # Apply max_images limit if specified
        if max_images is not None:
            image_paths = image_paths[:max_images]

        if not image_paths:
            raise FileNotFoundError(
                f"No PNG files found in {self.dataset_path}. "
                "Please download the Image-Bench DISTORTION_1K dataset and place the PNG images in this directory."
            )

        print(f"✓ Loaded {len(image_paths)} images from {self.dataset_path}\n")
        return image_paths

    def embed_watermarks(self, image_paths: List[Path]) -> Dict[str, Dict[str, Any]]:
        """
        Embed watermarks in all images and generate message_bits.

        Args:
            image_paths: List of paths to images

        Returns:
            dict: {
                image_name: {'original': PIL.Image, 'watermarked': PIL.Image, ...},
                '__message_bits__': np.ndarray  # Special key for message bits
            }
        """
        print("=" * 70)
        print("💧 Embedding watermarks")
        print("=" * 70)

        message = self.config['watermark']['message']
        watermarked_dir = self.output_dir / 'watermarked'
        watermarked_dir.mkdir(exist_ok=True)

        print(f"Message: '{message}'")
        print(f"Output: {watermarked_dir}\n")

        # Generate message_bits independently (not relying on video_watermark module)
        message_bits = string_to_bits(message, target_bits=256)
        print(f"✓ Generated message bits: {len(message_bits)} bits\n")

        results = {}
        iterator = tqdm(image_paths, desc="Embedding") if TQDM_AVAILABLE else image_paths

        for img_path in iterator:
            # Load original image
            img = Image.open(img_path)

            # Embed watermark
            wm_img = self.watermarker.embed_watermark(
                image_input=img,
                message=message
            )

            # Save watermarked image
            wm_path = watermarked_dir / img_path.name
            wm_img.save(wm_path)

            results[img_path.name] = {
                'original': img,
                'watermarked': wm_img,
                'watermarked_path': wm_path
            }

        # Save message_bits to results (shared by all images)
        # Use special key to avoid conflicts with image names
        results['__message_bits__'] = message_bits

        print(f"\n✓ Embedded watermarks in {len(results)-1} images\n")
        return results

    def apply_attacks(
        self,
        watermarked_images: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Dict[str, Image.Image]]]:
        """
        Apply all distortion attacks at configured strengths.

        Args:
            watermarked_images: Dict from embed_watermarks()

        Returns:
            dict: {attack_type: {strength: {image_name: PIL.Image}}}
        """
        print("=" * 70)
        print("⚔️  Applying distortion attacks")
        print("=" * 70)

        attacks = self.config['attacks']
        attacked_dir = self.output_dir / 'attacked'

        results = {}

        for attack_type in attacks['types']:
            print(f"\n🎯 Attack: {attack_type}")
            results[attack_type] = {}
            strengths = attacks['strengths'][attack_type]

            for strength in strengths:
                results[attack_type][strength] = {}

                # Create output directory
                attack_subdir = attacked_dir / attack_type / str(strength)
                attack_subdir.mkdir(parents=True, exist_ok=True)

                # Progress bar
                iterator = (tqdm(watermarked_images.items(),
                                desc=f"  Strength {strength}", leave=False)
                           if TQDM_AVAILABLE else watermarked_images.items())

                for img_name, img_data in iterator:
                    if img_name == '__message_bits__':
                        continue
                    wm_img = img_data['watermarked']

                    # Apply distortion
                    attacked_img = apply_single_distortion(
                        wm_img,
                        attack_type,
                        strength,
                        distortion_seed=0
                    )

                    # Save
                    attack_path = attack_subdir / img_name
                    attacked_img.save(attack_path)

                    results[attack_type][strength][img_name] = attacked_img

            print(f"  ✓ Completed {len(strengths)} strength levels")

        print(f"\n✓ Applied {len(attacks['types'])} attack types\n")
        return results

    def extract_watermarks(
        self,
        attacked_images: Dict[str, Dict[str, Dict[str, Image.Image]]]
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Extract watermarks from all attacked images.

        Args:
            attacked_images: Dict from apply_attacks()

        Returns:
            dict: {attack_type: {strength: [extraction_results]}}
        """
        print("=" * 70)
        print("🔍 Extracting watermarks")
        print("=" * 70)

        extract_params = self.config['watermark']['extract']
        print(f"Extract params: replicate={extract_params['replicate']}, "
              f"chunk_size={extract_params['chunk_size']}\n")

        results = {}

        for attack_type, strength_dict in attacked_images.items():
            print(f"\n🎯 Attack: {attack_type}")
            results[attack_type] = {}

            for strength, images in strength_dict.items():
                extraction_results = []

                iterator = (tqdm(images.items(),
                                desc=f"  Strength {strength}", leave=False)
                           if TQDM_AVAILABLE else images.items())

                for img_name, img in iterator:
                    try:
                        result = self.watermarker.extract_watermark(
                            image_input=img,
                            **extract_params
                        )
                        extraction_results.append(result)
                    except Exception as e:
                        print(f"\n⚠️  Extraction failed for {img_name}: {e}")
                        extraction_results.append({
                            'detected': False,
                            'confidence': 0.0,
                            'error': str(e)
                        })

                results[attack_type][strength] = extraction_results

            print(f"  ✓ Extracted from {len(strength_dict)} strength levels")

        print(f"\n✓ Extraction complete\n")
        return results

    def compute_metrics(
        self,
        watermarked_images: Dict[str, Dict[str, Any]],
        attacked_images: Dict[str, Dict[str, Dict[str, Image.Image]]],
        extraction_results: Dict[str, Dict[str, List[Dict[str, Any]]]]
    ) -> Dict[str, Any]:
        """
        Compute quality and detection metrics.

        Args:
            watermarked_images: Dict from embed_watermarks()
            attacked_images: Dict from apply_attacks()
            extraction_results: Dict from extract_watermarks()

        Returns:
            dict: Complete metrics structure
        """
        print("=" * 70)
        print("📊 Computing metrics")
        print("=" * 70)

        device = self.config['watermark'].get('device', 'cuda')

        # 1. Quality metrics (original vs watermarked)
        print("\n📈 Computing quality metrics (original vs watermarked)...")
        quality_list = []

        for img_name, img_data in watermarked_images.items():
            # Skip special key for message_bits
            if img_name == '__message_bits__':
                continue

            try:
                metrics = compute_quality_metrics(
                    img_data['original'],
                    img_data['watermarked'],
                    device=device
                )
                quality_list.append(metrics)
            except Exception as e:
                print(f"⚠️  Quality metric computation failed: {e}")

        if quality_list:
            avg_quality = {
                'psnr': sum(m['psnr'] for m in quality_list) / len(quality_list),
                'ssim': sum(m['ssim'] for m in quality_list) / len(quality_list),
                'lpips': sum(m['lpips'] for m in quality_list) / len(quality_list)
            }
            print(f"  PSNR: {avg_quality['psnr']:.2f} dB")
            print(f"  SSIM: {avg_quality['ssim']:.4f}")
            print(f"  LPIPS: {avg_quality['lpips']:.4f}")
        else:
            avg_quality = {'psnr': 0, 'ssim': 0, 'lpips': 0}

        # 2. Detection metrics per attack/strength (with Bit Accuracy)
        print("\n🎯 Computing detection metrics per attack...")
        robustness = {}

        # Get message_bits for bit accuracy computation
        message_bits = watermarked_images.get('__message_bits__')
        if message_bits is not None:
            print(f"✓ Using message bits: {len(message_bits)} bits\n")

        for attack_type, strength_dict in extraction_results.items():
            robustness[attack_type] = {}

            for strength, results in strength_dict.items():
                # Pass message_bits to compute bit accuracy
                det_metrics = compute_detection_metrics(
                    results,
                    message_bits=message_bits
                )
                robustness[attack_type][str(strength)] = det_metrics

                print(f"  {attack_type} @ {strength}: "
                      f"TPR={det_metrics['tpr']:.2%}, "
                      f"Bit Acc={det_metrics['bit_accuracy']:.2%}, "
                      f"Conf={det_metrics['avg_confidence']:.2f}")

        print("\n✓ Metrics computed\n")

        return {
            'quality_metrics': avg_quality,
            'robustness_by_attack': robustness
        }

    def save_results(self, metrics: Dict[str, Any]) -> None:
        """
        Save metrics to JSON.

        Args:
            metrics: Complete metrics dict from compute_metrics()
        """
        print("=" * 70)
        print("💾 Saving results")
        print("=" * 70)

        output_file = self.output_dir / 'metrics.json'

        # Add metadata
        results = {
            'metadata': {
                'algorithm': 'videoseal',
                'message': self.config['watermark']['message'],
                'num_images': self.config['dataset']['num_images'],
                'dataset': self.config['dataset']['name'],
                'timestamp': datetime.now().isoformat()
            },
            **metrics
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}\n")

    def run(self, max_images: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute complete benchmark pipeline.

        Args:
            max_images: Optional limit for testing (None = use config setting)

        Returns:
            dict: Complete evaluation metrics
        """
        print("\n" + "=" * 70)
        print("🚀 Starting Image-Bench evaluation")
        print("=" * 70 + "\n")

        start_time = datetime.now()

        try:
            # Check for max_images in config
            if max_images is None:
                max_images = self.config['dataset'].get('max_images')

            # 1. Load images
            image_paths = self.load_images(max_images)

            # 2. Embed watermarks
            watermarked = self.embed_watermarks(image_paths)

            # 3. Apply attacks
            attacked = self.apply_attacks(watermarked)

            # 4. Extract watermarks
            extractions = self.extract_watermarks(attacked)

            # 5. Compute metrics
            metrics = self.compute_metrics(watermarked, attacked, extractions)

            # 6. Save results
            self.save_results(metrics)

            elapsed = (datetime.now() - start_time).total_seconds()

            print("=" * 70)
            print("🎉 Benchmark complete!")
            print("=" * 70)
            print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
            print(f"📊 Results: {self.output_dir / 'metrics.json'}\n")

            return metrics

        except KeyboardInterrupt:
            print("\n\n⚠️  Benchmark interrupted by user")
            return {}
        except Exception as e:
            print(f"\n\n❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
