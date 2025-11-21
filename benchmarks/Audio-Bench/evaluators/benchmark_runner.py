"""
Audio-Bench Benchmark Runner for AudioSeal

Orchestrates the complete evaluation pipeline:
1. Load test audios
2. Embed watermarks
3. Apply audio attacks
4. Extract watermarks
5. Compute metrics (quality and detection)
6. Save results

Architecture: Completely replicates Image-Bench pipeline for audio watermarking.
"""

import json
import yaml
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add paths for imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not available, progress bars disabled")

# Import project AudioWatermark
from src.audio_watermark.audio_watermark import AudioWatermark

# Import local modules
audio_bench_dir = Path(__file__).resolve().parent.parent

# Import attack executor
sys.path.insert(0, str(audio_bench_dir))
from attacks.audio_attacks import AudioAttackExecutor

# Import metrics
from metrics.quality import compute_audio_quality_metrics
from metrics.detection import compute_audio_detection_metrics, aggregate_detection_metrics_by_attack

# Import utilities
from utils.message_encoding import string_to_bits_audio
from utils.plot_radar import generate_all_radars


class AudioBenchmarkRunner:
    """
    Evaluates AudioSeal watermark robustness against audio attacks.

    Workflow (6 stages):
        1. Load audios from dataset
        2. Embed watermarks using AudioWatermark (AudioSeal backend)
        3. Apply 9 attack types at various strengths
        4. Extract watermarks from attacked audios
        5. Compute quality metrics (SNR) and detection metrics (TPR, BA, confidence)
        6. Save results to JSON and generate radar charts
    """

    def __init__(self, config_path: str):
        """
        Initialize runner with YAML config.

        Args:
            config_path: Path to audioseal_robustness.yaml
        """
        print("=" * 70)
        print("🚀 Audio-Bench AudioSeal Robustness Evaluation")
        print("=" * 70)
        print(f"\n📂 Loading config from: {config_path}\n")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize AudioWatermark
        print("🔧 Initializing AudioSeal watermarker...")
        self.watermarker = AudioWatermark()

        # Initialize attack executor
        print("🔧 Initializing audio attack executor...")
        self.attack_executor = AudioAttackExecutor(sample_rate=16000)

        # Setup paths
        self.dataset_path = Path(self.config['dataset']['path'])
        self.output_dir = Path(self.config['output']['base_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"✓ Dataset path: {self.dataset_path}")
        print(f"✓ Output directory: {self.output_dir}\n")

    def load_audios(self, max_audios: Optional[int] = None) -> List[Path]:
        """
        Load test audios from dataset.

        Args:
            max_audios: Optional limit for testing (default: None = all audios)

        Returns:
            List of audio file paths
        """
        print("=" * 70)
        print("📂 Loading test audios")
        print("=" * 70)

        # Support both WAV and MP3 formats
        audio_paths = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
            audio_paths.extend(self.dataset_path.glob(ext))

        audio_paths = sorted(audio_paths)

        # Apply max_audios limit if specified
        if max_audios is not None:
            audio_paths = audio_paths[:max_audios]

        if not audio_paths:
            raise FileNotFoundError(
                f"No audio files found in {self.dataset_path}. "
                "Please place audio files (.wav, .mp3, .flac, .m4a) in this directory."
            )

        print(f"✓ Loaded {len(audio_paths)} audios from {self.dataset_path}\n")
        return audio_paths

    def embed_watermarks(self, audio_paths: List[Path]) -> Dict[str, Dict[str, Any]]:
        """
        Embed watermarks in all audios and generate message_bits.

        Args:
            audio_paths: List of paths to audios

        Returns:
            dict: {
                audio_name: {'original_path': Path, 'watermarked_path': Path, ...},
                '__message_bits__': torch.Tensor  # Special key for 16-bit message
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

        # Generate message_bits independently (16-bit for AudioSeal)
        message_bits = string_to_bits_audio(message, nbits=16)
        print(f"✓ Generated message bits: {len(message_bits)} bits (AudioSeal 16-bit)\n")

        results = {}
        iterator = tqdm(audio_paths, desc="Embedding") if TQDM_AVAILABLE else audio_paths

        for audio_path in iterator:
            # Embed watermark (AudioWatermark handles loading and saving)
            wm_path = watermarked_dir / audio_path.name

            try:
                self.watermarker.embed_watermark(
                    audio_input=str(audio_path),
                    message=message,
                    output_path=str(wm_path)
                )

                results[audio_path.name] = {
                    'original_path': audio_path,
                    'watermarked_path': wm_path
                }
            except Exception as e:
                print(f"\n⚠️  Embedding failed for {audio_path.name}: {e}")
                continue

        # Save message_bits to results (shared by all audios)
        results['__message_bits__'] = message_bits

        print(f"\n✓ Embedded watermarks in {len(results)-1} audios\n")
        return results

    def apply_attacks(
        self,
        watermarked_audios: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[Any, Dict[str, Path]]]:
        """
        Apply all audio attacks at configured strengths.

        Args:
            watermarked_audios: Dict from embed_watermarks()

        Returns:
            dict: {attack_type: {strength: {audio_name: Path}}}
        """
        print("=" * 70)
        print("⚔️  Applying audio attacks")
        print("=" * 70)

        attacks = self.config['attacks']
        attacked_dir = self.output_dir / 'attacked'

        results = {}

        for attack_type in attacks['types']:
            print(f"\n🎯 Attack: {attack_type}")
            results[attack_type] = {}
            strengths = attacks['strengths'][attack_type]

            for strength in strengths:
                # Convert list to tuple for hashable key (e.g., echo attack)
                strength_key = tuple(strength) if isinstance(strength, list) else strength
                results[attack_type][strength_key] = {}

                # Create output directory (use string representation for path)
                strength_str = '_'.join(map(str, strength)) if isinstance(strength, list) else str(strength)
                attack_subdir = attacked_dir / attack_type / strength_str
                attack_subdir.mkdir(parents=True, exist_ok=True)

                # Check if this attack+strength is already completed
                total_audios = len([k for k in watermarked_audios.keys() if k != '__message_bits__'])
                existing_files = list(attack_subdir.glob('*'))

                if len(existing_files) >= total_audios:
                    print(f"  ⏭️  Strength {strength_str} already completed, skipping...")
                    # Load existing results
                    for audio_name in watermarked_audios.keys():
                        if audio_name == '__message_bits__':
                            continue
                        attack_path = attack_subdir / audio_name
                        if attack_path.exists():
                            results[attack_type][strength_key][audio_name] = attack_path
                    continue

                # Progress bar
                iterator = (tqdm(watermarked_audios.items(),
                                desc=f"  Strength {strength_str}", leave=False)
                           if TQDM_AVAILABLE else watermarked_audios.items())

                for audio_name, audio_data in iterator:
                    if audio_name == '__message_bits__':
                        continue

                    wm_path = audio_data['watermarked_path']
                    attack_path = attack_subdir / audio_name

                    # Skip if already processed (resume support)
                    if attack_path.exists():
                        results[attack_type][strength_key][audio_name] = attack_path
                        continue

                    # Apply attack
                    try:
                        self.attack_executor.apply_attack(
                            input_path=wm_path,
                            output_path=attack_path,
                            attack_type=attack_type,
                            strength=strength  # Pass original strength (list or scalar)
                        )

                        results[attack_type][strength_key][audio_name] = attack_path

                    except Exception as e:
                        print(f"\n⚠️  Attack failed for {audio_name}: {e}")
                        continue

            print(f"  ✓ Completed {len(strengths)} strength levels")

        print(f"\n✓ Applied {len(attacks['types'])} attack types\n")
        return results

    def extract_watermarks(
        self,
        attacked_audios: Dict[str, Dict[Any, Dict[str, Path]]]
    ) -> Dict[str, Dict[Any, List[Dict[str, Any]]]]:
        """
        Extract watermarks from all attacked audios.

        Args:
            attacked_audios: Dict from apply_attacks()

        Returns:
            dict: {attack_type: {strength: [extraction_results]}}
        """
        print("=" * 70)
        print("🔍 Extracting watermarks")
        print("=" * 70)

        extract_params = self.config['watermark']['extract']
        print(f"Extract params: tau_prob={extract_params['tau_prob']}, "
              f"tau_ba={extract_params['tau_ba']}\n")

        results = {}

        for attack_type, strength_dict in attacked_audios.items():
            print(f"\n🎯 Attack: {attack_type}")
            results[attack_type] = {}

            for strength, audios in strength_dict.items():
                extraction_results = []

                iterator = (tqdm(audios.items(),
                                desc=f"  Strength {strength}", leave=False)
                           if TQDM_AVAILABLE else audios.items())

                for audio_name, audio_path in iterator:
                    try:
                        result = self.watermarker.extract_watermark(
                            audio_input=str(audio_path),
                            detection_threshold=extract_params['detection_threshold'],
                            message_threshold=extract_params['message_threshold']
                        )
                        extraction_results.append(result)
                    except Exception as e:
                        print(f"\n⚠️  Extraction failed for {audio_name}: {e}")
                        extraction_results.append({
                            'detected': False,
                            'confidence': 0.0,
                            'detection_prob': 0.0,
                            'bit_accuracy': 0.0,
                            'error': str(e)
                        })

                results[attack_type][strength] = extraction_results

            print(f"  ✓ Extracted from {len(strength_dict)} strength levels")

        print(f"\n✓ Extraction complete\n")
        return results

    def compute_metrics(
        self,
        watermarked_audios: Dict[str, Dict[str, Any]],
        attacked_audios: Dict[str, Dict[Any, Dict[str, Path]]],
        extraction_results: Dict[str, Dict[Any, List[Dict[str, Any]]]]
    ) -> Dict[str, Any]:
        """
        Compute quality and detection metrics.

        Args:
            watermarked_audios: Dict from embed_watermarks()
            attacked_audios: Dict from apply_attacks()
            extraction_results: Dict from extract_watermarks()

        Returns:
            Aggregated metrics dictionary
        """
        print("=" * 70)
        print("📊 Computing metrics")
        print("=" * 70)

        # 1. Quality metrics (SNR)
        print("\n📈 Computing quality metrics (SNR)...")
        quality_config = self.config.get('quality_metrics', {})
        quality_metrics = compute_audio_quality_metrics(
            watermarked_audios,
            enable_visqol=quality_config.get('enable_visqol', False),
            enable_pesq=quality_config.get('enable_pesq', False)
        )

        print(f"  ✓ SNR mean: {quality_metrics['snr_mean']:.2f} dB")
        print(f"  ✓ SNR std:  {quality_metrics['snr_std']:.2f} dB")

        # 2. Detection metrics (TPR, Bit Accuracy, Confidence)
        print("\n🎯 Computing detection metrics...")
        message_bits = watermarked_audios['__message_bits__']
        tau_prob = self.config['watermark']['extract']['tau_prob']
        tau_ba = self.config['watermark']['extract']['tau_ba']

        robustness = aggregate_detection_metrics_by_attack(
            extraction_results,
            original_msg_bits=message_bits.numpy() if isinstance(message_bits, torch.Tensor) else message_bits,
            tau_prob=tau_prob,
            tau_ba=tau_ba
        )

        # Print summary
        print("\n📋 Detection Summary:")
        for attack_type, strength_dict in robustness.items():
            print(f"\n  {attack_type}:")
            for strength, metrics in strength_dict.items():
                print(f"    Strength {strength}: "
                      f"TPR(prob)={metrics['tpr_prob']:.2%}, "
                      f"TPR(BA)={metrics['tpr_ba']:.2%}, "
                      f"Avg BA={metrics['avg_bit_accuracy']:.4f}")

        return {
            'quality_metrics': quality_metrics,
            'robustness_by_attack': robustness
        }

    def save_results(self, metrics: Dict[str, Any]) -> None:
        """
        Save results to JSON and generate radar charts.

        Args:
            metrics: Metrics dictionary from compute_metrics()
        """
        print("\n" + "=" * 70)
        print("💾 Saving results")
        print("=" * 70)

        # Save metrics JSON
        output_file = self.output_dir / 'metrics.json'
        results = {
            'metadata': {
                'algorithm': 'audioseal',
                'message': self.config['watermark']['message'],
                'message_bits': 16,
                'sample_rate': 16000,
                'tau_prob': self.config['watermark']['extract']['tau_prob'],
                'tau_ba': self.config['watermark']['extract']['tau_ba'],
                'num_audios': self.config['dataset'].get('num_audios', 'unknown'),
                'timestamp': datetime.now().isoformat()
            },
            **metrics
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Metrics saved to: {output_file}")

        # Generate radar charts
        print("\n🎨 Generating radar charts...")
        try:
            generate_all_radars(
                metrics_path=str(output_file),
                output_dir=str(self.output_dir),
                color_scheme='blue',
                dpi=300
            )
        except Exception as e:
            print(f"\n⚠️  Radar chart generation failed: {e}")
            import traceback
            traceback.print_exc()

    def run(self, max_audios: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute complete evaluation pipeline (6 stages).

        Args:
            max_audios: Optional limit for testing

        Returns:
            Final metrics dictionary
        """
        start_time = datetime.now()

        try:
            # Check for config override
            if max_audios is None:
                max_audios = self.config['dataset'].get('max_audios')

            # Stage 1: Load audios
            audio_paths = self.load_audios(max_audios)

            # Stage 2: Embed watermarks
            watermarked = self.embed_watermarks(audio_paths)

            # Stage 3: Apply attacks
            attacked = self.apply_attacks(watermarked)

            # Stage 4: Extract watermarks
            extractions = self.extract_watermarks(attacked)

            # Stage 5: Compute metrics
            metrics = self.compute_metrics(watermarked, attacked, extractions)

            # Stage 6: Save results
            self.save_results(metrics)

            elapsed = (datetime.now() - start_time).total_seconds()
            print("\n" + "=" * 70)
            print("✅ Audio-Bench evaluation completed successfully!")
            print("=" * 70)
            print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
            print(f"📊 Results: {self.output_dir / 'metrics.json'}")

            return metrics

        except Exception as e:
            print("\n" + "=" * 70)
            print("❌ Benchmark failed!")
            print("=" * 70)
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            return {}
