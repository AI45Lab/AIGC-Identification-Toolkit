"""
Video Benchmark Runner for VideoSeal Robustness Evaluation

Six-stage evaluation pipeline:
    1. Load videos from dataset
    2. Embed watermarks using VideoSeal
    3. Apply attacks (8 types × multiple strengths)
    4. Extract watermarks from attacked videos
    5. Compute metrics (quality + detection)
    6. Save results (JSON + visualizations)

Architecture adapted from Audio-Bench with video-specific modifications.
"""

import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional

# Add parent directories to path
BENCHMARK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT.parent.parent))

# Import Video-Bench modules
from attacks.video_attacks import VideoAttackExecutor
from metrics.quality import compute_all_quality_metrics
from metrics.detection import compute_all_detection_metrics
from utils.video_io import load_dataset_videos, save_video_tensor, get_video_info, load_video_tensor

# Import VideoSeal wrapper
from src.video_watermark.videoseal_wrapper import VideoSealWrapper


class VideoBenchmarkRunner:
    """
    Video watermark robustness evaluation engine

    Evaluates VideoSeal watermark performance under various attacks
    following VideoMarkBench methodology.
    """

    def __init__(self, config_path: str, device: str = 'cuda', skip_to_extraction: bool = False):
        """
        Initialize benchmark runner

        Args:
            config_path: path to YAML configuration file
            device: computation device ('cuda' or 'cpu')
            skip_to_extraction: if True, skip stages 1-3 and start from stage 4 (extraction)
        """
        self.config = self._load_config(config_path)
        self.device = device
        self.skip_to_extraction = skip_to_extraction

        # Setup output directory
        self.output_dir = Path(self.config['output']['base_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.watermarked_dir = self.output_dir / 'watermarked'
        self.attacked_dir = self.output_dir / 'attacked'

        # If skipping to extraction, validate that necessary files exist
        if self.skip_to_extraction:
            self._validate_existing_data()
            # Initialize watermark for extraction only
            self.watermark = self._init_watermark()
        else:
            # Initialize all components for full pipeline
            self.watermark = self._init_watermark()
            self.attack_executor = VideoAttackExecutor()

        print(f"VideoBenchmarkRunner initialized")
        print(f"  Device: {self.device}")
        print(f"  Output: {self.output_dir}")
        if self.skip_to_extraction:
            print(f"  Mode: Resume from Stage 4 (Extraction)")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _init_watermark(self) -> VideoSealWrapper:
        """Initialize VideoSeal watermark wrapper"""
        watermark_config = self.config['watermark']

        wrapper = VideoSealWrapper(device=self.device)

        print(f"VideoSeal initialized (device: {self.device})")
        return wrapper

    def _validate_existing_data(self):
        """
        Validate that watermarked and attacked directories exist with MP4 files

        This is called when skip_to_extraction=True to ensure stages 1-3 have been completed.

        Raises:
            FileNotFoundError: if required directories or files are missing
        """
        print("\n" + "="*60)
        print("VALIDATING EXISTING DATA FOR STAGE 4 RESUME")
        print("="*60)

        # 1. Check watermarked directory exists
        if not self.watermarked_dir.exists():
            raise FileNotFoundError(
                f"\n❌ Watermarked directory not found: {self.watermarked_dir}\n"
                f"   Please run stages 1-3 first without --skip-to-extraction flag."
            )

        # 2. Check for MP4 files in watermarked directory
        watermarked_files = list(self.watermarked_dir.glob('*.mp4'))
        if not watermarked_files:
            raise FileNotFoundError(
                f"\n❌ No .mp4 files found in {self.watermarked_dir}\n"
                f"   Expected files: video_0000.mp4, video_0001.mp4, ...\n"
                f"   Please run stage 2 (embedding) first."
            )

        # 3. Check attacked directory exists
        if not self.attacked_dir.exists():
            raise FileNotFoundError(
                f"\n❌ Attacked directory not found: {self.attacked_dir}\n"
                f"   Please run stage 3 (attacks) first."
            )

        # 4. Check for attack subdirectories
        attack_subdirs = [d for d in self.attacked_dir.iterdir() if d.is_dir()]
        if not attack_subdirs:
            raise FileNotFoundError(
                f"\n❌ No attack subdirectories found in {self.attacked_dir}\n"
                f"   Expected subdirectories: gaussian_noise_0.01/, crop_0.9/, etc.\n"
                f"   Please run stage 3 (attacks) first."
            )

        # 5. Validate at least one attack directory has MP4 files
        has_valid_attack = False
        for subdir in attack_subdirs:
            attack_files = list(subdir.glob('*.mp4'))
            if attack_files:
                has_valid_attack = True
                break

        if not has_valid_attack:
            raise FileNotFoundError(
                f"\n❌ No .mp4 files found in any attack subdirectories\n"
                f"   Checked: {self.attacked_dir}\n"
                f"   Please ensure stage 3 (attacks) completed successfully."
            )

        # 6. Print validation summary
        total_attacks = len(attack_subdirs)
        print(f"✓ Found {len(watermarked_files)} watermarked videos (.mp4)")
        print(f"✓ Found {total_attacks} attack configurations:")
        for subdir in sorted(attack_subdirs)[:5]:  # Show first 5
            attack_files = list(subdir.glob('*.mp4'))
            print(f"    - {subdir.name}: {len(attack_files)} videos")
        if total_attacks > 5:
            print(f"    ... and {total_attacks - 5} more")
        print(f"✓ Validation passed! Ready to start from Stage 4 (Extraction)")
        print("="*60 + "\n")

    def _load_existing_videos(self, max_videos: Optional[int] = None) -> tuple:
        """
        Load existing watermarked and attacked video paths from disk

        This is called when skip_to_extraction=True to resume from stage 4.

        Args:
            max_videos: optional limit on number of videos to load

        Returns:
            tuple: (watermarked_paths, attacked_paths)
                - watermarked_paths: List[Path] of watermarked video files
                - attacked_paths: Dict[str, List[Path]] of {attack_name: [video_paths]}
        """
        print("Loading existing video paths from disk...")

        # 1. Load watermarked video paths (sorted by filename)
        all_watermarked_paths = sorted(
            self.watermarked_dir.glob('*.mp4'),
            key=lambda p: p.name
        )

        # Apply max_videos limit
        if max_videos is not None and max_videos < len(all_watermarked_paths):
            watermarked_paths = all_watermarked_paths[:max_videos]
            print(f"  ✓ Found {len(all_watermarked_paths)} watermarked videos (limiting to {max_videos})")
        else:
            watermarked_paths = all_watermarked_paths
            print(f"  ✓ Found {len(watermarked_paths)} watermarked videos")

        # 2. Scan all attack configuration directories
        attacked_paths = {}  # {attack_name: [video_paths]}

        attack_dirs = sorted([d for d in self.attacked_dir.iterdir() if d.is_dir()])

        for attack_dir in attack_dirs:
            attack_name = attack_dir.name
            all_video_paths = sorted(
                attack_dir.glob('*.mp4'),
                key=lambda p: p.name
            )

            # Apply max_videos limit to match watermarked videos
            if max_videos is not None and max_videos < len(all_video_paths):
                video_paths = all_video_paths[:max_videos]
            else:
                video_paths = all_video_paths

            if video_paths:
                attacked_paths[attack_name] = video_paths
                if max_videos is not None and max_videos < len(all_video_paths):
                    print(f"  ✓ Found {len(all_video_paths)} videos for attack '{attack_name}' (limiting to {max_videos})")
                else:
                    print(f"  ✓ Found {len(video_paths)} videos for attack '{attack_name}'")

        # 3. Data consistency check
        total_attacks = len(attacked_paths)
        if total_attacks == 0:
            raise ValueError("No attacked videos found in any subdirectory!")

        # Check if each attack has the same number of videos as watermarked
        expected_count = len(watermarked_paths)
        inconsistent_attacks = []

        for attack_name, paths in attacked_paths.items():
            if len(paths) != expected_count:
                inconsistent_attacks.append(f"{attack_name} ({len(paths)} videos)")

        if inconsistent_attacks:
            print(f"\n⚠ Warning: Some attacks have inconsistent video counts:")
            print(f"  Expected: {expected_count} videos (matching watermarked/)")
            for attack_info in inconsistent_attacks:
                print(f"  - {attack_info}")
            print(f"  Continuing with available videos...\n")

        print(f"✓ Loaded {len(watermarked_paths)} watermarked videos")
        print(f"✓ Loaded {total_attacks} attack configurations\n")

        return watermarked_paths, attacked_paths

    def run(self, max_videos: Optional[int] = None) -> Dict[str, Any]:
        """
        Run complete six-stage evaluation pipeline

        Note: Memory-optimized version that saves intermediate results as files
              and keeps only paths in memory to avoid OOM with large datasets.

        Args:
            max_videos: optional limit on number of videos to evaluate

        Returns:
            dict: Complete evaluation results
        """
        if self.skip_to_extraction:
            # ==================== RESUME FROM STAGE 4 ====================
            print("\n" + "="*60)
            print("RESUMING FROM STAGE 4 (EXTRACTION)")
            print("Skipping stages 1-3 (Load/Embed/Attack)")
            print("="*60 + "\n")

            # Load existing video paths from disk (with max_videos limit)
            watermarked_paths, attacked_paths = self._load_existing_videos(max_videos=max_videos)

            # Get watermark message from config
            watermark_message = self.config['watermark']['message']

            # Reload original videos for quality metric computation (Option B)
            print("Reloading original videos for quality metrics computation...")
            print("  (This may take some time but ensures complete evaluation)\n")
            video_paths, original_videos = self._load_videos(max_videos=len(watermarked_paths))

            # Stage 4: Extract watermarks
            print("\n[Stage 4/6] Extracting watermarks from attacked videos...")
            extractions = self._extract_watermarks(attacked_paths, watermark_message)

            # Stage 5: Compute metrics (with quality metrics)
            print("\n[Stage 5/6] Computing metrics...")
            metrics = self._compute_metrics(
                original_videos, watermarked_paths, attacked_paths, extractions
            )

            # Release original videos after quality metrics computation
            del original_videos
            import gc
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            print("  Released original videos from memory")

            # Stage 6: Save results
            print("\n[Stage 6/6] Saving results...")
            self._save_results(metrics, video_paths, watermark_message)

            print("\n" + "="*60)
            print("✓ EVALUATION COMPLETED SUCCESSFULLY (Resumed from Stage 4)")
            print("="*60)
            print(f"\nResults saved to: {self.output_dir}")
            print(f"  - metrics.json")
            print(f"  - summary.txt")
            print(f"  - config.yaml\n")

            return metrics

        else:
            # ==================== FULL PIPELINE (STAGES 1-6) ====================
            print("\n" + "="*60)
            print("VideoSeal Robustness Evaluation - Six-Stage Pipeline")
            print("="*60)

            # Stage 1: Load videos
            print("\n[Stage 1/6] Loading videos from dataset...")
            video_paths, original_videos = self._load_videos(max_videos)

            # Stage 2: Embed watermarks
            print("\n[Stage 2/6] Embedding watermarks...")
            watermarked_paths, watermark_message = self._embed_watermarks(original_videos)

            # Stage 3: Apply attacks
            print("\n[Stage 3/6] Applying attacks...")
            attacked_paths = self._apply_attacks(watermarked_paths)

            # Stage 4: Extract watermarks
            print("\n[Stage 4/6] Extracting watermarks...")
            extractions = self._extract_watermarks(attacked_paths, watermark_message)

            # Stage 5: Compute metrics
            print("\n[Stage 5/6] Computing metrics...")
            metrics = self._compute_metrics(
                original_videos, watermarked_paths, attacked_paths, extractions
            )

            # Release original videos after quality metrics computation
            del original_videos
            import gc
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            print("  Released original videos from memory")

            # Stage 6: Save results
            print("\n[Stage 6/6] Saving results...")
            self._save_results(metrics, video_paths, watermark_message)

            print("\n" + "="*60)
            print("Evaluation Complete!")
            print(f"Results saved to: {self.output_dir}")
            print("="*60)

            return metrics

    def _load_videos(self, max_videos: Optional[int]) -> tuple:
        """
        Stage 1: Load videos from dataset

        Note: Videos are loaded to CPU to avoid GPU OOM.
              They will be moved to GPU during processing as needed.

        Returns:
            tuple: (video_paths, video_tensors)
        """
        dataset_path = self.config['dataset']['path']
        max_videos = max_videos or self.config['dataset'].get('max_videos')

        # IMPORTANT: Always load to CPU to prevent GPU OOM with large datasets
        # Videos will be moved to GPU during watermarking/processing
        video_paths, video_tensors = load_dataset_videos(
            dataset_path,
            max_videos=max_videos,
            device='cpu'  # Force CPU loading
        )

        print(f"  Loaded {len(video_tensors)} videos to CPU")
        if len(video_tensors) > 0:
            print(f"  Sample video shape: {video_tensors[0].shape}")
            print(f"  Sample video device: {video_tensors[0].device}")

        return video_paths, video_tensors

    def _embed_watermarks(self, videos: List[torch.Tensor]) -> tuple:
        """
        Stage 2: Embed watermarks using VideoSeal

        Note: Videos are moved to GPU one at a time for processing,
              then saved to disk. Only paths are kept in memory to avoid accumulation.

        Returns:
            tuple: (watermarked_paths, watermark_message)
        """
        watermark_message = self.config['watermark']['message']
        watermarked_paths = []
        save_format = self.config['output'].get('save_format', 'mp4')

        for i, video in enumerate(tqdm(videos, desc="Embedding")):
            # Move video to GPU for processing
            video_gpu = video.to(self.device)

            # Embed watermark
            watermarked = self.watermark.embed_watermark(
                video_gpu,
                message=watermark_message,
                is_video=True,
                lowres_attenuation=self.config['watermark'].get('lowres_attenuation', True)
            )

            # Move result back to CPU to save GPU memory
            watermarked_cpu = watermarked.cpu()

            # Save watermarked video (always save, not conditional)
            save_path = self.output_dir / 'watermarked' / f'video_{i:04d}.{save_format}'
            save_path.parent.mkdir(exist_ok=True)
            save_video_tensor(watermarked_cpu, save_path)
            watermarked_paths.append(save_path)

            # Clear GPU cache and CPU tensors
            del video_gpu, watermarked, watermarked_cpu
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        print(f"  Embedded watermarks in {len(watermarked_paths)} videos")
        return watermarked_paths, watermark_message

    def _apply_attacks(self, watermarked_paths: List[Path]) -> Dict[str, List[Path]]:
        """
        Stage 3: Apply all configured attacks (OPTIMIZED)

        Performance optimization: Loop structure reversed to minimize disk I/O.
        - Before: Outer loop = attacks, Inner loop = videos → 3,900 disk reads
        - After:  Outer loop = videos, Inner loop = attacks → 100 disk reads (39× faster)

        Each video is loaded once, all attacks applied in memory, then released.

        Args:
            watermarked_paths: List of paths to watermarked videos

        Returns:
            dict: {attack_name: [attacked_video_paths]}
        """
        attack_configs = self._get_attack_configs()
        save_format = self.config['output'].get('save_format', 'mp4')

        # Initialize result dictionary with empty lists for each attack
        attacked_paths = {attack_name: [] for attack_name in attack_configs.keys()}

        # OPTIMIZED: Outer loop over videos (load once), inner loop over attacks
        for i, video_path in enumerate(tqdm(watermarked_paths, desc="Applying attacks")):
            # Load video from disk once
            video = load_video_tensor(video_path, device='cpu')
            video_gpu = video.to(self.device)

            # Apply all attacks to this video
            for attack_name, attack_config in attack_configs.items():
                # Apply attack (clone to avoid modifying original)
                attacked = self.attack_executor.apply_attack(
                    video_gpu.clone(),
                    attack_type=attack_config['type'],
                    strength=attack_config['strength']
                )

                # Move result back to CPU
                attacked_cpu = attacked.cpu() if attacked.is_cuda else attacked

                # Save attacked video
                save_path = self.output_dir / 'attacked' / attack_name / f'video_{i:04d}.{save_format}'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_video_tensor(attacked_cpu, save_path)
                attacked_paths[attack_name].append(save_path)

                # Clear attacked tensor
                del attacked, attacked_cpu
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

            # Clear video tensors after processing all attacks
            del video, video_gpu
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        print(f"  Applied {len(attacked_paths)} attack configurations to {len(watermarked_paths)} videos")
        return attacked_paths

    def _get_attack_configs(self) -> Dict[str, dict]:
        """Generate all attack configurations from config"""
        attack_configs = {}
        attacks = self.config['attacks']

        for attack_type in attacks['types']:
            strengths = attacks['strengths'][attack_type]

            for strength in strengths:
                attack_name = f"{attack_type}_{strength}"
                attack_configs[attack_name] = {
                    'type': attack_type,
                    'strength': strength
                }

        return attack_configs

    def _extract_watermarks(
        self,
        attacked_paths: Dict[str, List[Path]],
        original_message: str
    ) -> Dict[str, List[dict]]:
        """
        Stage 4: Extract watermarks from attacked videos

        Note: Videos are loaded from disk paths and moved to GPU for extraction.

        Args:
            attacked_paths: Dict of {attack_name: [attacked_video_paths]}

        Returns:
            dict: {attack_name: [extraction_results]}
        """
        extractions = {}
        chunk_size = self.config['watermark']['extract'].get('chunk_size', 16)

        for attack_name, video_paths in tqdm(attacked_paths.items(), desc="Extracting"):
            extraction_batch = []

            for video_path in video_paths:
                # Load video from disk
                video = load_video_tensor(video_path, device='cpu')

                # Move video to GPU for extraction
                video_gpu = video.to(self.device)

                # Extract watermark
                result = self.watermark.extract_watermark(
                    video_gpu,
                    is_video=True,
                    chunk_size=chunk_size
                )

                # Add ground truth for comparison
                result['original_message'] = original_message

                extraction_batch.append(result)

                # Clear tensors
                del video, video_gpu
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

            extractions[attack_name] = extraction_batch

        print(f"  Extracted watermarks from {len(extractions)} attack configurations")
        return extractions

    def _compute_metrics(
        self,
        original_videos: List[torch.Tensor],
        watermarked_paths: List[Path],
        attacked_paths: Dict[str, List[Path]],
        extractions: Dict[str, List[dict]]
    ) -> Dict[str, Any]:
        """
        Stage 5: Compute all quality and detection metrics

        Note: Watermarked videos are loaded from disk paths on-demand
              to avoid memory accumulation.

        Args:
            watermarked_paths: List of paths to watermarked videos
            attacked_paths: Dict of {attack_name: [attacked_video_paths]}

        Returns:
            dict: Complete metrics dictionary
        """
        metrics = {
            'quality': {},
            'detection': {},
            'per_attack': {}
        }

        # Compute quality metrics on watermarked vs original
        print("  Computing quality metrics...")

        # Optimization: Initialize LPIPS model once outside loop (avoids 50x re-initialization)
        from metrics.quality import init_lpips_model
        lpips_model = init_lpips_model(device=self.device)
        print("    LPIPS model initialized (will be reused for all videos)")

        quality_results = []
        for orig, wm_path in zip(original_videos, watermarked_paths):
            # Load watermarked video from disk
            wm = load_video_tensor(wm_path, device='cpu')
            q_metrics = compute_all_quality_metrics(wm, orig, device=self.device, lpips_model=lpips_model)
            quality_results.append(q_metrics)
            del wm  # Immediately release

        # Average quality metrics
        metrics['quality'] = {
            'psnr': np.mean([r['psnr'] for r in quality_results]),
            'ssim': np.mean([r['ssim'] for r in quality_results if r['ssim'] is not None]),
            'tLP': np.mean([r['tLP'] for r in quality_results])
        }

        # Compute detection metrics per attack
        print("  Computing detection metrics...")
        detection_config = self.config['watermark']['extract']

        for attack_name, extraction_batch in extractions.items():
            # Extract logits from VideoSeal outputs
            video_logits = []
            for result in extraction_batch:
                # Convert raw_preds to numpy if needed
                if 'raw_preds' in result:
                    logits = result['raw_preds']
                    if isinstance(logits, torch.Tensor):
                        logits = logits.cpu().numpy()

                    # FIX: VideoSeal returns 1D averaged logits, but detection metrics expect 2D
                    # Reshape to (1, n_bits) to add a pseudo-frame dimension
                    if logits.ndim == 1:
                        logits = logits.reshape(1, -1)

                    video_logits.append(logits)

            # Compute detection metrics
            if len(video_logits) > 0:
                attack_metrics = compute_all_detection_metrics(
                    video_logits,
                    detection_config,
                    watermark_message=self.config['watermark']['message']  # Pass actual watermark message
                )
                metrics['per_attack'][attack_name] = attack_metrics

        # Compute overall detection metrics
        all_logits = []
        for attack_name, extraction_batch in extractions.items():
            for result in extraction_batch:
                if 'raw_preds' in result:
                    logits = result['raw_preds']
                    if isinstance(logits, torch.Tensor):
                        logits = logits.cpu().numpy()

                    # FIX: VideoSeal returns 1D averaged logits, but detection metrics expect 2D
                    # Reshape to (1, n_bits) to add a pseudo-frame dimension
                    if logits.ndim == 1:
                        logits = logits.reshape(1, -1)

                    all_logits.append(logits)

        if len(all_logits) > 0:
            overall_detection = compute_all_detection_metrics(
                all_logits,
                detection_config,
                watermark_message=self.config['watermark']['message']  # Pass actual watermark message
            )
            metrics['detection'] = overall_detection

        return metrics

    def _save_results(
        self,
        metrics: Dict[str, Any],
        video_paths: List[Path],
        watermark_message: str
    ):
        """
        Stage 6: Save evaluation results

        Saves:
            - metrics.json: Complete metrics dictionary
            - summary.txt: Human-readable summary
            - config.yaml: Copy of configuration used
        """
        # Save metrics as JSON
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  Saved metrics to: {metrics_path}")

        # Save configuration
        config_path = self.output_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        print(f"  Saved config to: {config_path}")

        # Save summary
        summary_path = self.output_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write("VideoSeal Robustness Evaluation Summary\n")
            f.write("="*60 + "\n\n")

            f.write("Quality Metrics (Watermarked vs Original):\n")
            f.write(f"  PSNR:  {metrics['quality']['psnr']:.2f} dB\n")
            f.write(f"  SSIM:  {metrics['quality']['ssim']:.4f}\n")
            f.write(f"  tLP:   {metrics['quality']['tLP']:.6f}\n\n")

            f.write("Detection Metrics (Overall):\n")
            f.write(f"  FNR:           {metrics['detection']['fnr']:.4f}\n")
            f.write(f"  Bit Accuracy:  {metrics['detection']['bit_accuracy']:.4f}\n")
            f.write(f"  Avg Confidence: {metrics['detection']['avg_confidence']:.4f}\n\n")

            f.write(f"Dataset: {len(video_paths)} videos\n")
            f.write(f"Watermark: {watermark_message}\n")
            f.write(f"Attacks: {len(metrics['per_attack'])} configurations\n")

        print(f"  Saved summary to: {summary_path}")


if __name__ == "__main__":
    # Test with minimal configuration
    print("Testing VideoBenchmarkRunner...")

    # This would require actual VideoMarkBench dataset
    # For now, just initialize the runner
    config_path = Path(__file__).parent.parent / 'configs' / 'videoseal_robustness.yaml'

    if config_path.exists():
        runner = VideoBenchmarkRunner(str(config_path), device='cpu')
        print("✅ VideoBenchmarkRunner initialized successfully")
    else:
        print(f"⚠️  Config file not found: {config_path}")
        print("   Run this from the benchmark root directory")
