"""
Audio Quality Metrics

All quality metric implementations are directly reused from AudioMarkBench.
Source: benchmarks/AudioMarkBench/no-box/nobox_audioseal_audiomarkdata.py
"""

import torch
import torchaudio
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path


def compute_snr(signal: torch.Tensor, noisy_signal: torch.Tensor) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR)

    Source: AudioMarkBench/no-box/nobox_audioseal_audiomarkdata.py:1155-1167

    Args:
        signal: Original audio signal (torch.Tensor)
        noisy_signal: Noisy/watermarked audio signal (torch.Tensor)

    Returns:
        float: SNR value in dB
    """
    # Compute the power of the original signal
    signal_power = torch.mean(signal ** 2)

    # Compute the power of the noise
    noise = noisy_signal - signal
    noise_power = torch.mean(noise ** 2)

    # Compute the Signal-to-Noise Ratio (SNR)
    snr = 10 * torch.log10(signal_power / noise_power)

    return snr.item()


def load_audio_tensor(audio_path: Union[str, Path]) -> torch.Tensor:
    """
    Load audio file as torch.Tensor

    Args:
        audio_path: Path to audio file

    Returns:
        torch.Tensor: Audio waveform
    """
    waveform, sr = torchaudio.load(str(audio_path))
    return waveform


def compute_audio_quality_metrics(
    watermarked_audios: Dict,
    original_audios: Optional[Dict] = None,
    enable_visqol: bool = False,
    enable_pesq: bool = False
) -> Dict[str, Any]:
    """
    Compute audio quality metrics for watermarked audios

    Based on AudioMarkBench implementation, focusing on SNR as the primary metric.
    ViSQOL and PESQ are optional and disabled by default.

    Args:
        watermarked_audios: Dictionary containing watermarked audio paths
        original_audios: Optional dictionary containing original audio paths
        enable_visqol: Whether to enable ViSQOL metric (requires visqol library)
        enable_pesq: Whether to enable PESQ metric (requires pypesq library)

    Returns:
        Dict containing quality metrics statistics
    """
    snr_values = []

    # Compute SNR for all watermarked audios
    for audio_name, audio_data in watermarked_audios.items():
        if audio_name == '__message_bits__':
            continue

        try:
            # Load original and watermarked audio
            original_path = audio_data['original_path']
            watermarked_path = audio_data['watermarked_path']

            original = load_audio_tensor(original_path)
            watermarked = load_audio_tensor(watermarked_path)

            # Ensure same shape
            min_len = min(original.shape[-1], watermarked.shape[-1])
            original = original[..., :min_len]
            watermarked = watermarked[..., :min_len]

            # Compute SNR
            snr = compute_snr(original, watermarked)
            snr_values.append(snr)

        except Exception as e:
            print(f"Warning: Failed to compute quality for {audio_name}: {e}")
            continue

    if not snr_values:
        return {
            'snr_mean': 0.0,
            'snr_std': 0.0,
            'snr_min': 0.0,
            'snr_max': 0.0,
            'num_audios': 0
        }

    results = {
        'snr_mean': float(np.mean(snr_values)),
        'snr_std': float(np.std(snr_values)),
        'snr_min': float(np.min(snr_values)),
        'snr_max': float(np.max(snr_values)),
        'num_audios': len(snr_values)
    }

    # Optional: ViSQOL metric
    if enable_visqol:
        try:
            from visqol import visqol_lib_py
            from visqol.pb2 import visqol_config_pb2

            # Initialize ViSQOL API
            config = visqol_config_pb2.VisqolConfig()
            config.audio.sample_rate = 16000
            config.options.use_speech_scoring = True

            import os
            svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
            config.options.svr_model_path = os.path.join(
                os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path
            )

            api = visqol_lib_py.VisqolApi()
            api.Create(config)

            # Compute ViSQOL scores
            visqol_scores = []
            for audio_name, audio_data in watermarked_audios.items():
                if audio_name == '__message_bits__':
                    continue

                original = load_audio_tensor(audio_data['original_path'])
                watermarked = load_audio_tensor(audio_data['watermarked_path'])

                score = api.Measure(
                    np.array(original.squeeze(), dtype=np.float64),
                    np.array(watermarked.squeeze(), dtype=np.float64)
                )
                visqol_scores.append(score.moslqo)

            results['visqol_mean'] = float(np.mean(visqol_scores))
            results['visqol_std'] = float(np.std(visqol_scores))

        except ImportError:
            print("Warning: ViSQOL library not installed, skipping ViSQOL metric")
        except Exception as e:
            print(f"Warning: ViSQOL computation failed: {e}")

    # Optional: PESQ metric
    if enable_pesq:
        try:
            from pypesq import pesq as pesq_val

            pesq_scores = []
            for audio_name, audio_data in watermarked_audios.items():
                if audio_name == '__message_bits__':
                    continue

                original = load_audio_tensor(audio_data['original_path'])
                watermarked = load_audio_tensor(audio_data['watermarked_path'])

                # PESQ requires 16kHz sampling rate
                pesq_score = pesq_val(
                    original.squeeze().numpy(),
                    watermarked.squeeze().numpy(),
                    fs=16000
                )
                pesq_scores.append(pesq_score)

            results['pesq_mean'] = float(np.mean(pesq_scores))
            results['pesq_std'] = float(np.std(pesq_scores))

        except ImportError:
            print("Warning: pypesq library not installed, skipping PESQ metric")
        except Exception as e:
            print(f"Warning: PESQ computation failed: {e}")

    return results
