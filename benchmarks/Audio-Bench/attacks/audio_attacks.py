"""
Audio Attack Executor

All attack implementations are directly reused from AudioMarkBench.
Source: benchmarks/AudioMarkBench/no-box/nobox_audioseal_audiomarkdata.py
"""

import torch
import torchaudio
import librosa
import julius
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from typing import Union, Any, List
from audiomentations import Mp3Compression


class AudioAttackExecutor:
    """
    Audio attack executor for watermark robustness testing

    All attack functions are directly copied from AudioMarkBench implementation.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def apply_attack(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        attack_type: str,
        strength: Any
    ) -> None:
        """
        Apply a single audio attack

        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            attack_type: Type of attack (one of 9 attack types)
            strength: Attack strength parameter (type depends on attack_type)
        """
        # Load audio
        waveform, sr = torchaudio.load(str(input_path))

        # Apply attack based on type
        if attack_type == 'gaussian_noise':
            attacked = self.pert_gaussian_noise(waveform, snr_db=strength)
        elif attack_type == 'background_noise':
            attacked = self.pert_background_noise(waveform, snr_db=strength)
        elif attack_type == 'time_stretch':
            attacked, sr = self.pert_time_stretch(waveform, sr, speed_factor=strength)
        elif attack_type == 'quantization':
            attacked = self.pert_quantization(waveform, quantization_bit=strength)
        elif attack_type == 'lowpass_filter':
            attacked = self.pert_lowpass(waveform, cutoff_ratio=strength, sample_rate=sr)
        elif attack_type == 'highpass_filter':
            attacked = self.pert_highpass(waveform, cutoff_ratio=strength, sample_rate=sr)
        elif attack_type == 'smooth':
            attacked = self.pert_smooth(waveform, window_size=strength)
        elif attack_type == 'echo':
            duration, volume = strength  # Expect [duration, volume] pair
            attacked = self.pert_echo(waveform, volume=volume, duration=duration, sample_rate=sr)
        elif attack_type == 'mp3_compression':
            attacked = self.pert_mp3(waveform, bitrate=strength, sample_rate=sr)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Save attacked audio
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), attacked, sr)

    # ===== Attack implementations (directly from AudioMarkBench) =====

    @staticmethod
    def pert_gaussian_noise(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Add Gaussian white noise to achieve target SNR

        Source: AudioMarkBench/no-box/nobox_audioseal_audiomarkdata.py:1191-1205

        Args:
            waveform: Input audio waveform
            snr_db: Target signal-to-noise ratio in dB

        Returns:
            torch.Tensor: Noisy audio waveform
        """
        # Calculate signal power
        signal_power = torch.mean(waveform ** 2).to(device=waveform.device)

        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate noise with calculated noise power
        noise = torch.randn(waveform.size()) * torch.sqrt(noise_power)
        waveform_noisy = waveform + noise

        return waveform_noisy

    @staticmethod
    def pert_background_noise(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Add background noise to achieve target SNR

        Source: AudioMarkBench/no-box/nobox_audioseal_audiomarkdata.py:1207-1231

        Args:
            waveform: Input audio waveform
            snr_db: Target signal-to-noise ratio in dB

        Returns:
            torch.Tensor: Audio with background noise
        """
        try:
            from torchaudio.utils import download_asset
            SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
            noise, _ = torchaudio.load(SAMPLE_NOISE)
        except:
            # Fallback to Gaussian noise if download fails
            return AudioAttackExecutor.pert_gaussian_noise(waveform, snr_db)

        # Resize the noise to match the length of the waveform
        if noise.size(1) > waveform.size(1):
            noise = noise[:, :waveform.size(1)]
        else:
            repeat_times = waveform.size(1) // noise.size(1) + 1
            noise = noise.repeat(1, repeat_times)
            noise = noise[:, :waveform.size(1)]

        # Calculate signal power and noise power
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)

        # Calculate the scaling factor for the noise
        snr_linear = 10 ** (snr_db / 10)
        scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))

        # Scale and add the noise
        noisy_waveform = waveform + noise * scaling_factor
        return noisy_waveform

    @staticmethod
    def pert_time_stretch(
        waveform: torch.Tensor,
        sample_rate: int,
        speed_factor: float
    ) -> tuple:
        """
        Apply time stretching (speed change)

        Source: AudioMarkBench/no-box/nobox_audioseal_audiomarkdata.py:1177-1189

        Args:
            waveform: Input audio waveform
            sample_rate: Sample rate
            speed_factor: Speed factor (>1 faster, <1 slower)

        Returns:
            tuple: (stretched_waveform, sample_rate)
        """
        waveform_np = waveform.numpy()
        if waveform_np.shape[0] == 1:
            waveform_np = waveform_np.squeeze()

        # Use librosa for time stretching
        waveform_stretched = librosa.effects.time_stretch(waveform_np, rate=speed_factor)
        time_stretched_waveform = torch.from_numpy(waveform_stretched).unsqueeze(0).float()

        # Maintain consistent length
        if time_stretched_waveform.shape[1] < waveform.shape[1]:
            time_stretched_waveform = F.pad(
                time_stretched_waveform,
                (0, waveform.shape[1] - time_stretched_waveform.shape[1])
            )
        elif time_stretched_waveform.shape[1] > waveform.shape[1]:
            time_stretched_waveform = time_stretched_waveform[:, :waveform.shape[1]]

        return time_stretched_waveform, sample_rate

    @staticmethod
    def pert_quantization(waveform: torch.Tensor, quantization_bit: int) -> torch.Tensor:
        """
        Apply audio quantization (reduce bit depth)

        Source: AudioMarkBench/no-box/nobox_audioseal_audiomarkdata.py:1296-1307

        Args:
            waveform: Input audio waveform
            quantization_bit: Quantization level (e.g., 4, 8, 16, 32, 64)

        Returns:
            torch.Tensor: Quantized audio waveform
        """
        # Normalize the waveform to the range of the quantization levels
        min_val, max_val = waveform.min(), waveform.max()
        normalized_waveform = (waveform - min_val) / (max_val - min_val)

        # Quantize the normalized waveform
        quantized_waveform = torch.round(normalized_waveform * (quantization_bit - 1))

        # Rescale the quantized waveform back to the original range
        rescaled_waveform = (quantized_waveform / (quantization_bit - 1)) * (max_val - min_val) + min_val

        return rescaled_waveform

    @staticmethod
    def pert_lowpass(
        waveform: torch.Tensor,
        cutoff_ratio: float,
        sample_rate: int = 16000
    ) -> torch.Tensor:
        """
        Apply lowpass filter

        Source: AudioMarkBench/no-box/nobox_audioseal_audiomarkdata.py:1318-1340

        Args:
            waveform: Input audio waveform
            cutoff_ratio: Cutoff frequency ratio (0-1)
            sample_rate: Sample rate

        Returns:
            torch.Tensor: Filtered audio waveform
        """
        return julius.lowpass_filter(waveform, cutoff=cutoff_ratio)

    @staticmethod
    def pert_highpass(
        waveform: torch.Tensor,
        cutoff_ratio: float,
        sample_rate: int = 16000
    ) -> torch.Tensor:
        """
        Apply highpass filter

        Source: AudioMarkBench/no-box/nobox_audioseal_audiomarkdata.py:1318-1340

        Args:
            waveform: Input audio waveform
            cutoff_ratio: Cutoff frequency ratio (0-1)
            sample_rate: Sample rate

        Returns:
            torch.Tensor: Filtered audio waveform
        """
        return julius.highpass_filter(waveform, cutoff=cutoff_ratio)

    @staticmethod
    def pert_smooth(waveform: torch.Tensor, window_size: int = 5) -> torch.Tensor:
        """
        Apply smoothing using moving average

        Source: AudioMarkBench/no-box/nobox_audioseal_audiomarkdata.py:1342-1360

        Args:
            waveform: Input audio waveform
            window_size: Smoothing window size

        Returns:
            torch.Tensor: Smoothed audio waveform
        """
        waveform = waveform.unsqueeze(0)
        window_size = int(window_size)

        # Create a uniform smoothing kernel
        kernel = torch.ones(1, 1, window_size).type(waveform.type()) / window_size
        kernel = kernel.to(waveform.device)

        smoothed = julius.fft_conv1d(waveform, kernel)

        # Ensure tensor size is not changed
        tmp = torch.zeros_like(waveform)
        tmp[..., : smoothed.shape[-1]] = smoothed
        smoothed = tmp

        return smoothed.squeeze().unsqueeze(0)

    @staticmethod
    def pert_echo(
        tensor: torch.Tensor,
        volume: float = 0.4,
        duration: float = 0.1,
        sample_rate: int = 16000
    ) -> torch.Tensor:
        """
        Add echo effect

        Source: AudioMarkBench/no-box/nobox_audioseal_audiomarkdata.py:1362-1411

        Args:
            tensor: Input audio waveform [channels, frames]
            volume: Echo volume (0-1)
            duration: Delay time in seconds
            sample_rate: Sample rate

        Returns:
            torch.Tensor: Audio with echo effect
        """
        tensor = tensor.unsqueeze(0)

        duration = torch.Tensor([duration])
        volume = torch.Tensor([volume])
        n_samples = int(sample_rate * duration)
        impulse_response = torch.zeros(n_samples).type(tensor.type()).to(tensor.device)

        impulse_response[0] = 1.0  # Direct sound
        impulse_response[int(sample_rate * duration) - 1] = volume  # First reflection

        # Add batch and channel dimensions to the impulse response
        impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)

        # Convolve the audio signal with the impulse response
        reverbed_signal = julius.fft_conv1d(tensor, impulse_response)

        # Normalize to the original amplitude range
        reverbed_signal = (
            reverbed_signal
            / torch.max(torch.abs(reverbed_signal))
            * torch.max(torch.abs(tensor))
        )

        # Ensure tensor size is not changed
        tmp = torch.zeros_like(tensor)
        tmp[..., : reverbed_signal.shape[-1]] = reverbed_signal
        reverbed_signal = tmp
        reverbed_signal = reverbed_signal.squeeze(0)

        return reverbed_signal

    @staticmethod
    def pert_mp3(
        waveform: torch.Tensor,
        bitrate: int,
        sample_rate: int = 16000
    ) -> torch.Tensor:
        """
        Apply MP3 compression

        Source: AudioMarkBench/no-box/nobox_audioseal_audiomarkdata.py:1413-1422

        Args:
            waveform: Input audio waveform
            bitrate: MP3 bitrate in kbps (e.g., 64, 128, 256)
            sample_rate: Sample rate

        Returns:
            torch.Tensor: MP3 compressed audio
        """
        mp3_compressor = Mp3Compression(
            min_bitrate=bitrate,
            max_bitrate=bitrate,
            backend="pydub",  # Use pydub/ffmpeg backend
            p=1.0
        )
        waveform_np = waveform.detach().cpu().numpy()
        mp3_compressor.randomize_parameters(waveform_np, sample_rate)
        waveform_pert = mp3_compressor.apply(waveform_np, sample_rate)
        return torch.tensor(waveform_pert)
