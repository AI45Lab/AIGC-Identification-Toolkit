"""
Message Encoding Utilities

Directly reuses the project's AudioSeal MessageEncoder implementation.
Source: src/audio_watermark/audioseal_wrapper.py
"""

import sys
import hashlib
import torch
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def string_to_bits_audio(message: str, nbits: int = 16) -> torch.Tensor:
    """
    Convert string message to 16-bit binary representation

    This function reuses the project's MessageEncoder implementation for consistency.
    Uses SHA256 hash to generate deterministic bit patterns from strings.

    Args:
        message: Input string message
        nbits: Number of bits (default 16 for AudioSeal)

    Returns:
        torch.Tensor: Binary tensor of shape (nbits,), values 0 or 1
    """
    try:
        # Try to import from project implementation
        from src.audio_watermark.audioseal_wrapper import MessageEncoder
        return MessageEncoder.string_to_bits(message, nbits)
    except ImportError:
        # Fallback: implement locally using same logic
        return _string_to_bits_fallback(message, nbits)


def _string_to_bits_fallback(message: str, nbits: int = 16) -> torch.Tensor:
    """
    Fallback implementation for string to bits conversion

    Uses SHA256 hash to generate deterministic bit patterns.

    Args:
        message: Input string message
        nbits: Number of bits

    Returns:
        torch.Tensor: Binary tensor of shape (nbits,), values 0 or 1
    """
    # SHA256 hash
    message_bytes = message.encode('utf-8')
    hash_object = hashlib.sha256(message_bytes)
    hash_bytes = hash_object.digest()

    # Convert to binary bits
    bits = []
    for byte in hash_bytes:
        for i in range(8):
            bits.append((byte >> (7-i)) & 1)

    # Truncate or tile to target nbits
    if len(bits) >= nbits:
        bits = bits[:nbits]
    else:
        while len(bits) < nbits:
            bits.extend(bits[:min(nbits - len(bits), len(bits))])

    return torch.tensor(bits, dtype=torch.int32).unsqueeze(0).squeeze(0)


def bits_to_string_audio(
    bits: torch.Tensor,
    original_messages: list = None
) -> str:
    """
    Convert 16-bit binary representation back to string message

    Attempts to match against original messages using Hamming distance.

    Args:
        bits: Binary tensor (nbits,)
        original_messages: List of original string messages to match against

    Returns:
        str: Matched message string or bit pattern string
    """
    try:
        # Try to import from project implementation
        from src.audio_watermark.audioseal_wrapper import MessageEncoder
        return MessageEncoder.bits_to_string(bits, original_messages)
    except ImportError:
        # Fallback: implement locally using same logic
        return _bits_to_string_fallback(bits, original_messages)


def _bits_to_string_fallback(
    bits: torch.Tensor,
    original_messages: list = None
) -> str:
    """
    Fallback implementation for bits to string conversion

    Args:
        bits: Binary tensor (nbits,)
        original_messages: List of original messages to match against

    Returns:
        str: Matched message or bit pattern string
    """
    if original_messages:
        best_match = None
        best_score = -1

        for msg in original_messages:
            encoded_bits = string_to_bits_audio(msg, len(bits.flatten()))
            matches = torch.sum(bits.flatten() == encoded_bits.flatten()).item()
            score = matches / len(bits.flatten())

            if score > best_score:
                best_score = score
                best_match = msg

        if best_score > 0.7:  # 70% match threshold
            return best_match

    # Fallback: return bit pattern string
    bit_string = ''.join(map(str, bits.flatten().cpu().numpy().tolist()))
    return f"bits_{bit_string[:16]}..."


def numpy_to_torch_bits(bits_np: np.ndarray) -> torch.Tensor:
    """
    Convert numpy bit array to torch tensor

    Args:
        bits_np: NumPy array of bits

    Returns:
        torch.Tensor: Torch tensor of bits
    """
    return torch.from_numpy(bits_np).float()


def torch_to_numpy_bits(bits_torch: torch.Tensor) -> np.ndarray:
    """
    Convert torch bit tensor to numpy array

    Args:
        bits_torch: Torch tensor of bits

    Returns:
        np.ndarray: NumPy array of bits
    """
    return bits_torch.cpu().numpy()


def encode_message(message: str, nbits: int = 16) -> int:
    """
    Encode string message to integer representation (for testing).

    Args:
        message: String message
        nbits: Number of bits (default 16)

    Returns:
        int: Integer representation of the message (0 to 2^nbits - 1)
    """
    bits_tensor = string_to_bits_audio(message, nbits)
    # Flatten to 1D if needed
    bits_flat = bits_tensor.flatten().cpu().numpy().tolist()

    # Convert binary list to integer
    result = 0
    for bit in bits_flat:
        result = (result << 1) | int(bit)

    return result


def decode_message(message_int: int, nbits: int = 16, original_messages: list = None) -> str:
    """
    Decode integer representation back to string message (for testing).

    Args:
        message_int: Integer representation
        nbits: Number of bits
        original_messages: List of original messages to match against

    Returns:
        str: Decoded message string
    """
    # Convert integer to binary tensor
    bits_list = []
    for i in range(nbits - 1, -1, -1):
        bits_list.append((message_int >> i) & 1)

    bits_tensor = torch.tensor(bits_list, dtype=torch.int32)

    # Convert bits to string
    return bits_to_string_audio(bits_tensor, original_messages)
