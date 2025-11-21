"""
Message Encoding Utilities for Image-Bench

Independent implementation of string-to-bits encoding for watermark messages.
Replicates VideoSeal's encoding logic to ensure consistency without depending
on the video_watermark module.

Key Design:
- UTF-8 encoding
- LSB-first bit ordering (matches VideoSeal)
- 256-bit target length (VideoSeal standard)
- Zero-padding for short messages
"""

import numpy as np
from typing import Union


def string_to_bits(message: str, target_bits: int = 256) -> np.ndarray:
    """
    Convert string to bit array using VideoSeal-compatible encoding.

    Implementation matches videoseal_wrapper._string_to_bits():
    1. UTF-8 encode to bytes
    2. Each byte → 8 bits (LSB-first ordering)
    3. Pad or truncate to target_bits length

    Args:
        message: Input string message
        target_bits: Target bit length (default 256, VideoSeal standard)

    Returns:
        np.ndarray: Bit array with shape (target_bits,), values 0 or 1

    Examples:
        >>> bits = string_to_bits("W-Bench-Test-2025")
        >>> bits.shape
        (256,)
        >>> bits[:8]  # First character 'W' bits
        array([1., 1., 1., 0., 1., 0., 1., 0.])  # 'W' = 87 = 0x57 (LSB-first)

        >>> # Verify encoding consistency
        >>> msg = "W-Bench-Test-2025"
        >>> bits = string_to_bits(msg)
        >>> decoded = bits_to_string(bits)
        >>> assert decoded == msg
    """
    # 1. UTF-8 encoding
    message_bytes = message.encode('utf-8')

    # 2. Convert to bit array (LSB-first, matches VideoSeal)
    bit_array = []
    for byte in message_bytes:
        for i in range(8):
            bit_array.append((byte >> i) & 1)

    # 3. Pad to minimum length (64 bits)
    min_bits = 64
    while len(bit_array) < min_bits:
        bit_array.append(0)

    # 4. Pad or truncate to target_bits
    if len(bit_array) > target_bits:
        bit_array = bit_array[:target_bits]
    else:
        while len(bit_array) < target_bits:
            bit_array.append(0)

    # 5. Convert to numpy array (float32 for compatibility)
    return np.array(bit_array, dtype=np.float32)


def bits_to_string(bits: Union[np.ndarray, list], encoding: str = 'utf-8') -> str:
    """
    Convert bit array back to string (for verification).

    Args:
        bits: Bit array or list
        encoding: Character encoding (default utf-8)

    Returns:
        str: Decoded string

    Examples:
        >>> bits = string_to_bits("Hello")
        >>> decoded = bits_to_string(bits)
        >>> decoded
        'Hello'
    """
    if isinstance(bits, np.ndarray):
        bits = bits.flatten()

    # Binarize
    bits_binary = (np.array(bits) > 0.5).astype(int)

    # Convert 8-bit groups to bytes
    byte_list = []
    for i in range(0, len(bits_binary), 8):
        if i + 8 > len(bits_binary):
            break

        # Reconstruct byte (LSB-first)
        byte_val = 0
        for j in range(8):
            if bits_binary[i + j]:
                byte_val |= (1 << j)

        byte_list.append(byte_val)

    # Convert to bytes and decode
    try:
        message_bytes = bytes(byte_list)
        # Remove padding zero bytes
        message_bytes = message_bytes.rstrip(b'\x00')
        return message_bytes.decode(encoding, errors='ignore')
    except:
        return ""


def verify_encoding_consistency(message: str) -> bool:
    """
    Verify encoding-decoding consistency.

    Tests that: string_to_bits(msg) → bits_to_string(bits) → original msg

    Args:
        message: Test message

    Returns:
        bool: True if encoding is consistent

    Examples:
        >>> verify_encoding_consistency("W-Bench-Test-2025")
        True
        >>> verify_encoding_consistency("测试中文123!@#")
        True
    """
    bits = string_to_bits(message)
    decoded = bits_to_string(bits)
    return decoded == message


def get_message_stats(message: str) -> dict:
    """
    Get statistics about encoded message.

    Args:
        message: Input message

    Returns:
        dict: {
            'original_length': int,
            'byte_length': int,
            'bit_length': int,
            'padded_bits': int,
            'encoding': str
        }

    Examples:
        >>> stats = get_message_stats("W-Bench-Test-2025")
        >>> stats['original_length']
        17
        >>> stats['bit_length']
        136
    """
    message_bytes = message.encode('utf-8')
    bits = string_to_bits(message, target_bits=256)

    # Find actual bit length (before padding)
    actual_bit_len = len(message_bytes) * 8

    return {
        'original_length': len(message),
        'byte_length': len(message_bytes),
        'bit_length': actual_bit_len,
        'padded_bits': 256 - actual_bit_len,
        'encoding': 'utf-8'
    }
