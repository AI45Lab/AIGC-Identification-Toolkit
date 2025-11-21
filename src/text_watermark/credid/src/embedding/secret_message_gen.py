import hashlib
import time

def secret_message_gen(model_version, extra_info, length):
    """
    Converts model version, current timestamp, and extra information into a binary sequence of specified length.

    Args:
    - model_version (str): The version information of the model.
    - extra_info (str): Additional information to be hashed.
    - length (int): Desired length of the binary sequence.

    Returns:
    - str: Binary sequence of specified length.
    """
    # Get the current timestamp
    timestamp = str(int(time.time()))

    # Concatenate the inputs
    input_string = model_version + timestamp + extra_info

    # Compute the SHA-256 hash of the concatenated string
    hash_object = hashlib.sha256(input_string.encode())
    hash_hex = hash_object.hexdigest()

    # Convert the hexadecimal hash to a binary sequence
    binary_sequence = bin(int(hash_hex, 16))[2:]

    # Ensure the binary sequence is at least as long as the desired length
    while len(binary_sequence) < length:
        binary_sequence += binary_sequence

    # Truncate the binary sequence to the desired length
    binary_sequence = binary_sequence[:length]

    return binary_sequence

def binary_to_decimal(binary_sequence):
    """
    Converts a binary sequence to a decimal number.

    Args:
    - binary_sequence (str): Binary sequence to be converted.

    Returns:
    - int: Decimal representation of the binary sequence.
    """
    return int(binary_sequence, 2)


# Example usage
model_version = "v1.2.3"
extra_info = "some extra info"
desired_length = 128  # Desired length of the binary sequence
binary_sequence = secret_message_gen(model_version, extra_info, desired_length)
print(binary_sequence)