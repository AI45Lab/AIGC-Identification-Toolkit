class ECC:
    def __init__(self):
        pass

    def encode_block(self, data_bits):
        """Encodes a 4-bit data block into a 7-bit Hamming code."""
        if len(data_bits) != 4:
            raise ValueError("Data bits should be a list of 4 bits.")

        # Initialize the 7-bit code with data bits and placeholders for parity bits
        code = [0] * 7
        code[2], code[4], code[5], code[6] = data_bits

        # Calculate parity bits
        code[0] = code[2] ^ code[4] ^ code[6]  # p1
        code[1] = code[2] ^ code[5] ^ code[6]  # p2
        code[3] = code[4] ^ code[5] ^ code[6]  # p4

        return code

    def decode_block(self, encoded_bits):
        """Decodes a 7-bit Hamming code block into 4-bit data, correcting a single-bit error if present."""
        if len(encoded_bits) != 7:
            raise ValueError("Encoded bits should be a list of 7 bits.")

        # Calculate parity check bits
        p1 = encoded_bits[0] ^ encoded_bits[2] ^ encoded_bits[4] ^ encoded_bits[6]
        p2 = encoded_bits[1] ^ encoded_bits[2] ^ encoded_bits[5] ^ encoded_bits[6]
        p4 = encoded_bits[3] ^ encoded_bits[4] ^ encoded_bits[5] ^ encoded_bits[6]

        # Calculate the error position
        error_position = p1 + (p2 << 1) + (p4 << 2)

        if error_position != 0:
            print(f"Error detected at position: {error_position}")
            # Correct the error
            encoded_bits[error_position - 1] ^= 1

        # Extract the original data bits
        data_bits = [encoded_bits[2], encoded_bits[4], encoded_bits[5], encoded_bits[6]]

        return data_bits

    def encode(self, data_bits):
        """Encodes a list of data bits into a list of encoded bits using Hamming code."""
        # Padding data to make its length a multiple of 4
        padding_length = (4 - len(data_bits) % 4) % 4
        padded_data_bits = data_bits + [0] * padding_length

        encoded_bits = []
        for i in range(0, len(padded_data_bits), 4):
            block = padded_data_bits[i:i+4]
            encoded_bits.extend(self.encode_block(block))
        
        return encoded_bits, padding_length

    def decode(self, encoded_bits, padding_length):
        """Decodes a list of encoded bits into a list of data bits, correcting single-bit errors if present."""
        if len(encoded_bits) % 7 != 0:
            raise ValueError("Encoded bits length should be a multiple of 7.")

        data_bits = []
        for i in range(0, len(encoded_bits), 7):
            block = encoded_bits[i:i+7]
            data_bits.extend(self.decode_block(block))
        
        # Remove padding
        if padding_length > 0:
            data_bits = data_bits[:-padding_length]
        
        return data_bits

# # Test the extended ECC class
# ecc = ECC()

# # Test data
# data_bits = [1, 0, 1, 1, 0, 1, 0, 0, 1]  # 9 bits

# # Encode
# encoded_bits, padding_length = ecc.encode(data_bits)
# print(f"Encoded bits: {encoded_bits}")

# # Introduce an error
# encoded_bits_with_error = encoded_bits.copy()
# encoded_bits_with_error[3] ^= 1  # Introduce an error at position 4
# print(f"Encoded bits with error: {encoded_bits_with_error}")

# # Decode
# decoded_bits = ecc.decode(encoded_bits_with_error, padding_length)
# print(f"Decoded bits: {decoded_bits}")
