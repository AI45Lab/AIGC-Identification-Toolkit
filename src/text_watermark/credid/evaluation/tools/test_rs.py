import reedsolo
from reedsolo import RSCodec

class BinaryCodec:
    def __init__(self, ecc_length):
        # Initialize Reed-Solomon encoder/decoder, ecc_length is the length of error correction code
        self.rs = RSCodec(ecc_length)

    def bin_list_to_bytearray(self, bin_list):
        byte_array = bytearray()
        # Ensure the length of the binary list is a multiple of 8
        while len(bin_list) % 8 != 0:
            bin_list.append(0)  # You can choose to pad with 0
        for i in range(0, len(bin_list), 8):
            byte = 0
            for bit in bin_list[i:i+8]:
                byte = (byte << 1) | bit
            byte_array.append(byte)
        return byte_array

    def bytearray_to_bin_list(self, byte_array):
        bin_list = []
        for byte in byte_array:
            # Get the binary representation of the current byte and ensure it is 8 bits long
            bin_list.extend([int(bit) for bit in '{:08b}'.format(byte)])
        return bin_list

    def encode(self, bin_list):
        # Convert to byte array
        byte_arr = self.bin_list_to_bytearray(bin_list)
        print("Encoded bytearray:", byte_arr)
        
        # Encode
        encoded = self.rs.encode(byte_arr)
        return encoded

    def decode(self, encoded):
        # Decode
        try:
            decoded, _, _ = self.rs.decode(encoded)
            # print("Decoded bytearray:", decoded)
            decoded_bin_list = self.bytearray_to_bin_list(decoded)
            return decoded_bin_list
        except reedsolo.ReedSolomonError as e:
            print("Decoding failed:", e)
            return None

# Example usage
codec = BinaryCodec(8)  # Use 8-byte error correction code
original_bin_list = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # Specified binary data

# Encode
encoded_data = codec.encode(original_bin_list)
print("Encoded data:", encoded_data)

# Decode
decoded_bin_list = codec.decode(encoded_data)
print("Original binary sequence:", original_bin_list)
print("Decoded binary sequence:", decoded_bin_list if decoded_bin_list is not None else "Decoding error")
