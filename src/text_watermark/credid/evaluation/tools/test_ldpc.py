import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message

# Define the original message
v = np.array([1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0])

# Parameter settings
n = 60  # Codeword length, try to choose a value that is divisible by d_c
d_v = 2  # Number of connections per check node
d_c = 5  # Number of connections per variable node
snr = 100  # Very high signal-to-noise ratio, simulating noiseless conditions

# Ensure that d_c divides n
if n % d_c != 0:
    n = (n // d_c + 1) * d_c  # Adjust n to be a multiple of d_c

# Generate LDPC matrix
H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
k = G.shape[1]  # Message length

# Ensure that the message length matches the input length
if len(v) > k:
    raise ValueError("Message length is greater than the allowed length of the generator matrix")
elif len(v) < k:
    # If v is shorter than k, zero padding is needed
    v_padded = np.zeros(k, dtype=int)
    v_padded[:len(v)] = v
    v = v_padded

# Encode the message
y = encode(G, v, snr)
print(y)
# Encoded sequence after being attacked
y_attacked = np.array([1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,1,0,1,1,1,1,1,0,0,0,0,1])

# Decode the message
d = decode(H, y_attacked, snr, maxiter=100)
v_decoded = get_message(G, d)

print("Original message:", v)
print("Encoded sequence after being attacked:", y_attacked)
print("Decoded message:", v_decoded)
print("Decoding success:", np.array_equal(v, v_decoded))
