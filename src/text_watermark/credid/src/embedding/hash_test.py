import torch

# Linear Congruential Generator (LCG)
def lcg(seed, a=1664525, c=1013904223, m=2**32):
    return (a * seed + c) % m

# First hash function version
def Hash1_v1(key):
    key = (~key) + (key << 21)
    key = key ^ (key >> 24)
    key = (key + (key << 3)) + (key << 8)
    key = key ^ (key >> 14)
    key = (key + (key << 2)) + (key << 4)
    key = key ^ (key >> 28)
    key = key + (key << 31)

    key = (key + (key << 3)) + (key << 8)
    key = key ^ (key >> 14)
    key = (key + (key << 2)) + (key << 4)
    key = key ^ (key >> 28)
    return key

def Hash1_v2(key):
    key = (~key) + (key << 21)
    key = key ^ (key >> 24)
    key = (key + (key << 3)) + (key << 8)
    key = key ^ (key >> 14)
    key = (key + (key << 2)) + (key << 4)
    key = key ^ (key >> 28)
    key = key + (key << 31)

    return key

# Generate random numbers and calculate collisions
def generate_hashes(hash_func, n):
    keys = torch.randint(0, 2**32, (n,), dtype=torch.int64)
    hashes = [hash_func(key.item()) for key in keys]
    unique_hashes = set(hashes)
    return len(hashes) - len(unique_hashes)

# Generate LCG sequence and calculate collisions
def generate_lcg_collisions(seed, n):
    sequence = []
    for _ in range(n):
        seed = lcg(seed)
        sequence.append(seed)
    unique_elements = set(sequence)
    return len(sequence) - len(unique_elements)

# Set parameters
n = 100000
seed = 1234

# Calculate collisions
collisions_lcg = generate_lcg_collisions(seed, n)
collisions_v1 = generate_hashes(Hash1_v1, n)
collisions_v2 = generate_hashes(Hash1_v2, n)

print(f"LCG collision count: {collisions_lcg}")
print(f"Hash1_v1 collision count: {collisions_v1}")
print(f"Hash1_v2 collision count: {collisions_v2}")
