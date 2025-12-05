import os
import math

def binary_to_signed_int(binary_str, bits=16):
    val = int(binary_str, 2)
    if val & (1 << (bits - 1)):
        val = val - (1 << bits)
    return val

def signed_int_to_binary(val, bits=16):
    if val < 0:
        val_unsigned = (1 << bits) + val
    else:
        val_unsigned = val
    return format(val_unsigned & ((1 << bits) - 1), f"0{bits}b")

def elu(x, alpha=1.0):
    if x > 0:
        return x
    else:
        return alpha * (math.exp(x) - 1.0)

def leaky_relu(x, alpha=0.01):
    if x > 0:
        return x
    else:
        return alpha * x

def quantize_to_int16(val):
    rounded = round(val)
    clamped = max(-32768, min(32767, rounded))
    return int(clamped)

# Test with first data line
with open("out_raw.txt", "r") as f:
    lines = f.readlines()

line = lines[3].strip()  # First data line
print(f"Input line length: {len(line)}")

# Process first channel
channel_binary = line[0:16]
val_int = binary_to_signed_int(channel_binary, bits=16)
val_float = float(val_int)
elu_val = elu(val_float, alpha=1.0)
elu_int = quantize_to_int16(elu_val)
elu_binary = signed_int_to_binary(elu_int, bits=16)

print(f"Channel 0: {val_int} -> ELU: {elu_val:.2f} -> {elu_int} -> {elu_binary}")

# Check if files exist
print(f"out_elu.txt exists: {os.path.exists('out_elu.txt')}")
print(f"out_leaky.txt exists: {os.path.exists('out_leaky.txt')}")

