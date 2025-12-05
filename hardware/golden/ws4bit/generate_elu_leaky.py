#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import os

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

def gelu(x):
    """
    GELU (Gaussian Error Linear Unit) activation function.
    
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    Args:
        x: Input value
    
    Returns:
        GELU output value
    """
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)))

def quantize_to_int16(val):
    rounded = round(val)
    clamped = max(-32768, min(32767, rounded))
    return int(clamped)

# Create viz directory if it doesn't exist
viz_dir = "viz"
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)

# Read input file
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "out_raw.txt")

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

output_elu_lines = []
output_leaky_lines = []
output_gelu_lines = []
viz_elu_lines = []
viz_leaky_lines = []
viz_gelu_lines = []

for line in lines:
    line = line.strip()
    
    if line.startswith("#") or line == "":
        output_elu_lines.append(line)
        output_leaky_lines.append(line)
        output_gelu_lines.append(line)
        viz_elu_lines.append(line)
        viz_leaky_lines.append(line)
        viz_gelu_lines.append(line)
        continue
    
    if len(line) != 128:
        output_elu_lines.append(line)
        output_leaky_lines.append(line)
        output_gelu_lines.append(line)
        viz_elu_lines.append(line)
        viz_leaky_lines.append(line)
        viz_gelu_lines.append(line)
        continue
    
    binary_str_elu = ""
    binary_str_leaky = ""
    binary_str_gelu = ""
    viz_values_elu = []
    viz_values_leaky = []
    viz_values_gelu = []
    
    for ch in range(8):
        start_idx = ch * 16
        end_idx = start_idx + 16
        channel_binary = line[start_idx:end_idx]
        
        val_int = binary_to_signed_int(channel_binary, bits=16)
        val_float = float(val_int)
        
        # ELU
        elu_val = elu(val_float, alpha=1.0)
        elu_int = quantize_to_int16(elu_val)
        elu_binary = signed_int_to_binary(elu_int, bits=16)
        binary_str_elu += elu_binary
        viz_values_elu.append(elu_int)
        
        # LeakyReLU
        leaky_val = leaky_relu(val_float, alpha=0.01)
        leaky_int = quantize_to_int16(leaky_val)
        leaky_binary = signed_int_to_binary(leaky_int, bits=16)
        binary_str_leaky += leaky_binary
        viz_values_leaky.append(leaky_int)
        
        # GELU
        gelu_val = gelu(val_float)
        gelu_int = quantize_to_int16(gelu_val)
        gelu_binary = signed_int_to_binary(gelu_int, bits=16)
        binary_str_gelu += gelu_binary
        viz_values_gelu.append(gelu_int)
    
    output_elu_lines.append(binary_str_elu)
    output_leaky_lines.append(binary_str_leaky)
    output_gelu_lines.append(binary_str_gelu)
    
    # Format viz line: 8 values with spacing (format: "   -20      8      4     27    -57    -65    -50    -58")
    viz_elu_line = "".join([f"{val:7d} " for val in viz_values_elu]).rstrip()
    viz_leaky_line = "".join([f"{val:7d} " for val in viz_values_leaky]).rstrip()
    viz_gelu_line = "".join([f"{val:7d} " for val in viz_values_gelu]).rstrip()
    viz_elu_lines.append(viz_elu_line)
    viz_leaky_lines.append(viz_leaky_line)
    viz_gelu_lines.append(viz_gelu_line)

# Write binary format files
output_elu_path = os.path.join(script_dir, "out_elu.txt")
output_leaky_path = os.path.join(script_dir, "out_leaky.txt")
output_gelu_path = os.path.join(script_dir, "out_gelu.txt")

with open(output_elu_path, "w", encoding="utf-8") as f:
    f.write("\n".join(output_elu_lines) + "\n")

with open(output_leaky_path, "w", encoding="utf-8") as f:
    f.write("\n".join(output_leaky_lines) + "\n")

with open(output_gelu_path, "w", encoding="utf-8") as f:
    f.write("\n".join(output_gelu_lines) + "\n")

# Write viz format files
viz_elu_path = os.path.join(script_dir, viz_dir, "viz_out_elu.txt")
viz_leaky_path = os.path.join(script_dir, viz_dir, "viz_out_leaky.txt")
viz_gelu_path = os.path.join(script_dir, viz_dir, "viz_out_gelu.txt")

with open(viz_elu_path, "w", encoding="utf-8") as f:
    f.write("\n".join(viz_elu_lines) + "\n")

with open(viz_leaky_path, "w", encoding="utf-8") as f:
    f.write("\n".join(viz_leaky_lines) + "\n")

with open(viz_gelu_path, "w", encoding="utf-8") as f:
    f.write("\n".join(viz_gelu_lines) + "\n")

print("SUCCESS: Generated")
print(f"  - {output_elu_path}")
print(f"  - {output_leaky_path}")
print(f"  - {output_gelu_path}")
print(f"  - {viz_elu_path}")
print(f"  - {viz_leaky_path}")
print(f"  - {viz_gelu_path}")

