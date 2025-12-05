#!/usr/bin/env python3
"""
Generate out_elu.txt and out_leaky.txt from out_raw.txt by applying ELU and LeakyReLU activations.

- Reads out_raw.txt (binary format: 8 channels × 16 bits = 128 bits per line)
- Parses each 16-bit value as signed integer
- Applies ELU: f(x) = x if x > 0, else α * (e^x - 1) where α = 1.0
- Applies LeakyReLU: f(x) = x if x > 0, else α * x where α = 0.01
- Writes out_elu.txt and out_leaky.txt in the same binary format
"""

import os
import math
import sys


def binary_to_signed_int(binary_str: str, bits: int = 16) -> int:
  """
  Convert binary string to signed integer (2's complement).
  
  Args:
    binary_str: Binary string (e.g., "1111111111101100")
    bits: Number of bits (default 16)
  
  Returns:
    Signed integer value
  """
  val = int(binary_str, 2)
  # Check if MSB is 1 (negative in 2's complement)
  if val & (1 << (bits - 1)):
    # Negative: subtract 2^bits
    val = val - (1 << bits)
  return val


def signed_int_to_binary(val: int, bits: int = 16) -> str:
  """
  Convert signed integer to binary string (2's complement).
  
  Args:
    val: Signed integer value
    bits: Number of bits (default 16)
  
  Returns:
    Binary string (e.g., "1111111111101100")
  """
  # Handle negative values using 2's complement
  if val < 0:
    val_unsigned = (1 << bits) + val
  else:
    val_unsigned = val
  # Format as binary string with zero padding
  return format(val_unsigned & ((1 << bits) - 1), f"0{bits}b")


def elu(x: float, alpha: float = 1.0) -> float:
  """
  ELU (Exponential Linear Unit) activation function.
  
  f(x) = x if x > 0, else α * (e^x - 1)
  
  Args:
    x: Input value
    alpha: ELU parameter (default 1.0)
  
  Returns:
    ELU output value
  """
  if x > 0:
    return x
  else:
    return alpha * (math.exp(x) - 1.0)


def leaky_relu(x: float, alpha: float = 0.01) -> float:
  """
  LeakyReLU activation function.
  
  f(x) = x if x > 0, else α * x
  
  Args:
    x: Input value
    alpha: LeakyReLU parameter (default 0.01)
  
  Returns:
    LeakyReLU output value
  """
  if x > 0:
    return x
  else:
    return alpha * x


def quantize_to_int16(val: float) -> int:
  """
  Quantize float value to 16-bit signed integer.
  
  Args:
    val: Float value
  
  Returns:
    16-bit signed integer (clamped to [-32768, 32767])
  """
  # Round to nearest integer
  rounded = round(val)
  # Clamp to 16-bit signed integer range
  clamped = max(-32768, min(32767, rounded))
  return int(clamped)


def main():
  script_dir = os.path.dirname(os.path.abspath(__file__))
  input_path = os.path.join(script_dir, "out_raw.txt")
  output_elu_path = os.path.join(script_dir, "out_elu.txt")
  output_leaky_path = os.path.join(script_dir, "out_leaky.txt")

  if not os.path.exists(input_path):
    print(f"[FAIL] out_raw.txt not found at {input_path}")
    return

  print("[1] Reading out_raw.txt ...")
  with open(input_path, "r") as f:
    lines = f.readlines()

  print(f"    Loaded {len(lines)} lines")

  print("[2] Applying ELU and LeakyReLU to each 16-bit value ...")
  output_elu_lines = []
  output_leaky_lines = []
  
  for line_idx, line in enumerate(lines):
    line = line.strip()
    
    # Keep header lines as-is
    if line.startswith("#") or line == "":
      output_elu_lines.append(line)
      output_leaky_lines.append(line)
      continue
    
    # Parse binary string: 8 channels × 16 bits = 128 bits
    if len(line) != 128:
      print(f"[WARN] Line {line_idx + 1} has unexpected length: {len(line)} (expected 128)")
      output_elu_lines.append(line)
      output_leaky_lines.append(line)
      continue
    
    # Split into 8 channels, each 16 bits
    binary_str_elu = ""
    binary_str_leaky = ""
    
    for ch in range(8):
      start_idx = ch * 16
      end_idx = start_idx + 16
      channel_binary = line[start_idx:end_idx]
      
      # Convert to signed integer
      val_int = binary_to_signed_int(channel_binary, bits=16)
      
      # Convert to float for activation functions
      val_float = float(val_int)
      
      # Apply ELU
      elu_val = elu(val_float, alpha=1.0)
      elu_int = quantize_to_int16(elu_val)
      elu_binary = signed_int_to_binary(elu_int, bits=16)
      binary_str_elu += elu_binary
      
      # Apply LeakyReLU
      leaky_val = leaky_relu(val_float, alpha=0.01)
      leaky_int = quantize_to_int16(leaky_val)
      leaky_binary = signed_int_to_binary(leaky_int, bits=16)
      binary_str_leaky += leaky_binary
    
    output_elu_lines.append(binary_str_elu)
    output_leaky_lines.append(binary_str_leaky)

  print("[3] Writing out_elu.txt ...")
  with open(output_elu_path, "w") as f:
    f.write("\n".join(output_elu_lines) + "\n")

  print("[4] Writing out_leaky.txt ...")
  with open(output_leaky_path, "w") as f:
    f.write("\n".join(output_leaky_lines) + "\n")

  print("[OK] Files generated successfully.")
  print(f"     Input  : {input_path}")
  print(f"     ELU    : {output_elu_path}")
  print(f"     LeakyReLU: {output_leaky_path}")


if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)

