#!/usr/bin/env python3
"""
Generate output_relu.txt from out.txt by applying ReLU activation.

- Reads out.txt (binary format: 8 channels × 16 bits = 128 bits per line)
- Parses each 16-bit value as signed integer
- Applies ReLU: max(0, x) - negative values become 0
- Writes output_relu.txt in the same binary format
"""

import os


def relu(x: int) -> int:
  """ReLU activation: max(0, x)"""
  return x if x > 0 else 0


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


def main():
  script_dir = os.path.dirname(os.path.abspath(__file__))
  input_path = os.path.join(script_dir, "out.txt")
  output_path = os.path.join(script_dir, "output_relu.txt")

  if not os.path.exists(input_path):
    print(f"[FAIL] out.txt not found at {input_path}")
    return

  print("[1] Reading out.txt ...")
  with open(input_path, "r") as f:
    lines = f.readlines()

  print(f"    Loaded {len(lines)} lines")

  print("[2] Applying ReLU to each 16-bit value ...")
  output_lines = []
  
  for line_idx, line in enumerate(lines):
    line = line.strip()
    
    # Keep header lines as-is
    if line.startswith("#") or line == "":
      output_lines.append(line)
      continue
    
    # Parse binary string: 8 channels × 16 bits = 128 bits
    if len(line) != 128:
      print(f"[WARN] Line {line_idx + 1} has unexpected length: {len(line)} (expected 128)")
      output_lines.append(line)
      continue
    
    # Split into 8 channels, each 16 bits
    binary_str = ""
    for ch in range(8):
      start_idx = ch * 16
      end_idx = start_idx + 16
      channel_binary = line[start_idx:end_idx]
      
      # Convert to signed integer
      val = binary_to_signed_int(channel_binary, bits=16)
      
      # Apply ReLU
      val_relu = relu(val)
      
      # Convert back to binary
      channel_binary_relu = signed_int_to_binary(val_relu, bits=16)
      binary_str += channel_binary_relu
    
    output_lines.append(binary_str)

  print("[3] Writing output_relu.txt ...")
  with open(output_path, "w") as f:
    f.write("\n".join(output_lines) + "\n")

  print("[OK] output_relu.txt generated.")
  print(f"     Input  : {input_path}")
  print(f"     Output : {output_path}")


if __name__ == "__main__":
  main()

