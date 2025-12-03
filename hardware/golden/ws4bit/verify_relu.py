#!/usr/bin/env python3
"""Quick verification script to check ReLU application"""

def bin_to_signed(b):
  val = int(b, 2)
  if val & (1 << 15):
    val = val - (1 << 16)
  return val

def relu(x):
  return x if x > 0 else 0

# Original line 4 from out.txt
orig = "11111111111011000000000000001000000000000000010000000000000110111111111111000111111111111011111111111111110011101111111111000110"

# ReLU line 4 from output_relu.txt
relu_line = "00000000000000000000000000001000000000000000010000000000000110110000000000000000000000000000000000000000000000000000000000000000"

print("Line 4 comparison (8 channels):")
print("Channel | Original | ReLU Applied | Expected")
print("-" * 50)
for i in range(8):
  orig_val = bin_to_signed(orig[i*16:(i+1)*16])
  relu_val = bin_to_signed(relu_line[i*16:(i+1)*16])
  expected = relu(orig_val)
  match = "OK" if relu_val == expected else "FAIL"
  print(f"   {i}    | {orig_val:7d} | {relu_val:12d} | {expected:8d} {match}")

