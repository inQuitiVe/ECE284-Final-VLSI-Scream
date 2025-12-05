#!/usr/bin/env python3
"""
Calculate partial sums (psum) for systolic array
For each round (kij), calculates 8 psum values (one per output channel/column)
Each psum is the sum of activation * weight for all rows in that column
"""

import os
import re
import glob

# Parameters
ROW = 8
COL = 8  # Only calculate first 8 output channels
NUM_KIJ = 9

def bin_to_signed(bin_str, bit_width):
    """Convert binary string to signed integer"""
    val = int(bin_str, 2)
    # Sign extension for signed numbers
    if val >= (1 << (bit_width - 1)):
        val -= (1 << bit_width)
    return val

def read_activation_file(filename):
    """Read activation file and return as 2D list [time][row]"""
    activations = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Each line has 16 bits (8 x 2-bit values)
        row_vals = []
        for i in range(ROW):
            start_idx = i * 2
            end_idx = start_idx + 2
            if end_idx <= len(line):
                bin_val = line[start_idx:end_idx]
                # 2-bit unsigned value (0-3)
                val = int(bin_val, 2)
                row_vals.append(val)
            else:
                row_vals.append(0)
        activations.append(row_vals)
    
    return activations

def read_weight_file(filename):
    """Read weight file and return as 2D list [col][row]"""
    weights = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Each line has 32 bits (8 x 4-bit signed values)
        # Each line represents one column, with 8 weight values (one per row)
        col_vals = []
        for i in range(ROW):
            start_idx = i * 4
            end_idx = start_idx + 4
            if end_idx <= len(line):
                bin_val = line[start_idx:end_idx]
                # 4-bit signed value (-8 to 7)
                val = bin_to_signed(bin_val, 4)
                col_vals.append(val)
            else:
                col_vals.append(0)
        weights.append(col_vals)
    
    return weights

def calculate_psum_round(act_tile0, act_tile1, weight_tile0, weight_tile1, time_idx):
    """
    Calculate psum for one round (one time step)
    Returns 8 psum values (one per column)
    
    In 2-bit mode:
    - Activation: two tiles, interleaved when fed to array
    - Weight: tile1 first (itile1_otile0), then tile0 (itile0_otile0)
    - For each column: psum = sum(activation[row] * weight[col][row] for all rows)
    """
    psums = [0] * COL
    
    # Check if we have valid data
    if time_idx >= len(act_tile0) or time_idx >= len(act_tile1):
        return psums
    
    # For each column (output channel)
    for col in range(COL):
        psum = 0
        
        # For each row
        for row in range(ROW):
            # Get activation values for this row from both tiles
            act0 = act_tile0[time_idx][ROW-row-1]  # tile0 activation (2-bit, 0-3)
            act1 = act_tile1[time_idx][ROW-row-1]  # tile1 activation (2-bit, 0-3)
            
            # Get weights: weight files are [col][row]
            # weight_tile1 is itile1_otile0 (loaded first)
            # weight_tile0 is itile0_otile0 (loaded second)
            if col < len(weight_tile1) and row < len(weight_tile1[col]):
                w1 = weight_tile1[col][ROW-row-1]  # tile1 weight (4-bit signed)
            else:
                w1 = 0
            
            if col < len(weight_tile0) and row < len(weight_tile0[col]):
                w0 = weight_tile0[col][ROW-row-1]  # tile0 weight (4-bit signed)
            else:
                w0 = 0
            
            # In 2-bit mode with interleaved activation:
            # The array receives: [t1_7[1:0] t0_7[1:0] t1_6[1:0] t0_6[1:0] ...]
            # Weight is loaded: tile1 first (w1), then tile0 (w0)
            # So for each row:
            # - tile1 activation (act1) multiplies with tile1 weight (w1)
            # - tile0 activation (act0) multiplies with tile0 weight (w0)
            # Then sum both contributions
            
            # Calculate MAC contributions
            print(f"act0: {act0}, w0: {w0}, act1: {act1}, w1: {w1}")
            psum += act0 * w0  # tile0: activation * weight
            psum += act1 * w1  # tile1: activation * weight
        
        psums[col] = psum
    
    return psums

def main():
    """Main calculation function"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Read activation files
    print("Reading activation files...")
    act_tile0 = read_activation_file(os.path.join(script_dir, "activation_tile0.txt"))
    act_tile1 = read_activation_file(os.path.join(script_dir, "activation_tile1.txt"))
    print(f"  Loaded {len(act_tile0)} time steps for tile0")
    print(f"  Loaded {len(act_tile1)} time steps for tile1")
    
    # Calculate psum for each kij
    print("\nCalculating psum for each kij round...")
    print("=" * 80)
    
    all_psums = []
    
    for kij in range(NUM_KIJ):
        # Read weight files for this kij
        weight_file0 = os.path.join(script_dir, f"weight_itile0_otile0_kij{kij}.txt")
        weight_file1 = os.path.join(script_dir, f"weight_itile1_otile0_kij{kij}.txt")
        
        if not os.path.exists(weight_file0) or not os.path.exists(weight_file1):
            print(f"Warning: Weight files for kij={kij} not found, skipping...")
            continue
        
        weight_tile0 = read_weight_file(weight_file0)
        weight_tile1 = read_weight_file(weight_file1)
        
        print(f"\nkij = {kij}")
        print("-" * 80)
        
        # Calculate psum for each time step
        num_times = min(len(act_tile0), len(act_tile1))
        round_psums = []
        
        for time_idx in range(num_times):
            psums = calculate_psum_round(act_tile0, act_tile1, weight_tile0, weight_tile1, time_idx)
            round_psums.append(psums)
            
            # Print all time steps (or first few and last few for long sequences)
            if num_times <= 20 or time_idx < 5 or time_idx >= num_times - 3:
                psum_str = " ".join([f"{p:7d}" for p in psums])
                print(f"  time={time_idx:2d}: {psum_str}")
        
        all_psums.append(round_psums)
    
    print("\n" + "=" * 80)
    print("Calculation complete!")
    
    # Save results to file
    output_file = os.path.join(script_dir, "calc_psum_output.txt")
    with open(output_file, 'w') as f:
        f.write("# Calculated psum for each kij round\n")
        f.write("# Format: kij, time, psum[0], psum[1], ..., psum[7]\n")
        f.write("#" + "=" * 78 + "\n")
        
        for kij, round_psums in enumerate(all_psums):
            f.write(f"\n# kij = {kij}\n")
            for time_idx, psums in enumerate(round_psums):
                psum_str = " ".join([f"{p:7d}" for p in psums])
                f.write(f"kij={kij:2d} time={time_idx:2d}: {psum_str}\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()

