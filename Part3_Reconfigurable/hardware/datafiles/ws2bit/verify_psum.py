#!/usr/bin/env python3
"""
Verify calculated psum against output.txt
Maps nij_cnt and kij to o_addr correctly, then compares with output.txt
"""

import os
import re

def bin_to_signed(bin_str, bit_width):
    """Convert binary string to signed integer"""
    val = int(bin_str, 2)
    if val >= (1 << (bit_width - 1)):
        val -= (1 << bit_width)
    return val

def relu(x):
    """ReLU activation: max(0, x)"""
    return max(0, x)

def calculate_o_addr(nij_cnt, kij):
    """
    Calculate o_addr from nij_cnt and kij (matching SFU.v logic)
    row_a = nij_cnt / 6
    col_a = nij_cnt % 6
    k_row = kij / 3
    k_col = kij % 3
    o_row = row_a - k_row
    o_col = col_a - k_col
    o_addr = o_row * 4 + o_col (if o_row in [0,3] and o_col in [0,3])
    """
    # row_a = nij_cnt / 6
    row_a = nij_cnt // 6
    
    # col_a = nij_cnt % 6
    col_a = nij_cnt % 6
    
    # k_row = kij / 3
    k_row = kij // 3
    
    # k_col = kij % 3
    k_col = kij % 3
    
    # o_row = row_a - k_row
    o_row = row_a - k_row
    
    # o_col = col_a - k_col
    o_col = col_a - k_col
    
    # Check if in valid range [0, 3] for both o_row and o_col
    if 0 <= o_row < 4 and 0 <= o_col < 4:
        o_addr = o_row * 4 + o_col  # 0..15
        return o_addr, True
    else:
        return 0, False  # acc = 0, o_addr doesn't matter

def read_output_file(filename):
    """Read output file and return as list of lists [o_addr][col]"""
    outputs = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Each line has 256 bits (16 x 16-bit signed values for 2-bit mode with col=16)
        # Each line corresponds to one o_addr (0-15)
        col_vals = []
        for i in range(16):  # Read all 16 columns
            start_idx = i * 16
            end_idx = start_idx + 16
            if end_idx <= len(line):
                bin_val = line[start_idx:end_idx]
                val = bin_to_signed(bin_val, 16)
                col_vals.append(val)
            else:
                col_vals.append(0)
        outputs.append(col_vals)
    
    return outputs

def read_calc_psum_output(filename):
    """Read calculated psum output file
    Returns: dict[kij][nij_cnt] = [psum0, psum1, ..., psum7]
    """
    psums_by_kij_nij = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Parse psum line: "kij= 0 time= 7:       2       4       5       0      -9      -7      -5       8"
        match = re.match(r'kij\s*=\s*(\d+)\s+time\s*=\s*(\d+):\s+(.+)', line)
        if match:
            kij = int(match.group(1))
            nij_cnt = int(match.group(2))  # This is actually nij_cnt (time in activation flow)
            psum_str = match.group(3)
            
            # Parse psum values
            psum_vals = []
            for val_str in psum_str.split():
                psum_vals.append(int(val_str))
            
            if kij not in psums_by_kij_nij:
                psums_by_kij_nij[kij] = {}
            psums_by_kij_nij[kij][nij_cnt] = psum_vals
    
    return psums_by_kij_nij

def accumulate_psums_by_o_addr(psums_by_kij_nij, num_kij=9, num_nij=36):
    """
    Accumulate psums by o_addr (0-15) instead of by nij_cnt
    For each o_addr, sum psums from all kij rounds where acc=1
    """
    # Initialize: o_addr_psums[o_addr][col] = accumulated psum
    o_addr_psums = {}
    for o_addr in range(16):
        o_addr_psums[o_addr] = [0] * 8  # 8 columns
    
    # For each kij and nij_cnt, calculate o_addr and accumulate
    for kij in range(num_kij):
        if kij not in psums_by_kij_nij:
            continue
        
        for nij_cnt in range(num_nij):
            if nij_cnt not in psums_by_kij_nij[kij]:
                continue
            
            # Calculate o_addr for this (nij_cnt, kij) pair
            o_addr, acc = calculate_o_addr(nij_cnt, kij)
            
            if acc:  # Only accumulate if acc=1
                kij_psums = psums_by_kij_nij[kij][nij_cnt]
                for col in range(8):
                    o_addr_psums[o_addr][col] += kij_psums[col]
    
    return o_addr_psums

def main():
    """Main verification function"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 80)
    print("PSUM Verification Script")
    print("=" * 80)
    
    # Read files
    print("\n[1] Reading output.txt...")
    expected_output = read_output_file(os.path.join(script_dir, "output.txt"))
    print(f"    Loaded {len(expected_output)} o_addr entries (0-15)")
    print(f"    Each entry has {len(expected_output[0]) if expected_output else 0} columns")
    
    print("\n[2] Reading calc_psum_output.txt...")
    calc_psums = read_calc_psum_output(os.path.join(script_dir, "calc_psum_output.txt"))
    print(f"    Loaded psums for {len(calc_psums)} kij rounds")
    
    # Accumulate psums by o_addr
    print("\n[3] Accumulating psums by o_addr (matching SFU logic)...")
    o_addr_psums = accumulate_psums_by_o_addr(calc_psums)
    print(f"    Accumulated psums for 16 o_addr entries (0-15)")
    
    # Apply ReLU to accumulated psums (since SFU.v now has ReLU)
    print("\n[4] Applying ReLU to accumulated psums...")
    o_addr_psums_relu = {}
    for o_addr in range(16):
        o_addr_psums_relu[o_addr] = [relu(x) for x in o_addr_psums[o_addr]]
    
    # Compare
    print("\n[5] Comparing results...")
    print("=" * 80)
    
    errors = 0
    matches = 0
    
    print("\nDetailed comparison:")
    print("-" * 80)
    
    for o_addr in range(min(16, len(expected_output))):
        expected = expected_output[o_addr][:8]  # First 8 columns
        calculated = o_addr_psums_relu[o_addr]
        
        match = True
        diff_cols = []
        for col in range(8):
            if expected[col] != calculated[col]:
                match = False
                diff_cols.append(col)
        
        if match:
            matches += 1
            if o_addr < 3 or o_addr >= 13:
                print(f"  o_addr={o_addr:2d}: [MATCH]    {calculated}")
        else:
            errors += 1
            print(f"  o_addr={o_addr:2d}: [MISMATCH]")
            print(f"    Expected:  {expected}")
            print(f"    Calculated: {calculated}")
            print(f"    Diff:      {[expected[col] - calculated[col] for col in range(8)]}")
            print(f"    Diff cols: {diff_cols}")
            
            # Show breakdown for first few errors
            if errors <= 3:
                print(f"    Breakdown by kij for o_addr={o_addr}:")
                # Find which (nij_cnt, kij) pairs contribute to this o_addr
                contrib_list = []
                for kij in range(9):
                    if kij not in calc_psums:
                        continue
                    for nij_cnt in range(36):
                        if nij_cnt not in calc_psums[kij]:
                            continue
                        calc_o_addr, acc = calculate_o_addr(nij_cnt, kij)
                        if acc and calc_o_addr == o_addr:
                            kij_psum = calc_psums[kij][nij_cnt]
                            contrib_list.append((kij, nij_cnt, kij_psum))
                
                # Show all contributions
                for kij, nij_cnt, kij_psum in contrib_list[:10]:  # Show first 10
                    print(f"      kij={kij}, nij_cnt={nij_cnt}: {kij_psum}")
                if len(contrib_list) > 10:
                    print(f"      ... (total {len(contrib_list)} contributions)")
                
                # Show sum of all contributions
                sum_contrib = [0] * 8
                for kij, nij_cnt, kij_psum in contrib_list:
                    for col in range(8):
                        sum_contrib[col] += kij_psum[col]
                print(f"      Sum of all contributions: {sum_contrib}")
                print(f"      After ReLU: {[relu(x) for x in sum_contrib]}")
            
            if errors >= 10:  # Limit error output
                print(f"    ... (showing first 10 errors)")
                break
    
    print("=" * 80)
    print(f"\n[6] Summary:")
    print(f"    Total o_addr entries: {min(16, len(expected_output))}")
    print(f"    Matches: {matches}")
    print(f"    Mismatches: {errors}")
    print(f"    Match rate: {matches*100.0/min(16, len(expected_output)):.1f}%")
    
    if errors == 0:
        print("\n[OK] All psum calculations match the expected output!")
    else:
        print(f"\n[FAIL] Found {errors} mismatches.")
        print("       Please check:")
        print("       1. Calculation logic in calc_psum.py")
        print("       2. o_addr mapping logic (nij_cnt, kij -> o_addr)")
        print("       3. Whether output.txt includes ReLU or not")
    
    # Save comparison results
    output_file = os.path.join(script_dir, "verify_psum_result.txt")
    with open(output_file, 'w') as f:
        f.write("# PSUM Verification Results\n")
        f.write(f"# Matches: {matches}/{min(16, len(expected_output))}\n")
        f.write("#" + "=" * 78 + "\n\n")
        
        for o_addr in range(min(16, len(expected_output))):
            expected = expected_output[o_addr][:8]
            calculated = o_addr_psums_relu[o_addr]
            
            match = all(expected[col] == calculated[col] for col in range(8))
            status = "MATCH" if match else "MISMATCH"
            
            f.write(f"o_addr={o_addr:2d}: [{status}]\n")
            f.write(f"  Expected:  {expected}\n")
            f.write(f"  Calculated: {calculated}\n")
            if not match:
                f.write(f"  Diff:      {[expected[col] - calculated[col] for col in range(8)]}\n")
            f.write("\n")
    
    print(f"\n[7] Detailed results saved to: verify_psum_result.txt")

if __name__ == "__main__":
    main()
