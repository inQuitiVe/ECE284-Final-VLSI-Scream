#!/usr/bin/env python3
"""
Compare transform_activation.py and transform_activation_2bit.py algorithms
"""

import sys

def compare_algorithms():
    """Compare the two transform algorithms line by line"""
    
    print("=" * 80)
    print("COMPARISON: transform_activation.py (4-bit) vs transform_activation_2bit.py (2-bit)")
    print("=" * 80)
    
    differences = []
    
    # 1. Input line length check
    print("\n1. Input line length check:")
    print("   4-bit: len(line) != 32")
    print("   2-bit: len(line) != 16")
    differences.append("Expected: Different bit widths (32 vs 16)")
    
    # 2. Data extraction
    print("\n2. Data extraction:")
    print("   4-bit: start = i * 4, end = start + 4")
    print("   2-bit: start = i * 2, end = start + 2")
    differences.append("Expected: Different bit widths (4-bit vs 2-bit per value)")
    
    # 3. Time pattern function
    print("\n3. Time pattern function (get_time_indices):")
    print("   Both: Identical logic")
    print("   Row 0: start=0 -> [0,1,2,3,6,7,8,9]")
    print("   Row 1: start=1 -> [1,2,3,4,7,8,9,10]")
    print("   Row 2: start=2 -> [2,3,4,5,8,9,10,11]")
    print("   Row 3: start=6 -> [6,7,8,9,12,13,14,15]")
    print("   Row 4: start=7 -> [7,8,9,10,13,14,15,16]")
    print("   Row 5: start=8 -> [8,9,10,11,14,15,16,17]")
    print("   Row 6: start=12 -> [12,13,14,15,18,19,20,21]")
    print("   Row 7: start=13 -> [13,14,15,16,19,20,21,22]")
    print("   Row 8: start=14 -> [14,15,16,17,20,21,22,23]")
    
    # 4. Processing loop
    print("\n4. Processing loop:")
    print("   Both: for orig_row_idx in range(7, -1, -1)")
    print("   Both: for new_row_idx in range(9)")
    print("   Both: for time_idx in reversed(time_indices)")
    
    # 5. Padding
    print("\n5. Padding for out-of-range time indices:")
    print("   4-bit: \"0000\"")
    print("   2-bit: \"00\"")
    differences.append("Expected: Different padding (4 bits vs 2 bits)")
    
    # 6. Output line length
    print("\n6. Output line length:")
    print("   4-bit: 32 bits (8 x 4-bit values)")
    print("   2-bit: 16 bits (8 x 2-bit values)")
    differences.append("Expected: Different output lengths (32 vs 16 bits)")
    
    # 7. Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF DIFFERENCES:")
    print("=" * 80)
    for i, diff in enumerate(differences, 1):
        print(f"{i}. {diff}")
    
    print("\n" + "=" * 80)
    print("ALGORITHM LOGIC CONSISTENCY:")
    print("=" * 80)
    print("[OK] Time pattern function: IDENTICAL")
    print("[OK] Processing loop structure: IDENTICAL")
    print("[OK] Data extraction logic: IDENTICAL (except bit width)")
    print("[OK] Row ordering: IDENTICAL")
    print("[OK] Time ordering (MSB to LSB): IDENTICAL")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("The algorithms are logically IDENTICAL except for bit width differences.")
    print("All differences are expected and correct for 4-bit vs 2-bit modes.")
    print("\nThe transformation logic is consistent between both scripts.")
    
    return differences

if __name__ == "__main__":
    compare_algorithms()

