#!/usr/bin/env python3
"""
Parse script to convert weight files from from_yufan format to ws4bit format
Converts:
  - weight_tile_0_kij_0.txt -> weight_itile0_otile0_kij0.txt
  - Changes comment format from "time" to "col"
"""

import os
import re
import shutil

# Paths
FROM_DIR = "from_yufan"
OUTPUT_DIR = "../ws2bit"

def convert_weight_file(input_file, output_file):
    """Convert a single weight file from from_yufan format to ws4bit format"""
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # Convert comment lines: time -> col
        converted_lines = []
        for line in lines:
            if line.startswith('#'):
                # Replace "time" with "col" in comments
                converted_line = line.replace('time', 'col')
                converted_lines.append(converted_line)
            else:
                # Keep data lines as is
                converted_lines.append(line)
        
        # Write to output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.writelines(converted_lines)
        
        return True
    except Exception as e:
        print(f"Error converting {input_file}: {e}")
        return False

def convert_filename(from_yufan_name):
    """Convert filename from from_yufan format to ws4bit format
    
    Args:
        from_yufan_name: e.g., "weight_tile_0_kij_0.txt"
    
    Returns:
        ws4bit_name: e.g., "weight_itile0_otile0_kij0.txt"
    
    Mapping:
        tile_0 -> itile0_otile0
        tile_1 -> itile0_otile1
        tile_2 -> itile1_otile0
        tile_3 -> itile1_otile1
    """
    # Pattern: weight_tile_<tile>_kij_<kij>.txt
    match = re.match(r'weight_tile_(\d+)_kij_(\d+)\.txt', from_yufan_name)
    if match:
        tile = int(match.group(1))
        kij = match.group(2)
        
        # Mapping: tile -> (itile, otile)
        # 0 -> (0, 0)
        # 1 -> (0, 1)
        # 2 -> (1, 0)
        # 3 -> (1, 1)
        tile_to_itile_otile = {
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1)
        }
        
        if tile in tile_to_itile_otile:
            itile, otile = tile_to_itile_otile[tile]
            return f"weight_itile{itile}_otile{otile}_kij{kij}.txt"
        else:
            print(f"Warning: Unknown tile number {tile}")
            return None
    return None

def main():
    """Main conversion function"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    from_dir = os.path.join(script_dir, FROM_DIR)
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    
    if not os.path.exists(from_dir):
        print(f"Error: Source directory not found: {from_dir}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all weight files
    weight_files = [f for f in os.listdir(from_dir) if f.startswith('weight_tile_') and f.endswith('.txt')]
    
    if not weight_files:
        print(f"No weight files found in {from_dir}")
        return
    
    print(f"Found {len(weight_files)} weight files to convert")
    print("=" * 60)
    
    success_count = 0
    for weight_file in sorted(weight_files):
        input_path = os.path.join(from_dir, weight_file)
        output_filename = convert_filename(weight_file)
        
        if output_filename:
            output_path = os.path.join(output_dir, output_filename)
            if convert_weight_file(input_path, output_path):
                print(f"[OK] Converted: {weight_file} -> {output_filename}")
                success_count += 1
            else:
                print(f"[FAIL] Failed: {weight_file}")
        else:
            print(f"[FAIL] Invalid filename format: {weight_file}")
    
    print("=" * 60)
    print(f"Conversion complete: {success_count}/{len(weight_files)} files converted successfully")

if __name__ == "__main__":
    main()

