#!/usr/bin/env python3
"""
Visualization script to convert binary data files to human-readable decimal format
Converts:
  - activation_tile*.txt -> viz_activation_tile*.txt (2-bit values, 8 per row)
  - weight_itile*_otile*_kij*.txt -> viz_weight_*.txt (4-bit signed values, 8 per row)
  - output.txt -> viz_output.txt (16-bit signed values, 16 per row for 2-bit mode)
"""

import os
import re
import glob

# Paths
SOURCE_DIR = "."
OUTPUT_DIR = "viz"

def bin_to_signed(bin_str, bit_width):
    """Convert binary string to signed integer"""
    val = int(bin_str, 2)
    # Sign extension for signed numbers
    if val >= (1 << (bit_width - 1)):
        val -= (1 << bit_width)
    return val

def convert_activation_file(input_file, output_file):
    """Convert activation file from binary to decimal format"""
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        converted_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                converted_lines.append(line)
            else:
                # Each line has 16 bits (8 x 2-bit values)
                # Convert to 8 decimal values
                values = []
                for i in range(8):
                    start_idx = i * 2
                    end_idx = start_idx + 2
                    if end_idx <= len(line):
                        bin_val = line[start_idx:end_idx]
                        # 2-bit unsigned value (0-3)
                        val = int(bin_val, 2)
                        values.append(f"{val:3d}")
                    else:
                        values.append("  0")
                converted_lines.append(" ".join(values))
        
        # Write to output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write('\n'.join(converted_lines) + '\n')
        
        return True
    except Exception as e:
        print(f"Error converting {input_file}: {e}")
        return False

def convert_weight_file(input_file, output_file):
    """Convert weight file from binary to decimal format"""
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        converted_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                converted_lines.append(line)
            else:
                # Each line has 32 bits (8 x 4-bit signed values)
                # Convert to 8 signed decimal values
                values = []
                for i in range(8):
                    start_idx = i * 4
                    end_idx = start_idx + 4
                    if end_idx <= len(line):
                        bin_val = line[start_idx:end_idx]
                        # 4-bit signed value (-8 to 7)
                        val = bin_to_signed(bin_val, 4)
                        values.append(f"{val:4d}")
                    else:
                        values.append("   0")
                converted_lines.append(" ".join(values))
        
        # Write to output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write('\n'.join(converted_lines) + '\n')
        
        return True
    except Exception as e:
        print(f"Error converting {input_file}: {e}")
        return False

def convert_output_file(input_file, output_file):
    """Convert output file from binary to decimal format"""
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        converted_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                converted_lines.append(line)
            else:
                # Each line has 256 bits (16 x 16-bit signed values for 2-bit mode with col=16)
                # Convert to 16 signed decimal values
                values = []
                for i in range(16):
                    start_idx = i * 16
                    end_idx = start_idx + 16
                    if end_idx <= len(line):
                        bin_val = line[start_idx:end_idx]
                        # 16-bit signed value
                        val = bin_to_signed(bin_val, 16)
                        values.append(f"{val:7d}")
                    else:
                        values.append("      0")
                converted_lines.append(" ".join(values))
        
        # Write to output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write('\n'.join(converted_lines) + '\n')
        
        return True
    except Exception as e:
        print(f"Error converting {input_file}: {e}")
        return False

def main():
    """Main conversion function"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = script_dir
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    # Convert activation files
    print("Converting activation files...")
    activation_files = glob.glob(os.path.join(source_dir, "activation_tile*.txt"))
    for act_file in sorted(activation_files):
        filename = os.path.basename(act_file)
        output_filename = f"viz_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        total_count += 1
        if convert_activation_file(act_file, output_path):
            print(f"  [OK] {filename} -> {output_filename}")
            success_count += 1
        else:
            print(f"  [FAIL] {filename}")
    
    # Convert weight files
    print("\nConverting weight files...")
    weight_files = glob.glob(os.path.join(source_dir, "weight_itile*_otile*_kij*.txt"))
    for weight_file in sorted(weight_files):
        filename = os.path.basename(weight_file)
        output_filename = f"viz_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        total_count += 1
        if convert_weight_file(weight_file, output_path):
            print(f"  [OK] {filename} -> {output_filename}")
            success_count += 1
        else:
            print(f"  [FAIL] {filename}")
    
    # Convert output file
    print("\nConverting output file...")
    output_file = os.path.join(source_dir, "output.txt")
    if os.path.exists(output_file):
        total_count += 1
        output_path = os.path.join(output_dir, "viz_output.txt")
        if convert_output_file(output_file, output_path):
            print(f"  [OK] output.txt -> viz_output.txt")
            success_count += 1
        else:
            print(f"  [FAIL] output.txt")
    
    print("\n" + "=" * 60)
    print(f"Conversion complete: {success_count}/{total_count} files converted successfully")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()

