#!/usr/bin/env python3
"""
Analyze partial sum (psum) intermediate process for OS mode
Generates psum for each of the 8 input channels separately
"""

import os
import glob

def bin_to_signed(bin_str, bit_width):
    """Convert binary string to signed integer"""
    val = int(bin_str, 2)
    if val >= (1 << (bit_width - 1)):
        val -= (1 << bit_width)
    return val

def bin_to_unsigned(bin_str, bit_width):
    """Convert binary string to unsigned integer"""
    return int(bin_str, 2)

def read_activation_file(filename):
    """Read activation file and return as 2D array [row][time]"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip header (first 3 lines)
    data_lines = [line.strip() for line in lines[3:] if line.strip()]
    
    # Parse: each line has 32 bits = 8 x 4-bit values
    # Format: rowXtime9[msb-lsb],rowXtime8[msb-lsb],....,rowXtime0[msb-lsb]
    # So each line: [time9, time8, time7, time6, time5, time4, time3, time2, time1, time0] (MSB to LSB)
    # Actually, looking at the pattern, it's 8 time values per row
    activations = []
    for line in data_lines:
        if len(line) != 32:
            continue
        row_data = []
        # Extract 8 x 4-bit values (MSB to LSB, so time9 to time0)
        for i in range(8):
            start = i * 4
            end = start + 4
            bin_val = line[start:end]
            val = bin_to_unsigned(bin_val, 4)  # 4-bit unsigned (0-15)
            row_data.append(val)
        activations.append(row_data)
    
    return activations

def read_weight_file(filename):
    """Read weight file and return as 2D array [row][col]"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip header (first 3 lines)
    data_lines = [line.strip() for line in lines[3:] if line.strip()]
    
    # Parse: each line has 32 bits = 8 x 4-bit signed values
    # Format: rowXcol7[msb-lsb],rowXcol6[msb-lsb],....,rowXcol0[msb-lsb]
    weights = []
    for line in data_lines:
        if len(line) != 32:
            continue
        row_data = []
        # Extract 8 x 4-bit signed values
        for i in range(8):
            start = i * 4
            end = start + 4
            bin_val = line[start:end]
            val = bin_to_signed(bin_val, 4)  # 4-bit signed (-8 to 7)
            row_data.append(val)
        weights.append(row_data)
    
    return weights

def calculate_psum_per_channel(activations, weights_dict, num_input_channels=8, num_output_channels=8):
    """
    Calculate psum for each input channel separately
    
    In OS mode:
    - Activation data: 72 rows = 8 input channels × 9 kernel positions (kij 0-8)
    - Each input channel has 9 rows, one for each kij
    - Each row has 8 time values
    - Weight data: 9 files (kij 0-8), each has 8 rows (output channels) × 8 cols (input channels)
    
    activations: [row][time] - activation data (72 rows, 8 time values per row)
                 Row organization: [ic0_kij0, ic0_kij1, ..., ic0_kij8, ic1_kij0, ..., ic7_kij8]
    weights_dict: {kij: [row][col]} - weight data for each kij (0-8)
                  weights[kij][oc][ic] = weight for output channel oc, input channel ic, kernel position kij
    """
    # Initialize psum arrays: [input_channel][output_channel][time]
    psum_per_channel = {}
    
    for ic in range(num_input_channels):
        psum_per_channel[ic] = {}
        for oc in range(num_output_channels):
            psum_per_channel[ic][oc] = [0] * 8  # Initialize for 8 time steps
    
    # Process each input channel
    for ic in range(num_input_channels):
        # Each input channel has 9 activation rows (one per kij)
        start_row = ic * 9
        
        # Process each kernel position (kij 0-8)
        for kij in range(9):
            if kij not in weights_dict:
                continue
            
            act_row_idx = start_row + kij
            
            if act_row_idx >= len(activations):
                continue
            
            act_row = activations[act_row_idx]
            weights = weights_dict[kij]
            
            # For each output channel
            for oc in range(num_output_channels):
                if oc >= len(weights):
                    continue
                
                weight_row = weights[oc]
                
                # Get weight for this input channel
                if ic >= len(weight_row):
                    continue
                
                weight_val = weight_row[ic]
                
                # For each time step
                for time_idx in range(min(len(act_row), 8)):
                    act_val = act_row[time_idx]
                    
                    # Calculate product: activation × weight
                    product = act_val * weight_val
                    
                    # Accumulate psum for this input channel, output channel, and time step
                    psum_per_channel[ic][oc][time_idx] += product
    
    return psum_per_channel

def main():
    """Main analysis function"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir
    
    # Read activation file
    print("Reading activation file...")
    act_file = os.path.join(base_dir, "activation_tile0.txt")
    if not os.path.exists(act_file):
        print(f"Error: {act_file} not found")
        return
    
    activations = read_activation_file(act_file)
    print(f"  Loaded {len(activations)} activation rows")
    
    # Read all weight files (kij 0-8)
    print("\nReading weight files...")
    weights_dict = {}
    for kij in range(9):
        weight_file = os.path.join(base_dir, f"weight_itile0_otile0_kij{kij}.txt")
        if os.path.exists(weight_file):
            weights = read_weight_file(weight_file)
            weights_dict[kij] = weights
            print(f"  Loaded kij{kij}: {len(weights)} weight rows")
        else:
            print(f"  Warning: {weight_file} not found")
    
    if not weights_dict:
        print("Error: No weight files found")
        return
    
    # Calculate psum per channel
    print("\nCalculating psum per input channel...")
    psum_per_channel = calculate_psum_per_channel(activations, weights_dict)
    
    # Create output directory
    output_dir = os.path.join(base_dir, "psum_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Write results for each input channel
    print("\nWriting psum analysis results...")
    for ic in range(8):
        output_file = os.path.join(output_dir, f"psum_input_channel_{ic}.txt")
        with open(output_file, 'w') as f:
            f.write(f"# PSUM Analysis for Input Channel {ic} #\n")
            f.write(f"# Format: [time] -> psum for each output channel (0-7) #\n")
            f.write("#" + "="*70 + "#\n\n")
            
            # Get max time length
            max_time = 0
            for oc in range(8):
                if len(psum_per_channel[ic][oc]) > max_time:
                    max_time = len(psum_per_channel[ic][oc])
            
            # Write header
            f.write("Time".ljust(8))
            for oc in range(8):
                f.write(f"OC{oc:2d}".rjust(10))
            f.write("\n")
            f.write("-" * 88 + "\n")
            
            # Write data
            for time_idx in range(max_time):
                f.write(f"{time_idx:4d}".ljust(8))
                for oc in range(8):
                    if time_idx < len(psum_per_channel[ic][oc]):
                        psum_val = psum_per_channel[ic][oc][time_idx]
                        f.write(f"{psum_val:10d}")
                    else:
                        f.write("         0")
                f.write("\n")
        
        print(f"  [OK] Input channel {ic} -> psum_input_channel_{ic}.txt")
    
    # Write summary file
    summary_file = os.path.join(output_dir, "psum_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("# PSUM Summary - Total contribution from each input channel #\n")
        f.write("#" + "="*70 + "#\n\n")
        
        f.write("Input Channel".ljust(20))
        for oc in range(8):
            f.write(f"OC{oc:2d} Total".rjust(12))
        f.write("\n")
        f.write("-" * 116 + "\n")
        
        for ic in range(8):
            f.write(f"IC{ic:2d}".ljust(20))
            for oc in range(8):
                total = sum(psum_per_channel[ic][oc])
                f.write(f"{total:12d}")
            f.write("\n")
    
    print(f"\n  [OK] Summary -> psum_summary.txt")
    print(f"\nAnalysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()

