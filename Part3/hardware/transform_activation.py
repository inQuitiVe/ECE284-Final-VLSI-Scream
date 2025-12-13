#!/usr/bin/env python3
"""
Transform activation files from golden/os4bit/ref/ to golden/os4bit/
Reorganizes data according to OS mode requirements.
"""

import os
import glob

def transform_activation_file(input_file, output_file):
    """
    Transform activation file for OS mode.
    
    Original format: Each line represents a time, with 8 rows (row7 to row0)
                    Format: timeXrow7[msb-lsb], timeXrow6[msb-lsb], ..., timeXrow0[msb-lsb]
                    Each line: 32 bits = 8 x 4-bit values (row7 to row0)
    
    Output format: Each line represents a new row, with 8 time elements
                   Pattern for new rows:
                   - row 0: time [0,1,2,3, 6,7,8,9] (MSB to LSB)
                   - row 1: time [1,2,3,4, 7,8,9,10]
                   - row 2: time [3,4,5,6, 9,10,11,12]
                   - row 3: time [6,7,8,9, 12,13,14,15]
                   - row 4: time [9,10,11,12, 15,16,17,18]
                   - row 5: time [12,13,14,15, 18,19,20,21]
                   - row 6: time [15,16,17,18, 21,22,23,24]
                   - row 7: time [18,19,20,21, 24,25,26,27]
                   - row 8: time [21,22,23,24, 27,28,29,30]
                   This pattern repeats 8 times (once for each original row)
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Keep the first 3 comment lines
    header = lines[:3]
    
    # Extract data lines (skip empty lines)
    data_lines = [line.strip() for line in lines[3:] if line.strip()]
    
    if len(data_lines) != 36:
        print(f"Warning: {input_file} has {len(data_lines)} data lines, expected 36")
        return False
    
    # Parse each line: 32 bits = 8 x 4-bit values
    # Original format: timeXrow7[msb-lsb], timeXrow6[msb-lsb], ..., timeXrow0[msb-lsb]
    # So each line contains: [row7, row6, row5, row4, row3, row2, row1, row0]
    # data[time_idx][row_idx] where row_idx=0 is row7, row_idx=7 is row0
    data = []
    for time_idx, line in enumerate(data_lines):
        if len(line) != 32:
            print(f"Error: {input_file} line {time_idx+4} has length {len(line)}, expected 32")
            return False
        
        # Extract 8 x 4-bit values (each 4 bits)
        # line format: [row7, row6, row5, row4, row3, row2, row1, row0] (MSB to LSB)
        row_data = []
        for i in range(8):
            start = i * 4
            end = start + 4
            row_data.append(line[start:end])
        data.append(row_data)
    
    # Define the time pattern for each new row (0-8)
    # Pattern: [start, start+1, start+2, start+3, start+6, start+7, start+8, start+9]
    # Row 0: [0, 1, 2, 3, 6, 7, 8, 9]
    # Row 1: [1, 2, 3, 4, 7, 8, 9, 10]
    # Row 2: [3, 4, 5, 6, 9, 10, 11, 12]
    # Row 3: [6, 7, 8, 9, 12, 13, 14, 15]
    # Row 4: [9, 10, 11, 12, 15, 16, 17, 18]
    # Row 5: [12, 13, 14, 15, 18, 19, 20, 21]
    # Row 6: [15, 16, 17, 18, 21, 22, 23, 24]
    # Row 7: [18, 19, 20, 21, 24, 25, 26, 27]
    # Row 8: [14, 15, 16, 17, 20, 21, 22, 23]  (special case)
    def get_time_indices(new_row_idx):
        """Get time indices for a new row"""
        if new_row_idx == 0:
            start = 0
        elif new_row_idx == 1:
            start = 1
        elif new_row_idx == 2:
            start = 3
        elif new_row_idx == 8:
            # Special case for row 8
            start = 14
        else:
            # For row 3-7: start = 3 * new_row_idx - 3
            start = 3 * new_row_idx - 3
        
        return [start, start+1, start+2, start+3, start+6, start+7, start+8, start+9]
    
    # Generate new rows
    # Process from row0 to row7 (orig_row_idx=7 to 0)
    # For each original row, generate 9 new rows
    # Each new row contains 8 time elements, arranged from time large (MSB) to time small (LSB)
    new_rows = []
    for orig_row_idx in range(7, -1, -1):  # Start from row0 (orig_row_idx=7) to row7 (orig_row_idx=0)
        for new_row_idx in range(9):  # new row 0 to 8
            time_indices = get_time_indices(new_row_idx)
            new_row_data = []
            
            # Collect data from specified time indices
            # Arrange from time large (MSB) to time small (LSB), so reverse the time_indices
            for time_idx in reversed(time_indices):
                if time_idx < len(data):
                    # Get the value from this time and original row
                    # data[time_idx][orig_row_idx] gives us the value at (time_idx, orig_row_idx)
                    # where orig_row_idx=0 is row7, orig_row_idx=7 is row0
                    new_row_data.append(data[time_idx][orig_row_idx])
                else:
                    # If time index is out of range, pad with zeros
                    new_row_data.append("0000")
            
            # Join to form 32-bit line (8 x 4-bit values)
            new_rows.append(''.join(new_row_data))
    
    # Write output file
    with open(output_file, 'w') as f:
        # Write header (update format to reflect new structure)
        f.write("#row0time9[msb-lsb],row0time8[msb-lsb],....,row0time0[msb-lsb]#\n")
        f.write("#row1time10[msb-lsb],row1time9[msb-lsb],....,row1time1[msb-lsb]#\n")
        f.write("#................#\n")
        
        # Write transformed data (72 rows total: 8 original rows × 9 new rows)
        for row in new_rows:
            f.write(row + '\n')
    
    return True

def main():
    # Paths
    ref_dir = "golden/os4bit/ref"
    output_dir = "golden/os4bit"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find activation files in ref directory
    pattern = os.path.join(ref_dir, "activation*.txt")
    activation_files = glob.glob(pattern)
    
    if not activation_files:
        print(f"No activation files found in {ref_dir}")
        return
    
    print(f"Found {len(activation_files)} activation files to transform")
    
    success_count = 0
    for input_file in sorted(activation_files):
        # Get filename
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        
        print(f"Processing: {filename}")
        
        if transform_activation_file(input_file, output_file):
            print(f"  -> Successfully transformed to {output_file}")
            success_count += 1
        else:
            print(f"  -> Failed to transform {input_file}")
    
    print(f"\nCompleted: {success_count}/{len(activation_files)} files processed successfully")

if __name__ == "__main__":
    main()

