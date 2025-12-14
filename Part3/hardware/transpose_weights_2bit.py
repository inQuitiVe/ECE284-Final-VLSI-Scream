#!/usr/bin/env python3
"""
Transpose weight files from golden/os2bit/ref/ to golden/os2bit/
Converts col-row format to row-col format (transpose) for 2-bit mode
"""

import os
import glob

def transpose_weight_file(input_file, output_file):
    """
    Transpose a weight file from col-row format to row-col format.
    
    Input format: Each line represents a column with 8 rows
                  Format: colXrow7[msb-lsb], colXrow6[msb-lsb], ..., colXrow0[msb-lsb]
                  Each line: 32 bits = 8 x 4-bit signed values (row7 to row0)
    
    Output format: Each line represents a row with 8 columns
                   Format: rowXcol0[msb-lsb], rowXcol1[msb-lsb], ..., rowXcol7[msb-lsb]
                   Output order: row7, row6, row5, row4, row3, row2, row1, row0
                   Each line: 32 bits = 8 x 4-bit signed values (col0 to col7)
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Keep the first 3 comment lines
    header = lines[:3]
    
    # Extract data lines (skip empty lines)
    data_lines = [line.strip() for line in lines[3:] if line.strip()]
    
    if len(data_lines) != 8:
        print(f"Warning: {input_file} has {len(data_lines)} data lines, expected 8")
        return False
    
    # Parse each line: 32 bits = 8 x 4-bit values
    # matrix[col_idx][row_idx] where row_idx=0 is row7, row_idx=1 is row6, ..., row_idx=7 is row0
    matrix = []
    for col_idx, line in enumerate(data_lines):
        if len(line) != 32:
            print(f"Error: {input_file} has line with length {len(line)}, expected 32")
            return False
        
        # Extract 8 x 4-bit values (each 4 bits)
        # line format: [row7, row6, row5, row4, row3, row2, row1, row0]
        row_data = []
        for i in range(8):
            start = i * 4
            end = start + 4
            row_data.append(line[start:end])
        matrix.append(row_data)
    
    # Transpose the matrix: [col][row] -> [row][col]
    # Original: matrix[col_idx][row_idx] where row_idx=0 is row7, row_idx=7 is row0
    # Output format: row0col7 to row0col0, row1col7 to row1col0, ..., row7col7 to row7col0
    # So output order: row0, row1, row2, ..., row7 (row_idx=7, 6, 5, 4, 3, 2, 1, 0)
    # And within each row: col7, col6, col5, ..., col0 (col_idx=7, 6, 5, 4, 3, 2, 1, 0)
    transposed = []
    for output_row_idx in range(8):  # output_row_idx=0 is row0, output_row_idx=7 is row7
        # Map output_row_idx to original row_idx: row0 -> row_idx=7, row1 -> row_idx=6, ..., row7 -> row_idx=0
        orig_row_idx = 7 - output_row_idx
        new_row = []
        # Within each row, output col7 to col0 (col_idx=7, 6, 5, 4, 3, 2, 1, 0)
        for output_col_idx in range(8):  # output_col_idx=0 is col7, output_col_idx=7 is col0
            # Map output_col_idx to original col_idx: col7 -> col_idx=7, col6 -> col_idx=6, ..., col0 -> col_idx=0
            orig_col_idx = 7 - output_col_idx
            # Get value from original matrix
            new_row.append(matrix[orig_col_idx][orig_row_idx])
        transposed.append(''.join(new_row))
    
    # Write output file
    with open(output_file, 'w') as f:
        # Write transposed header (swap col and row in the format, and reverse order)
        # Original: #col0row7[msb-lsb],col0row6[msb-lst],....,col0row0[msb-lst]#
        # Transposed: #row0col7[msb-lsb],row0col6[msb-lst],....,row0col0[msb-lst]#
        for output_row_idx in range(8):  # output_row_idx=0 is row0, output_row_idx=7 is row7
            row_num = output_row_idx  # row0, row1, ..., row7
            if output_row_idx < 2:  # Only write first 2 rows as header format
                header_line = f"#row{row_num}col7[msb-lsb],row{row_num}col6[msb-lst],....,row{row_num}col0[msb-lst]#\n"
                f.write(header_line)
            elif output_row_idx == 2:
                f.write("#................#\n")
        # Write transposed data (row0 to row7, each with col7 to col0)
        for row in transposed:
            f.write(row + '\n')
    
    return True

def main():
    # Paths
    ref_dir = "golden/os2bit/ref"
    output_dir = "golden/os2bit"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all weight files in ref directory
    pattern = os.path.join(ref_dir, "weight*.txt")
    weight_files = glob.glob(pattern)
    
    if not weight_files:
        print(f"No weight files found in {ref_dir}")
        return
    
    print(f"Found {len(weight_files)} weight files to transpose")
    
    success_count = 0
    for input_file in sorted(weight_files):
        # Get filename
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        
        print(f"Processing: {filename}")
        
        if transpose_weight_file(input_file, output_file):
            print(f"  -> Successfully transposed to {output_file}")
            success_count += 1
        else:
            print(f"  -> Failed to transpose {input_file}")
    
    print(f"\nCompleted: {success_count}/{len(weight_files)} files processed successfully")

if __name__ == "__main__":
    main()

