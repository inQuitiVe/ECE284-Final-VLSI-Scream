#!/usr/bin/env python3
"""
Verify transform_activation_2bit.py results
"""

def verify_transformation():
    """Verify the transformation logic"""
    
    # Read original data
    with open("golden/os2bit/ref/activation_tile0.txt", 'r') as f:
        lines = f.readlines()
    
    data_lines = [line.strip() for line in lines[3:] if line.strip()]
    
    # Parse original data
    data = []
    for time_idx, line in enumerate(data_lines):
        if len(line) != 16:
            continue
        row_data = []
        for i in range(8):
            start = i * 2
            end = start + 2
            row_data.append(line[start:end])
        data.append(row_data)
    
    # Read transformed data
    with open("golden/os2bit/activation_tile0.txt", 'r') as f:
        trans_lines = f.readlines()
    
    trans_data_lines = [line.strip() for line in trans_lines[3:] if line.strip()]
    
    # Verify transformation
    def get_time_indices(new_row_idx):
        if new_row_idx == 0:
            start = 0
        elif new_row_idx == 1:
            start = 1
        elif new_row_idx == 2:
            start = 2
        elif new_row_idx == 3:
            start = 6
        elif new_row_idx == 4:
            start = 7
        elif new_row_idx == 5:
            start = 8
        elif new_row_idx == 6:
            start = 12
        elif new_row_idx == 7:
            start = 13
        elif new_row_idx == 8:
            start = 14
        else:
            start = 0
        return [start, start+1, start+2, start+3, start+6, start+7, start+8, start+9]
    
    print("Verifying transformation...")
    print("=" * 80)
    
    errors = []
    row_idx = 0
    
    for orig_row_idx in range(7, -1, -1):  # row0 to row7
        for new_row_idx in range(9):
            time_indices = get_time_indices(new_row_idx)
            expected_row = []
            
            # Build expected row (MSB to LSB, so reverse time_indices)
            for time_idx in reversed(time_indices):
                if time_idx < len(data):
                    expected_row.append(data[time_idx][orig_row_idx])
                else:
                    expected_row.append("00")
            
            expected = ''.join(expected_row)
            actual = trans_data_lines[row_idx] if row_idx < len(trans_data_lines) else ""
            
            if expected != actual:
                errors.append({
                    'orig_row': orig_row_idx,
                    'new_row': new_row_idx,
                    'row_idx': row_idx,
                    'expected': expected,
                    'actual': actual
                })
            
            row_idx += 1
    
    if errors:
        print(f"Found {len(errors)} errors:")
        for err in errors[:10]:  # Show first 10 errors
            print(f"\nRow {err['row_idx']} (orig_row={err['orig_row']}, new_row={err['new_row']}):")
            print(f"  Expected: {err['expected']}")
            print(f"  Actual:   {err['actual']}")
    else:
        print("All transformations verified correctly!")
    
    print(f"\nTotal rows checked: {row_idx}")
    print(f"Errors found: {len(errors)}")

if __name__ == "__main__":
    verify_transformation()

