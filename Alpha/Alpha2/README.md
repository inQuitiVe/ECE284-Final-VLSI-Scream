# Alpha 2: MAC Array Components

## Overview

Alpha 2 provides the core MAC array building blocks that form the foundation of the 2D systolic array. These modules implement the hierarchical structure: MAC array → MAC row → MAC tile (PE).

## Features

- **Hierarchical Design**: Three-level hierarchy (array → row → tile)
- **OS/WS Reconfigurable**: Support for both Output Stationary and Weight Stationary modes
- **2-bit/4-bit Reconfigurable**: Switchable activation bit-width
- **Complete MAC Array**: Full 8×8 systolic array implementation

## Directory Structure

```
Alpha2/
├── mac_array.v    # Top-level MAC array (8×8)
├── mac_row.v      # MAC row (8 PEs in a row)
└── mac_tile.v     # MAC tile (single PE)
```

## Module Hierarchy

```
mac_array (8×8)
  └── mac_row[0:7] (8 rows)
      └── mac_tile[0:7] (8 tiles per row)
```

## Module Descriptions

### mac_array.v
Top-level MAC array module that instantiates multiple MAC rows.

**Parameters:**
- `bw`: Bit-width for weights/activations (default: 4)
- `psum_bw`: Bit-width for partial sums (default: 16)
- `col`: Number of columns (default: 8)
- `row`: Number of rows (default: 8)

**Inputs:**
- `clk`, `reset`: Clock and reset signals
- `in_w[row*bw-1:0]`: Weight/activation input
- `in_n[psum_bw*col-1:0]`: Partial sum input (for OS mode)
- `inst_w[2:0]`: Instruction word
  - WS mode: `{reserved, execute, kernel_loading}`
  - OS mode: `{flush_psum, execute, psum_loading}`
- `is_os`: Output Stationary mode flag
- `act_4b_mode`: 4-bit activation mode flag (0=2-bit, 1=4-bit)

**Outputs:**
- `out_s[psum_bw*col-1:0]`: Partial sum output
- `valid[col-1:0]`: Valid signal for each column

### mac_row.v
MAC row module that instantiates multiple MAC tiles in a row.

**Function:**
- Connects tiles horizontally
- Manages data flow between tiles
- Handles instruction propagation

### mac_tile.v
Single Processing Element (PE) - the basic MAC unit.

**Function:**
- Performs multiply-accumulate operation
- Stores weight/activation in local registers
- Manages partial sum flow (upward in WS, downward in OS)

## Dataflow Modes

### Weight Stationary (WS) Mode
- Weights are loaded and stored in each PE
- Activations flow from left to right
- Partial sums flow from top to bottom

### Output Stationary (OS) Mode
- Partial sums are accumulated in each PE
- Weights flow from top to bottom
- Activations flow from left to right
- Different instruction encoding for psum management

## Usage

### Integration

These modules are typically integrated into a larger system:

```verilog
mac_array #(
    .bw(bw),
    .psum_bw(psum_bw),
    .col(8),
    .row(8)
) mac_array_instance (
    .clk(clk),
    .reset(reset),
    .out_s(mac_output),
    .in_w(activation_input),
    .in_n(psum_input),
    .inst_w(instruction),
    .valid(valid_signal),
    .is_os(is_os_mode),
    .act_4b_mode(activation_mode)
);
```

### Standalone Testing

```bash
# Compile and test
iverilog -o test_mac mac_array.v mac_row.v mac_tile.v mac_tb.v
vvp test_mac
```

## Key Design Features

1. **Reconfigurable Dataflow**: Single implementation supports both OS and WS modes
2. **SIMD Support**: 2-bit/4-bit activation mode switching
3. **Scalable Architecture**: Easy to modify array dimensions via parameters
4. **Efficient Data Path**: Optimized for systolic array data flow

## Notes

- These modules form the core of the MAC array used in Part2 and Part3
- The hierarchical design allows for easy modification and optimization
- OS/WS reconfigurability is implemented at the array level

