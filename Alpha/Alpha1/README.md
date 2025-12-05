# Alpha 1: MAC Implementation

## Overview

Alpha 1 contains the initial MAC (Multiply-Accumulate) unit implementations, including both vanilla and Output Stationary (OS) versions. This version explores different MAC architectures and supports 2-bit/4-bit activation mode switching.

## Features

- **Multiple MAC Implementations**: Vanilla and flexible MAC designs
- **2-bit/4-bit Reconfigurable**: Support for switching between 2-bit and 4-bit activation processing
- **OS Mode Support**: Output Stationary MAC implementation
- **Reference Designs**: Hardware reference implementations for comparison

## Directory Structure

```
Alpha1/
└── hardware/
    ├── src/
    │   ├── mac_vanilla.v      # Basic vanilla MAC implementation
    │   ├── mac_flex.v         # Flexible MAC implementation
    │   ├── os/                # Output Stationary MAC implementation
    │   │   ├── mac_array.v    # OS MAC array
    │   │   ├── mac_row.v      # OS MAC row
    │   │   ├── mac_tile.v     # OS MAC tile (PE)
    │   │   └── README.md      # OS implementation notes
    │   └── hw_ref/            # Reference hardware implementations
    │       ├── mac_array.v
    │       ├── mac_row.v
    │       ├── mac_tile.v
    │       └── mac.v
    ├── filelist              # Verilog file list
    └── compiled              # Compiled output
```

## MAC Implementations

### mac_vanilla.v
Basic MAC unit implementation with standard multiply-accumulate operation.

**Parameters:**
- `bw`: Bit-width for weights/activations (default: 4)
- `psum_bw`: Bit-width for partial sums (default: 16)

**Features:**
- Standard MAC operation: `out = a * b + c`
- Supports 2-bit/4-bit activation mode via `act_4b_mode` signal

### mac_flex.v
Flexible MAC implementation with additional configurability.

### OS Implementation (os/)
Output Stationary MAC implementation that explores different dataflow patterns.

**Key Features:**
- Partial sum stationary in MAC array
- Different instruction encoding for OS mode
- Support for both OS and WS modes (switchable via `is_os` signal)

**Implementation Notes:**
See `os/README.md` for detailed discussion on:
- Partial sum flow mechanisms
- Instruction encoding differences
- OS/WS mode switching implementation

## Usage

### Compilation

```bash
cd Alpha/Alpha1/hardware
iverilog -f filelist -o compiled
vvp compiled
```

### Testing Individual MAC Units

```bash
# Test vanilla MAC
iverilog -o test_vanilla mac_vanilla.v mac_tb.v
vvp test_vanilla

# Test OS MAC array
iverilog -o test_os os/mac_array.v os/mac_row.v os/mac_tile.v os_tb.v
vvp test_os
```

## Key Innovations

1. **2-bit/4-bit Reconfigurability**: Early exploration of SIMD reconfiguration
2. **OS Mode Exploration**: Initial implementation of Output Stationary dataflow
3. **Flexible Architecture**: Multiple MAC designs for comparison and optimization

## Reference Designs

The `hw_ref/` directory contains reference implementations that serve as:
- Baseline designs for comparison
- Documentation of standard MAC array structures
- Starting point for further optimizations

## Notes

- Alpha 1 represents early exploration of MAC architectures
- The OS implementation includes detailed notes on design decisions and challenges
- These implementations form the foundation for later parts (Part2, Part3)

