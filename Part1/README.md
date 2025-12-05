# Part 1: Vanilla Version

## Overview

Part 1 is the basic implementation of the 2D systolic array-based AI accelerator. This version provides the foundational architecture with standard 4-bit activation processing and Weight Stationary (WS) dataflow.

## Features

- **Basic 2D Systolic Array**: 8×8 MAC array implementation
- **Weight Stationary Dataflow**: Fixed WS mode
- **4-bit Activation Processing**: Standard 4-bit activation and weight bit-width
- **ReLU Activation**: Standard ReLU activation function in SFU
- **Complete Pipeline**: MAC array → SFU → Output

## Directory Structure

```
Part1/
├── hardware/
│   ├── src/
│   │   ├── core_tb.v      # Testbench
│   │   ├── core.v         # Top-level core module
│   │   └── SFU/           # Summation and Function Unit
│   │       ├── SFU.v      # SFU main module
│   │       ├── ReLU.v     # ReLU activation function
│   │       └── onij_calculator.v  # Output address calculator
│   └── golden/            # Test data files
│       ├── out.txt        # Expected output
│       └── viz/           # Human-readable format
│           └── viz_out.txt
└── software/
    └── Part1_golden_gen.ipynb  # Golden pattern generator
```

## Hardware Components

### Core Module
- **Input**: Activation and weight data through X_MEM
- **Processing**: 2D systolic array computation
- **Output**: Partial sums processed through SFU with ReLU

### SFU (Summation and Function Unit)
- Accumulates partial sums from MAC array
- Applies ReLU activation function
- Manages PSUM memory read/write operations

## Golden Data

- **`out.txt`**: Expected output in binary format
- **`viz/viz_out.txt`**: Human-readable decimal format for verification

## Usage

### Compilation

```bash
cd Part1/hardware/src
iverilog -o compiled core_tb.v core.v SFU/*.v
vvp compiled
```

### Simulation

The testbench (`core_tb.v`) will:
1. Load activation and weight data
2. Execute the convolution operation
3. Compare output with golden data
4. Report any mismatches

## Notes

- This is the baseline version without reconfiguration features
- All subsequent parts (Part2, Part3) extend this basic implementation
- Used as reference for understanding the core architecture

