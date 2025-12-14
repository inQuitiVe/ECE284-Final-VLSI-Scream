# Alpha 1: MAC Implementation

[![Status](https://img.shields.io/badge/status-stable-green.svg)]()
[![Hardware](https://img.shields.io/badge/hardware-FPGA-blue.svg)]()
[![SIMD](https://img.shields.io/badge/SIMD-2b%2F4b-orange.svg)]()
[![Dataflow](https://img.shields.io/badge/dataflow-OS%2FWS-purple.svg)]()

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Hardware Configuration](#hardware-configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [References](#references)

## Overview

Alpha 1 provides the foundational MAC (Multiply-Accumulate) unit implementations for the 2D systolic array accelerator. This version explores different MAC architectures including vanilla and Output Stationary (OS) implementations, with support for 2-bit/4-bit SIMD reconfiguration and OS/WS dataflow modes.

### Key Innovations

- **Multiple MAC Architectures**: Vanilla and flexible MAC designs for comparison
- **SIMD Reconfigurability**: Switchable 2-bit and 4-bit activation processing
- **Dataflow Flexibility**: Support for both Weight Stationary (WS) and Output Stationary (OS) modes
- **Modular Design**: Hierarchical structure (array → row → tile) for scalability

## Features

- ✅ Vanilla MAC implementation
- ✅ Flexible MAC implementation
- ✅ Output Stationary (OS) MAC implementation
- ✅ **2-bit/4-bit activation mode switching** (bit-width refers to activation data)
  - 2-bit mode: Parallel processing of two activation-weight pairs per MAC unit
  - Power and area efficient design through SIMD parallelism
- ✅ OS/WS dataflow mode switching
- ✅ Reference hardware implementations
- ✅ Complete test infrastructure

## Architecture

### MAC Hierarchy

```
mac_array (8×8)
  └── mac_row[0:7] (8 rows)
      └── mac_tile[0:7] (8 tiles per row)
          └── mac (single PE)
```

### MAC Tile (Processing Element)

Each MAC tile performs:
- **Multiply-Accumulate**: `psum = activation × weight + psum`
- **Weight Storage**: Stores weights locally in WS mode
- **Partial Sum Accumulation**: Accumulates psums in OS mode
- **Data Flow Control**: Manages activation and weight flow

### Dataflow Modes

#### Weight Stationary (WS) Mode
- Weights are loaded and stored in each PE
- Activations flow from left to right
- Partial sums flow from top to bottom
- Efficient for weight reuse scenarios

#### Output Stationary (OS) Mode
- Partial sums are accumulated in each PE
- Weights flow from top to bottom
- Activations flow from left to right
- Different instruction encoding for psum management

### SIMD Modes

The bit-width modes (2-bit and 4-bit) refer specifically to the **activation** data bit-width. This SIMD (Single Instruction, Multiple Data) reconfiguration enables flexible precision processing with significant power and area benefits.

#### 4-bit Mode (Default)
- **Activation bit-width**: 4 bits per activation value
- Standard 4-bit activation and weight processing
- Full precision MAC operations
- Each MAC unit processes one activation-weight pair per cycle

#### 2-bit Mode
- **Activation bit-width**: 2 bits per activation value
- **Parallel Processing**: In 2-bit mode, the hardware processes **two activation-weight pairs in parallel** within the same MAC unit
- **Power and Area Benefits**: 
  - **Reduced Power Consumption**: Processing two 2-bit operations in parallel is more power-efficient than sequential 4-bit operations
  - **Area Efficiency**: The same MAC hardware can handle two parallel 2-bit multiplications, effectively doubling throughput without doubling area
  - **Higher Throughput**: Two independent activation-weight pairs are computed simultaneously, improving overall processing speed
- **Data Organization**: 
  - Requires interleaved data loading from two activation tiles (tile0 and tile1)
  - Activations are interleaved with tile1 in MSB position, processed in 2-bit steps
  - Supports tile-based activation processing for efficient data management
- **Implementation Details**:
  - Two activation tiles are loaded and processed simultaneously
  - Each MAC unit performs two 2-bit multiplications in parallel
  - Results are accumulated separately for each activation-weight pair

## Prerequisites

### Software Requirements

- **Icarus Verilog**: 10.0+ (for simulation)
- **GTKWave**: (optional, for waveform viewing)
- **Make**: (for build automation)
- **Python**: 3.7+ (for data transformation scripts)

### Hardware Requirements

- **FPGA**: Cyclone IV GX (or compatible)
- **Development Board**: With appropriate I/O interfaces

### Dependencies

```bash
# Verilog tools (Ubuntu/Debian)
sudo apt-get install iverilog gtkwave

# Verilog tools (macOS)
brew install icarus-verilog gtkwave

# Python packages
pip install numpy
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ECE284-Final-VLSI-Scream/Alpha/Alpha1
```

### 2. Verify Installation

```bash
# Check Verilog compiler
iverilog -v

# Check Make
make --version
```

### 3. Verify Source Files

```bash
# Check that source files exist
ls -la hardware/src/Mac/
ls -la hardware/src/SFU/
```

## Quick Start

If you have the test data already prepared, you can directly run simulations:

```bash
cd hardware

# Run vanilla mode (4-bit, WS)
make vanilla

# Run 2-bit mode (2-bit, WS)
make act_2b

# Run OS vanilla mode (4-bit, OS)
make os_vanilla

# Run OS 2-bit mode (2-bit, OS)
make os_2b

# Run all modes
make all
```

### View Results

```bash
# View waveform
make view

# Check output files
cat hardware/golden/ws4bit/out.txt        # For ws4bit mode
cat hardware/golden/ws2bit/output.txt     # For ws2bit mode
```

> **Note**: If you need to prepare your own test data, see [Data Preparation](#data-preparation) section below.

## Project Structure

```
Alpha1/
├── README.md                 # This file
│
├── hardware/                 # Hardware implementation
│   ├── Makefile             # Build configuration
│   ├── filelist             # WS mode file list
│   ├── filelist_os          # OS mode file list
│   │
│   ├── src/                 # Verilog source code
│   │   ├── core_tb.v       # WS mode testbench
│   │   ├── core_tb_os.v    # OS mode testbench
│   │   ├── core.v          # Top-level core module
│   │   ├── corelet.v       # Corelet (MAC array + FIFOs)
│   │   ├── mac_vanilla.v   # Vanilla MAC implementation
│   │   ├── mac_flex.v      # Flexible MAC implementation
│   │   │
│   │   ├── Mac/            # MAC array modules
│   │   │   ├── mac_array.v # MAC array top-level
│   │   │   ├── mac_row.v   # MAC row
│   │   │   ├── mac_tile.v  # MAC tile (PE)
│   │   │   └── mac.v       # Basic MAC unit
│   │   │
│   │   ├── os/             # Output Stationary implementation
│   │   │   ├── mac_array.v # OS MAC array
│   │   │   ├── mac_row.v   # OS MAC row
│   │   │   ├── mac_tile.v  # OS MAC tile
│   │   │   └── README.md   # OS implementation notes
│   │   │
│   │   ├── hw_ref/         # Reference hardware implementations
│   │   │   ├── mac_array.v
│   │   │   ├── mac_row.v
│   │   │   ├── mac_tile.v
│   │   │   └── mac.v
│   │   │
│   │   ├── SFU/            # Summation and Function Unit
│   │   │   ├── SFU.v       # SFU main module
│   │   │   ├── ReLU.v      # ReLU activation
│   │   │   └── onij_calculator.v  # Output address calculator
│   │   │
│   │   ├── FIFO/           # FIFO modules
│   │   │   ├── l0.v        # L0 FIFO
│   │   │   ├── ififo.v     # Input FIFO (OS mode)
│   │   │   ├── ofifo.v     # Output FIFO
│   │   │   ├── fifo_depth64.v  # Base FIFO
│   │   │   ├── fifo_mux_2_1.v  # 2-to-1 FIFO mux
│   │   │   ├── fifo_mux_8_1.v  # 8-to-1 FIFO mux
│   │   │   └── fifo_mux_16_1.v # 16-to-1 FIFO mux
│   │   │
│   │   └── SRAM/           # SRAM modules
│   │       ├── sram_32b_w2048.v   # X_MEM
│   │       ├── sram_128b_w16_RW.v # PSUM_MEM
│   │       └── sram_64b_w256.v    # Additional SRAM
│   │
│   ├── golden/             # Test data
│   │   ├── ws4bit/         # WS mode, 4-bit data
│   │   │   ├── activation_tile0.txt
│   │   │   ├── weight_itile0_otile0_kij*.txt
│   │   │   ├── out.txt, out_raw.txt
│   │   │   └── viz/        # Visualization files
│   │   │
│   │   ├── ws2bit/         # WS mode, 2-bit data
│   │   │   ├── activation_tile0.txt, activation_tile1.txt
│   │   │   ├── weight_itile*_otile*_kij*.txt
│   │   │   ├── output.txt
│   │   │   ├── viz/        # Visualization files
│   │   │   └── *.py        # Analysis scripts
│   │   │
│   │   ├── os4bit/         # OS mode, 4-bit data
│   │   │   ├── activation_tile0.txt
│   │   │   ├── weight_itile0_otile0_kij*.txt
│   │   │   ├── out.txt, out_raw.txt
│   │   │   ├── ref/        # Reference files
│   │   │   ├── viz/        # Visualization files
│   │   │   └── psum_analysis/  # PSUM analysis
│   │   │
│   │   └── os2bit/         # OS mode, 2-bit data
│   │       ├── activation_tile0.txt, activation_tile1.txt
│   │       ├── weight_itile*_otile*_kij*.txt
│   │       ├── expected_output*.txt
│   │       ├── ref/        # Reference files
│   │       └── viz/        # Visualization files
│   │
│   └── *.py                # Data transformation scripts (in hardware/)
│       ├── transform_activation.py      # 4-bit activation transform
│       ├── transform_activation_2bit.py # 2-bit activation transform
│       ├── transpose_weights.py         # Weight transpose
│       ├── transpose_weights_2bit.py    # 2-bit weight transpose
│       ├── verify_transform_2bit.py     # Verification script
│       └── compare_transforms.py        # Compare transform algorithms
```

## Usage

### Basic Usage

The hardware supports four main configurations:

```bash
cd hardware

# Run vanilla mode (4-bit, WS)
make vanilla

# Run 2-bit mode (2-bit, WS)
make act_2b

# Run OS vanilla mode (4-bit, OS)
make os_vanilla

# Run OS 2-bit mode (2-bit, OS)
make os_2b

# Run all modes
make all
```

### Makefile Targets

| Target | Description |
|--------|-------------|
| `vanilla` | 4-bit activation, Weight Stationary mode |
| `act_2b` | 2-bit activation, Weight Stationary mode |
| `os_vanilla` | 4-bit activation, Output Stationary mode |
| `os_2b` | 2-bit activation, Output Stationary mode |
| `all` | Run all test modes sequentially |
| `clean` | Remove compiled files and VCD files |
| `view` | View waveform with GTKWave (checks for core_tb.vcd) |
| `help` | Show all available targets |

### Advanced Usage

#### Custom Compilation

```bash
# Compile with custom flags
iverilog -DACT_2BIT -DIS_OS -f filelist_os -o compiled
vvp compiled
```

#### Waveform Viewing

```bash
# Generate VCD and view
make vanilla
make view

# Or manually (VCD filename depends on mode)
# Note: VCD files are generated in hardware/ directory
cd hardware
gtkwave core_tb_vanilla.vcd    # For vanilla mode
gtkwave core_tb_2bit.vcd       # For 2-bit mode
gtkwave core_tb_os_vanilla.vcd # For OS vanilla mode
gtkwave core_tb_os_2bit.vcd    # For OS 2-bit mode

# Note: Makefile's 'view' target checks for core_tb.vcd which may not exist
# Use the specific VCD filename based on the mode you ran
```

#### Debug Mode

Enable verbose output in testbench:

```verilog
// In core_tb.v, uncomment debug prints
`define DEBUG 1
```

## Data Preparation

This section describes how to prepare your own test data if you want to use custom inputs.

### Data Format Specifications

#### Activation File Format

**Important**: The bit-width (2-bit or 4-bit) refers to the **activation** data bit-width. In 2-bit mode, the hardware processes two activation-weight pairs in parallel for improved power and area efficiency.

**4-bit Mode:**
- **Activation bit-width**: 4 bits per activation value
- Each line: 32 bits = 8 × 4-bit activation values
- Format: `timeXrow7[msb-lsb], timeXrow6[msb-lsb], ..., timeXrow0[msb-lsb]`
- 36 time steps (nij = 0 to 35)
- Each MAC unit processes one 4-bit activation-weight pair per cycle

**2-bit Mode:**
- **Activation bit-width**: 2 bits per activation value
- **Parallel Processing**: Two activation tiles (tile0 and tile1) are processed **in parallel**
  - Each MAC unit simultaneously processes two 2-bit activation-weight pairs
  - This parallel operation doubles throughput while reducing power consumption
  - Area overhead is minimal as the same MAC hardware handles both operations
- **Data Organization**:
  - Requires two activation tiles (tile0 and tile1)
  - Each tile: 16 bits = 8 × 2-bit activation values per line
  - Tiles are interleaved with tile1 in MSB position
  - Data is processed in 2-bit steps
  - Two weight files are loaded sequentially (w1 from tile1 first, then w from tile0)

#### Weight File Format

**File Naming Convention:**
- Format: `weight_itile<itile>_otile<otile>_kij<kij>.txt`
- `itile`: Input tile index (0, 1)
- `otile`: Output tile index (0, 1)
- `kij`: Kernel iteration (0-8)

**4-bit Mode:**
- Each line: 32 bits = 8 × 4-bit values (one column)
- 8 lines per file (8 columns)
- Format: `colXrow7[msb-lsb], colXrow6[msb-lsb], ..., colXrow0[msb-lsb]`

**2-bit Mode:**
- Similar structure but with 2-bit values
- 16 bits per line = 8 × 2-bit values

#### Output File Format

- Binary format: 128 bits per line (8 channels × 16 bits)
- Each channel represents one output feature (mac_col = 8)
- Each channel is 16 bits (psum_bw = 16)
- Data is ordered from col7 to col0 (MSB to LSB)
- Format: `timeXcol7[msb-lsb], timeXcol6[msb-lsb], ..., timeXcol0[msb-lsb]`

### Step 1: Prepare Activation Data

#### For 4-bit Mode (OS)

```bash
cd hardware
python transform_activation.py
```

**What it does:**
- Transforms activation data from `golden/os4bit/ref/` to `golden/os4bit/`
- Converts from time-row format to row-time format for OS mode
- Reorganizes data according to OS mode requirements
- **Note**: This script is for OS mode only, uses hardcoded paths
- Automatically processes all activation files in the ref directory

#### For 2-bit Mode (OS)

```bash
cd hardware
python transform_activation_2bit.py
```

**What it does:**
- Transforms activation data from `golden/os2bit/ref/` to `golden/os2bit/`
- Interleaves two activation tiles with tile1 in MSB
- Processes data in 2-bit steps
- **Note**: This script is for OS mode only, uses hardcoded paths
- Automatically processes all activation files in the ref directory

### Step 2: Prepare Weight Data

#### For 4-bit Mode (OS)

```bash
cd hardware
python transpose_weights.py
```

**What it does:**
- Transposes weight files from `golden/os4bit/ref/` to `golden/os4bit/`
- Converts weight matrices from col-row format to row-col format
- Required for OS mode dataflow
- Uses hardcoded paths, automatically processes all weight files

#### For 2-bit Mode (OS)

```bash
cd hardware
python transpose_weights_2bit.py
```

**What it does:**
- Transposes weight files from `golden/os2bit/ref/` to `golden/os2bit/`
- Similar to 4-bit transpose but handles 2-bit values
- Processes tile-based weight organization
- Uses hardcoded paths, automatically processes all weight files

### Step 3: Verify Data Format

```bash
cd hardware

# Verify 2-bit transformation (for OS mode)
python verify_transform_2bit.py

# Compare transformation algorithms
python compare_transforms.py
```

**Note**: `verify_transform_2bit.py` verifies OS 2-bit transformations using hardcoded paths (`golden/os2bit/ref/` and `golden/os2bit/`).

### Step 4: Generate Visualization Files

Binary data files are not human-readable. Use the `viz.py` scripts to convert them to decimal format for debugging and verification.

#### For 4-bit Mode

```bash
# WS mode
cd hardware/golden/ws4bit
python viz.py

# OS mode
cd hardware/golden/os4bit
python viz.py
```

#### For 2-bit Mode

```bash
# WS mode
cd hardware/golden/ws2bit
python viz.py

# OS mode
cd hardware/golden/os2bit
python viz.py
```

**What it does:**
- Converts binary data files to human-readable decimal format
- Creates `viz/` directory with converted files
- **Activation files**: Converts 4-bit/2-bit unsigned values to decimal (0-15 for 4-bit, 0-3 for 2-bit)
- **Weight files**: Converts 4-bit signed values to decimal (-8 to 7)
- **Output files**: Converts 16-bit signed values to decimal (-32768 to 32767)

**Generated files:**
- `viz/viz_activation_tile*.txt`: Human-readable activation values
- `viz/viz_weight_*.txt`: Human-readable weight values
- `viz/viz_out.txt`: Human-readable output values (after ReLU)
- `viz/viz_out_raw.txt`: Human-readable raw output values (before ReLU)

**Example usage:**
```bash
# View converted activation data
cat hardware/golden/ws4bit/viz/viz_activation_tile0.txt

# View converted output
cat hardware/golden/ws4bit/viz/viz_out.txt
```

### Python Scripts Reference

All data transformation scripts are located in `hardware/`:

| Script | Location | Purpose | Input Path | Output Path | Notes |
|--------|----------|---------|------------|-------------|-------|
| `transform_activation.py` | `hardware/` | Transform 4-bit activations for OS mode | `golden/os4bit/ref/` | `golden/os4bit/` | Hardcoded paths, no args |
| `transform_activation_2bit.py` | `hardware/` | Transform 2-bit activations for OS mode | `golden/os2bit/ref/` | `golden/os2bit/` | Hardcoded paths, no args |
| `transpose_weights.py` | `hardware/` | Transpose 4-bit weights for OS mode | `golden/os4bit/ref/` | `golden/os4bit/` | Hardcoded paths, no args |
| `transpose_weights_2bit.py` | `hardware/` | Transpose 2-bit weights for OS mode | `golden/os2bit/ref/` | `golden/os2bit/` | Hardcoded paths, no args |
| `verify_transform_2bit.py` | `hardware/` | Verify 2-bit OS transformation | `golden/os2bit/` | Console output | Hardcoded paths, no args |
| `compare_transforms.py` | `hardware/` | Compare transform algorithms | N/A | Console output | No arguments needed |
| `viz.py` | `golden/*/` | Convert binary to decimal format | Current directory | `viz/` | Run from golden subdirectory |

### Pre-compilation Data Preparation Workflow

Complete workflow for preparing custom test data:

```bash
cd hardware

# 1. Prepare 4-bit OS mode data
# (Must be run from hardware/ directory)
python transform_activation.py
python transpose_weights.py

# 2. Prepare 2-bit OS mode data
# (Must be run from hardware/ directory)
python transform_activation_2bit.py
python transpose_weights_2bit.py

# 3. Verify transformations
# (Must be run from hardware/ directory)
python verify_transform_2bit.py
python compare_transforms.py

# Note: All scripts use hardcoded paths relative to hardware/ directory
# WS mode data preparation scripts are in golden/ws2bit/ directory

# 4. Run simulations
make vanilla
make act_2b
make os_vanilla
make os_2b
```

## Hardware Configuration

### Configuration Modes

The hardware supports four main configurations:

| Mode | Activation Bits | Dataflow | Makefile Target |
|------|----------------|----------|-----------------|
| Vanilla | 4-bit | WS | `vanilla` |
| 2-bit | 2-bit | WS | `act_2b` |
| OS Vanilla | 4-bit | OS | `os_vanilla` |
| OS 2-bit | 2-bit | OS | `os_2b` |

### Compilation Flags

The Makefile uses the following flags:

- **`-DACT_2BIT`**: Enable 2-bit activation mode
- **`-DIS_OS`**: Enable Output Stationary dataflow mode

### MAC Array Parameters

Default parameters (configurable in source):

```verilog
parameter bw = 4;        // Bit-width for weights/activations
parameter psum_bw = 16;  // Bit-width for partial sums
parameter col = 8;       // Number of columns
parameter row = 8;       // Number of rows
```

### Activation Bit-Width and Parallel Processing

**Important Note on Bit-Width Modes:**

The 2-bit and 4-bit modes specifically refer to the **activation** data bit-width, not the weight bit-width. This design choice enables significant hardware efficiency improvements:

- **4-bit Mode**: Each MAC unit processes one 4-bit activation with one weight per cycle
- **2-bit Mode**: Each MAC unit processes **two 2-bit activations with two weights in parallel** per cycle
  - This parallel processing effectively doubles the throughput
  - Power consumption is reduced compared to sequential 4-bit operations
  - Area overhead is minimal since the same MAC hardware handles both parallel operations
  - The parallel act-weight pairs are processed simultaneously, maximizing hardware utilization

This SIMD approach allows the accelerator to adapt to different precision requirements while maintaining optimal power and area efficiency.

## Testing

### Test Data Organization

Test data is organized by mode and bit-width:

```
golden/
├── ws4bit/          # Weight Stationary, 4-bit
│   ├── activation_tile0.txt
│   ├── weight_itile0_otile0_kij*.txt
│   ├── out.txt              # Output (after ReLU)
│   └── out_raw.txt          # Raw output (before ReLU)
├── ws2bit/          # Weight Stationary, 2-bit
│   ├── activation_tile0.txt, activation_tile1.txt
│   ├── weight_itile*_otile*_kij*.txt
│   └── output.txt           # Output file
├── os4bit/          # Output Stationary, 4-bit
│   ├── activation_tile0.txt
│   ├── weight_itile0_otile0_kij*.txt
│   ├── out.txt, out_raw.txt
│   └── ...
└── os2bit/          # Output Stationary, 2-bit
    ├── activation_tile0.txt, activation_tile1.txt
    ├── weight_itile*_otile*_kij*.txt
    └── expected_output*.txt
```

### Running Tests

```bash
# Run specific test mode
make vanilla

# Verify output (testbench automatically compares)
# For ws4bit: compares against hardware/golden/ws4bit/out.txt
# For ws2bit: compares against hardware/golden/ws2bit/expected_output_from_psum_binary.txt
# For os4bit: compares against hardware/golden/os4bit/out.txt
# For os2bit: compares against hardware/golden/os2bit/expected_output_from_psum_binary.txt
```

### Expected Output Format

Output files contain:
- Binary format: 128 bits per line (8 channels × 16 bits)
- Each channel represents one output feature (mac_col = 8)
- Each channel is 16 bits (psum_bw = 16)
- Data is ordered from col7 to col0 (MSB to LSB)
- Format: `timeXcol7[msb-lsb], timeXcol6[msb-lsb], ..., timeXcol0[msb-lsb]`

### Viewing Results with Visualization

For easier debugging and verification, use the visualization scripts to convert binary outputs to human-readable format:

```bash
# Generate visualization files
cd hardware/golden/ws4bit
python viz.py

# View human-readable output
cat viz/viz_out.txt
```

The visualization files show:
- **Activation values**: Decimal representation of input activations
- **Weight values**: Decimal representation of weights (signed integers)
- **Output values**: Decimal representation of final outputs (16-bit signed integers)
- **Raw outputs**: Output values before ReLU activation

This makes it much easier to:
- Debug data loading issues
- Verify intermediate calculations
- Compare expected vs actual outputs
- Understand data flow through the accelerator

## Troubleshooting

### Common Issues

#### Issue: Compilation errors

**Solution:**
```bash
# Clean and rebuild
cd hardware
make clean
# Note: clean only removes core_tb.vcd, but actual VCD files have mode-specific names
# You may need to manually remove: core_tb_vanilla.vcd, core_tb_2bit.vcd, etc.
make vanilla

# Check filelist
cat hardware/filelist

# Verify source files exist
ls -la hardware/src/Mac/
```

#### Issue: Simulation fails

**Solution:**
```bash
# Check testbench syntax
cd hardware
iverilog -t null src/core_tb.v

# Verify golden data exists
ls -la hardware/golden/ws4bit/

# Check file paths in testbench
grep "golden" hardware/src/core_tb.v
```

#### Issue: Output mismatch

**Solution:**
```bash
# Verify input data format (for OS 2-bit mode)
cd hardware
python verify_transform_2bit.py
# Note: This script verifies OS mode transformations only

# Check weight file format
head -n 5 hardware/golden/ws4bit/weight_itile0_otile0_kij0.txt

# Compare with expected output (check testbench output)
# The testbench will automatically compare and report mismatches
```

#### Issue: OS mode not working

**Solution:**
```bash
# Verify OS filelist
cd hardware
cat filelist_os

# Check OS-specific modules
ls -la hardware/src/FIFO/ififo.v

# Verify OS testbench
iverilog -DIS_OS -f filelist_os -o test_os
```

### Debug Tips

1. **Enable Verbose Output**: Uncomment `$display` statements in testbench
2. **Check Waveforms**: Use `make view` to inspect signal timing
3. **Verify Data Loading**: Check X_MEM write operations in waveform
4. **Inspect MAC Operations**: Monitor `mac_out` signals in MAC tiles

## Contributing

We welcome contributions! Please follow these guidelines:

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes
# Test thoroughly
make all

# Submit pull request
```

### Code Style

- Use 2-space indentation
- Follow Verilog naming conventions
- Add comments for complex logic
- Write tests for new features

### Testing Requirements

- All new features must pass existing tests
- Add tests for new functionality
- Update documentation as needed

## References

### Related Alpha Versions

- **Alpha 2**: MAC Array Components (hierarchical structure)
- **Alpha 4**: Whole Conv Layer (complete pipeline)
- **Alpha 6**: Flexible Activation Functions
- **Alpha 7**: SFU FSM Implementation

### Technical Documentation

- **OS Implementation Notes**: See `hardware/src/os/README.md` for OS mode details
- **MAC Architecture**: See `hardware/src/Mac/` for implementation details
- **Data Format**: See `golden/` for data format specifications

### External Resources

- [Systolic Array Architecture](https://en.wikipedia.org/wiki/Systolic_array)
- [Icarus Verilog Documentation](http://iverilog.wikia.com/)
- [Verilog Coding Standards](https://www.verilog.com/)

## License

Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department  
Please do not spread this code without permission

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{alpha1_mac,
  title={MAC Implementation for Reconfigurable 2D Systolic Array Accelerator},
  author={VVIP Lab, UCSD},
  year={2024},
  note={Alpha 1 of ECE284 Final Project}
}
```

---

**Status**: Stable  
**Last Updated**: 2024  
**Maintainer**: VVIP Lab @ UCSD
