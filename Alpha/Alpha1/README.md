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
- ✅ 2-bit/4-bit activation mode switching
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

#### 4-bit Mode (Default)
- Standard 4-bit activation and weight processing
- Full precision MAC operations

#### 2-bit Mode
- 2-bit activation processing for higher throughput
- Requires interleaved data loading
- Supports tile-based activation processing

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

### 3. Build Reference Implementations

```bash
cd hardware/src/hw_ref
iverilog -o test_mac mac.v mac_tb.v
vvp test_mac
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
cat golden/ws4bit/output.txt
```

> **Note**: If you need to prepare your own test data, see [Data Preparation](#data-preparation) section below.

## Project Structure

```
Alpha1/
├── README.md                 # This file
├── Makefile                  # Top-level Makefile
├── filelist                  # Verilog source file list
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
│   │   │
│   │   ├── Mac/            # MAC array modules
│   │   │   ├── mac_array.v # MAC array top-level
│   │   │   ├── mac_row.v   # MAC row
│   │   │   ├── mac_tile.v  # MAC tile (PE)
│   │   │   └── mac.v       # Basic MAC unit
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
│   │   │   └── fifo_depth64.v  # Base FIFO
│   │   │
│   │   └── SRAM/           # SRAM modules
│   │       ├── sram_32b_w2048.v   # X_MEM
│   │       ├── sram_128b_w16_RW.v # PSUM_MEM
│   │       └── sram_64b_w256.v    # Additional SRAM
│   │
│   ├── golden/             # Test data
│   │   ├── ws4bit/         # WS mode, 4-bit data
│   │   ├── ws2bit/         # WS mode, 2-bit data
│   │   ├── os4bit/         # OS mode, 4-bit data
│   │   └── os2bit/         # OS mode, 2-bit data
│   │
│   └── scripts/            # Data transformation scripts
│       ├── transform_activation.py      # 4-bit activation transform
│       ├── transform_activation_2bit.py # 2-bit activation transform
│       ├── transpose_weights.py         # Weight transpose
│       ├── transpose_weights_2bit.py    # 2-bit weight transpose
│       └── verify_transform_2bit.py     # Verification script
│
└── golden/                 # Additional test data
    ├── ws2bit/             # WS 2-bit test data
    ├── ws4bit/             # WS 4-bit test data
    ├── os2bit/             # OS 2-bit test data
    └── os4bit/             # OS 4-bit test data
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
| `clean` | Remove compiled files and VCD |
| `view` | View waveform with GTKWave |
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

# Or manually
gtkwave core_tb_vanilla.vcd
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

**4-bit Mode:**
- Each line: 32 bits = 8 × 4-bit values
- Format: `timeXrow7[msb-lsb], timeXrow6[msb-lsb], ..., timeXrow0[msb-lsb]`
- 36 time steps (nij = 0 to 35)

**2-bit Mode:**
- Requires two activation tiles (tile0 and tile1)
- Each tile: 16 bits = 8 × 2-bit values per line
- Tiles are interleaved with tile1 in MSB position
- Data is processed in 2-bit steps

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

- Binary format: 256 bits per line (16 channels × 16 bits)
- Each channel represents one output feature
- Data is ordered from row7 to row0 (MSB to LSB)

### Step 1: Prepare Activation Data

#### For 4-bit Mode (WS)

```bash
cd hardware
python transform_activation.py \
    --input golden/os4bit/ref/activation_tile0.txt \
    --output golden/os4bit/activation_tile0.txt
```

**What it does:**
- Transforms activation data from time-row format to row-time format for OS mode
- Reorganizes data according to OS mode requirements
- See `transform_activation.py` for detailed transformation algorithm

#### For 2-bit Mode (WS)

```bash
python transform_activation_2bit.py \
    --input_tile0 golden/ws2bit/activation_tile0.txt \
    --input_tile1 golden/ws2bit/activation_tile1.txt \
    --output golden/ws2bit/activation_interleaved.txt
```

**What it does:**
- Interleaves two activation tiles with tile1 in MSB
- Processes data in 2-bit steps
- See `transform_activation_2bit.py` for detailed algorithm

### Step 2: Prepare Weight Data

#### For 4-bit Mode (OS)

```bash
python transpose_weights.py \
    --input_dir golden/os4bit/ref/ \
    --output_dir golden/os4bit/
```

**What it does:**
- Transposes weight matrices from col-row format to row-col format
- Required for OS mode dataflow
- See `transpose_weights.py` for detailed transformation

#### For 2-bit Mode

```bash
python transpose_weights_2bit.py \
    --input_dir golden/ws2bit/from_yufan/ \
    --output_dir golden/ws2bit/
```

**What it does:**
- Similar to 4-bit transpose but handles 2-bit values
- Processes tile-based weight organization
- See `transpose_weights_2bit.py` for details

### Step 3: Verify Data Format

```bash
# Verify 2-bit transformation
python verify_transform_2bit.py \
    --input golden/ws2bit/activation_interleaved.txt

# Compare transformation algorithms
python compare_transforms.py
```

### Python Scripts Reference

All data transformation scripts are located in `hardware/`:

| Script | Purpose | Input Format | Output Format |
|--------|---------|--------------|---------------|
| `transform_activation.py` | Transform 4-bit activations for OS mode | Time-row format | Row-time format |
| `transform_activation_2bit.py` | Interleave 2-bit activation tiles | Two tile files | Interleaved file |
| `transpose_weights.py` | Transpose 4-bit weights for OS mode | Col-row format | Row-col format |
| `transpose_weights_2bit.py` | Transpose 2-bit weights | Col-row format | Row-col format |
| `verify_transform_2bit.py` | Verify 2-bit transformation | Interleaved file | Validation report |
| `compare_transforms.py` | Compare transform algorithms | N/A | Comparison report |

### Pre-compilation Data Preparation Workflow

Complete workflow for preparing custom test data:

```bash
cd hardware

# 1. Prepare 4-bit OS mode data
python transform_activation.py
python transpose_weights.py

# 2. Prepare 2-bit WS mode data
python transform_activation_2bit.py
python transpose_weights_2bit.py

# 3. Verify transformations
python verify_transform_2bit.py
python compare_transforms.py

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

## Testing

### Test Data Organization

Test data is organized by mode and bit-width:

```
golden/
├── ws4bit/          # Weight Stationary, 4-bit
│   ├── activation_tile0.txt
│   ├── weight_itile0_otile0_kij*.txt
│   └── output.txt
├── ws2bit/          # Weight Stationary, 2-bit
│   ├── activation_tile0.txt
│   ├── activation_tile1.txt
│   ├── weight_itile*_otile*_kij*.txt
│   └── output.txt
├── os4bit/          # Output Stationary, 4-bit
│   └── ...
└── os2bit/          # Output Stationary, 2-bit
    └── ...
```

### Running Tests

```bash
# Run specific test mode
make vanilla

# Verify output
python scripts/verify_output.py \
    --expected golden/ws4bit/output.txt \
    --actual output/result.txt
```

### Expected Output Format

Output files contain:
- Binary format: 256 bits per line (16 channels × 16 bits)
- Each channel represents one output feature
- Data is ordered from row7 to row0 (MSB to LSB)

## Troubleshooting

### Common Issues

#### Issue: Compilation errors

**Solution:**
```bash
# Clean and rebuild
make clean
make vanilla

# Check filelist
cat filelist

# Verify source files exist
ls -la src/Mac/
```

#### Issue: Simulation fails

**Solution:**
```bash
# Check testbench syntax
iverilog -t null src/core_tb.v

# Verify golden data exists
ls -la golden/ws4bit/

# Check file paths in testbench
grep "golden" src/core_tb.v
```

#### Issue: Output mismatch

**Solution:**
```bash
# Verify input data format
python scripts/verify_transform_2bit.py

# Check weight file format
head -n 5 golden/ws4bit/weight_itile0_otile0_kij0.txt

# Compare with expected output
diff output/result.txt golden/ws4bit/output.txt
```

#### Issue: OS mode not working

**Solution:**
```bash
# Verify OS filelist
cat filelist_os

# Check OS-specific modules
ls -la src/FIFO/ififo.v

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
