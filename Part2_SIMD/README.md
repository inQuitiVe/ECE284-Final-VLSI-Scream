# Part 2: 2-bit/4-bit Reconfigurable SIMD

[![Status](https://img.shields.io/badge/status-stable-green.svg)]()
[![Hardware](https://img.shields.io/badge/hardware-FPGA-blue.svg)]()
[![SIMD](https://img.shields.io/badge/SIMD-2b%2F4b-orange.svg)]()
[![Dataflow](https://img.shields.io/badge/dataflow-WS-purple.svg)]()

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

Part 2 implements a 2-bit/4-bit reconfigurable SIMD (Single Instruction, Multiple Data) accelerator for the 2D systolic array. This version focuses on **Weight Stationary (WS) dataflow mode only**, with support for switchable 2-bit and 4-bit activation processing. The SIMD reconfiguration enables flexible precision processing with significant power and area benefits through parallel activation-weight pair processing.

### Key Focus

- **Weight Stationary Dataflow**: Part 2 exclusively implements WS mode, where weights are loaded and stored in each processing element (PE)
- **SIMD Reconfigurability**: Switchable 2-bit and 4-bit activation processing modes
- **Power and Area Efficiency**: 2-bit mode processes two activation-weight pairs in parallel, reducing power consumption and improving area efficiency

### Key Innovations

- **SIMD Reconfigurability**: Switchable 2-bit and 4-bit activation processing
- **Parallel Processing**: 2-bit mode enables parallel processing of two activation-weight pairs per MAC unit
- **Weight Stationary Dataflow**: Efficient weight reuse through WS mode implementation
- **Modular Design**: Hierarchical structure (array → row → tile) for scalability

## Features

- ✅ Weight Stationary (WS) dataflow mode implementation
- ✅ **2-bit/4-bit activation mode switching** (bit-width refers to activation data)
  - 2-bit mode: Parallel processing of two activation-weight pairs per MAC unit
  - Power and area efficient design through SIMD parallelism
- ✅ MAC array implementation with hierarchical structure
- ✅ Complete test infrastructure for both 2-bit and 4-bit modes
- ✅ Visualization tools for data analysis

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
- **Data Flow Control**: Manages activation and weight flow

### Weight Stationary (WS) Mode

Part 2 implements **only Weight Stationary (WS) mode**:

- Weights are loaded and stored in each PE
- Activations flow from left to right
- Partial sums flow from top to bottom
- Efficient for weight reuse scenarios
- Weights remain stationary in PEs while activations stream through

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
cd ECE284-Final-VLSI-Scream/Part2_SIMD
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
ls -la verilog/Mac/
ls -la verilog/SFU/
```

## Quick Start

If you have the test data already prepared, you can directly run simulations:

```bash
cd hardware

# Run all modes (default)
make all

# Or run individual modes:
make vanilla    # 4-bit activation, WS mode
make act_2b     # 2-bit activation, WS mode
```

### View Results

```bash
# View waveform
make view

# Check output files
cat datafiles/ws4bit/out.txt        # For ws4bit mode
cat datafiles/ws2bit/expected_output_from_psum_binary.txt     # For ws2bit mode
```

> **Note**: If you need to prepare your own test data, see [Data Preparation](#data-preparation) section below.

## Project Structure

```
Part2_SIMD/
├── README.md                 # This file
│
├── hardware/                 # Hardware implementation
│   ├── Makefile             # Build configuration
│   │
│   ├── verilog/             # All HDL sources
│   │   ├── core_tb.v       # Testbench
│   │   ├── core.v          # Top-level core module
│   │   ├── corelet.v       # Corelet (MAC array + FIFOs)
│   │   ├── Mac/            # MAC array modules
│   │   │   ├── mac_array.v # MAC array top-level
│   │   │   ├── mac_row.v   # MAC row
│   │   │   ├── mac_tile.v  # MAC tile (PE)
│   │   │   └── mac.v       # Basic MAC unit
│   │   ├── SFU/            # Summation and Function Unit
│   │   │   ├── SFU.v       # SFU main module
│   │   │   ├── ReLU.v      # ReLU activation
│   │   │   └── onij_calculator.v  # Output address calculator
│   │   ├── FIFO/           # FIFO modules
│   │   │   ├── l0.v        # L0 FIFO
│   │   │   ├── ofifo.v     # Output FIFO
│   │   │   ├── ififo.v     # Input FIFO
│   │   │   ├── fifo_depth64.v  # Base FIFO
│   │   │   ├── fifo_mux_2_1.v  # 2-to-1 FIFO mux
│   │   │   ├── fifo_mux_8_1.v  # 8-to-1 FIFO mux
│   │   │   └── fifo_mux_16_1.v # 16-to-1 FIFO mux
│   │   └── SRAM/           # SRAM modules
│   │       ├── sram_32b_w2048.v   # X_MEM
│   │       ├── sram_128b_w16_RW.v # PSUM_MEM
│   │       └── sram_64b_w256.v    # Additional SRAM
│   │
│   ├── datafiles/           # Input files used by the testbench
│   │   ├── ws4bit/         # WS mode, 4-bit data
│   │   │   ├── activation_tile0.txt
│   │   │   ├── weight_itile0_otile0_kij*.txt
│   │   │   ├── out.txt, out_raw.txt
│   │   │   └── viz/        # Visualization files
│   │   │
│   │   └── ws2bit/         # WS mode, 2-bit data
│   │       ├── activation_tile0.txt, activation_tile1.txt
│   │       ├── weight_itile*_otile*_kij*.txt
│   │       ├── expected_output_from_psum_binary.txt
│   │       ├── viz/        # Visualization files
│   │       └── *.py        # Analysis scripts (parse.py, viz.py, calc_psum.py, etc.)
│   │
│   └── sim/                 # Simulation files and the runtime filelist
│       └── filelist         # REQUIRED: Plain text file with relative paths (../verilog/) to design files
│
└── software/                # Quantization models and tools
    └── ...
```

## Usage

### Basic Usage

The hardware supports two main configurations (both in WS mode):

```bash
cd hardware

# Run vanilla mode (4-bit, WS)
make vanilla

# Run 2-bit mode (2-bit, WS)
make act_2b

# Run all modes
make all
```

### Makefile Targets

| Target | Description | Status |
|--------|-------------|--------|
| `vanilla` | 4-bit activation, Weight Stationary mode | ✅ Supported |
| `act_2b` | 2-bit activation, Weight Stationary mode | ✅ Supported |
| `all` | Run all test modes sequentially (default) | ✅ Supported |
| `clean` | Remove compiled files and VCD files | ✅ Supported |
| `view` | View waveform with GTKWave | ✅ Supported |
| `help` | Show all available targets | ✅ Supported |

> **Important Note**: 
> - Part 2 **only supports Weight Stationary (WS) mode**. OS mode is not implemented in this version.
> - The `all` target runs `vanilla` and `act_2b` sequentially.

### Advanced Usage

#### Custom Compilation

```bash
# Compile with custom flags (from hardware/ directory)
cd sim
iverilog -DACT_2BIT -f filelist -o compiled
vvp compiled
```

#### Waveform Viewing

```bash
# Generate VCD and view
make vanilla
make view

# Or manually (VCD files are generated in sim/ directory)
cd sim
gtkwave core_tb_vanilla.vcd    # For vanilla mode
gtkwave core_tb_2bit.vcd       # For 2-bit mode
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

- Binary format: 128 bits per line (8 channels × 16 bits) for both 4-bit and 2-bit modes
- Each channel represents one output feature (mac_col = 8)
- Each channel is 16 bits (psum_bw = 16)
- Data is ordered from col7 to col0 (MSB to LSB)
- Format: `timeXcol7[msb-lsb], timeXcol6[msb-lsb], ..., timeXcol0[msb-lsb]`

### Step 1: Prepare Weight Data (for 2-bit mode)

For 2-bit mode, you may need to convert weight files from other formats:

```bash
cd datafiles/ws2bit
python parse.py
```

**What it does:**
- Converts weight files from `from_yufan/` directory to the required format
- Maps tile indices according to the naming convention
- Generates `weight_itile*_otile*_kij*.txt` files

### Step 2: Generate Visualization Files

Binary data files are not human-readable. Use the `viz.py` scripts to convert them to decimal format for debugging and verification.

#### For 4-bit Mode

```bash
cd datafiles/ws4bit
# Note: viz.py may need to be copied from ws2bit/ directory if it doesn't exist here
python viz.py
```

#### For 2-bit Mode

```bash
cd datafiles/ws2bit
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
cat datafiles/ws4bit/viz/viz_activation_tile0.txt

# View converted output
cat datafiles/ws4bit/viz/viz_out.txt
```

### Step 3: Calculate Expected PSUM (for 2-bit mode)

For 2-bit mode, you can calculate expected partial sums:

```bash
cd datafiles/ws2bit
python calc_psum.py
```

**What it does:**
- Calculates expected psum values for the systolic array
- Reads activation and weight files
- Generates `calc_psum_output.txt` with calculated psums

### Python Scripts Reference

All data transformation scripts are located in `datafiles/ws2bit/`:

| Script | Location | Purpose | Notes |
|--------|----------|---------|-------|
| `parse.py` | `datafiles/ws2bit/` | Convert weight files from yufan format | Hardcoded paths, no args |
| `viz.py` | `datafiles/ws2bit/` | Convert binary to decimal format | Run from ws2bit directory. For ws4bit, may need to copy from ws2bit/ |
| `calc_psum.py` | `datafiles/ws2bit/` | Calculate expected psum values | Hardcoded paths, no args |
| `verify_psum.py` | `datafiles/ws2bit/` | Verify calculated psums | Hardcoded paths, no args |
| `make_expected_from_psum.py` | `datafiles/ws2bit/` | Generate expected output from psum | Hardcoded paths, no args |

## Hardware Configuration

### Configuration Modes

The hardware supports two main configurations (both in WS mode):

| Mode | Activation Bits | Dataflow | Makefile Target |
|------|----------------|----------|-----------------|
| Vanilla | 4-bit | WS | `vanilla` |
| 2-bit | 2-bit | WS | `act_2b` |

> **Note**: Part 2 only implements Weight Stationary (WS) mode. Output Stationary (OS) mode is not supported in this version.

### Compilation Flags

The Makefile uses the following flags:

- **`-DACT_2BIT`**: Enable 2-bit activation mode

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
datafiles/
├── ws4bit/          # Weight Stationary, 4-bit
│   ├── activation_tile0.txt
│   ├── weight_itile0_otile0_kij*.txt
│   ├── out.txt              # Output (after ReLU)
│   └── out_raw.txt          # Raw output (before ReLU)
└── ws2bit/          # Weight Stationary, 2-bit
    ├── activation_tile0.txt, activation_tile1.txt
    ├── weight_itile*_otile*_kij*.txt
    └── expected_output_from_psum_binary.txt  # Expected output file
```

### Running Tests

```bash
# Run specific test mode
make vanilla

# Verify output (testbench automatically compares)
# For ws4bit: compares against datafiles/ws4bit/out.txt
# For ws2bit: compares against datafiles/ws2bit/expected_output_from_psum_binary.txt
```

### Expected Output Format

Output files contain:
- **Both 4-bit and 2-bit modes**: 128 bits per line (8 channels × 16 bits)
- Each channel represents one output feature (mac_col = 8)
- Each channel is 16 bits (psum_bw = 16)
- Data is ordered from col7 to col0 (MSB to LSB)
- Format: `timeXcol7[msb-lsb], timeXcol6[msb-lsb], ..., timeXcol0[msb-lsb]`

### Viewing Results with Visualization

For easier debugging and verification, use the visualization scripts to convert binary outputs to human-readable format:

```bash
# Generate visualization files
cd datafiles/ws4bit
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
make clean
make vanilla

# Check filelist
cat sim/filelist

# Verify source files exist
ls -la verilog/Mac/
```

#### Issue: Simulation fails

**Solution:**
```bash
# Check testbench syntax
iverilog -t null verilog/core_tb.v

# Verify datafiles exist
ls -la datafiles/ws4bit/

# Check file paths in testbench
grep "datafiles" verilog/core_tb.v
```

#### Issue: Output mismatch

**Solution:**
```bash
# Check weight file format
head -n 5 datafiles/ws4bit/weight_itile0_otile0_kij0.txt

# Compare with expected output (check testbench output)
# The testbench will automatically compare and report mismatches

# For 2-bit mode, verify psum calculations
cd datafiles/ws2bit
python verify_psum.py
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

### Related Project Versions

- **Part 1**: Vanilla Version
- **Part 3**: OS/WS reconfigurable Version (includes both WS and OS modes)
- **Alpha 1**: MAC Implementation (comprehensive design with WS/OS and 2b/4b)

### Technical Documentation

- **MAC Architecture**: See `verilog/Mac/` for implementation details
- **Data Format**: See `datafiles/` for data format specifications

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
@misc{part2_simd,
  title={2-bit/4-bit Reconfigurable SIMD Accelerator with Weight Stationary Dataflow},
  author={VVIP Lab, UCSD},
  year={2024},
  note={Part 2 of ECE284 Final Project}
}
```

---

**Status**: Stable  
**Last Updated**: 2024  
**Maintainer**: VVIP Lab @ UCSD
