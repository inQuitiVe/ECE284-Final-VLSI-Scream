# ECE284 Final VLSI Scream Project

[![Status](https://img.shields.io/badge/status-stable-green.svg)]()
[![Hardware](https://img.shields.io/badge/hardware-FPGA-blue.svg)]()
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)]()

A comprehensive hardware accelerator project implementing a 2D systolic array architecture for deep neural network inference, featuring multiple innovations in MAC design, dataflow reconfiguration, SIMD processing, and post-processing units.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Alpha Versions](#alpha-versions)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Hardware Requirements](#hardware-requirements)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Quick Start

Get started with the accelerator in minutes:

```bash
# Clone the repository
git clone <repository-url>
cd ECE284-Final-VLSI-Scream

# Install dependencies (Ubuntu/Debian)
sudo apt-get install iverilog gtkwave

# Run a basic simulation (Alpha1 - Baseline)
cd Part5_Alpha/Alpha1/hardware
make vanilla

# View results
make view  # Opens GTKWave to view waveforms
```

**What you just ran**: A complete 8×8 systolic array accelerator simulation processing 4-bit activations in Weight Stationary mode, performing convolution operations and ReLU activation.

### Try Different Modes

```bash
# 2-bit SIMD mode (parallel processing)
make act_2b

# Output Stationary dataflow
make os_vanilla

# Combined OS + 2-bit mode
make os_2b

# Run all test modes sequentially (recommended for comprehensive testing)
make all
```

**Recommended**: Use `make all` to run all available test modes sequentially. This ensures comprehensive verification across all configurations.

### Explore Other Versions

- **Part2**: 2-bit/4-bit SIMD (WS only) - `cd Part2_SIMD/hardware && make vanilla`
- **Part3**: Full reconfigurable (OS/WS + 2-bit/4-bit) - `cd Part3_Reconfigurable/hardware && make vanilla`
- **Alpha5**: Conv-BN fusion + MaxPool - `cd Part5_Alpha/Alpha5 && iverilog -f filelist -o compiled && vvp compiled`
- **Alpha6**: Multiple activation functions - `cd Part5_Alpha/Alpha6 && make vanilla_elu`

See [Installation](#installation) and [Usage](#usage) sections for detailed instructions.

## Overview

This project implements a **complete hardware accelerator for convolutional neural network inference** using a 2D systolic array architecture. The accelerator features an **8×8 array of processing elements (PEs)**, each performing multiply-accumulate (MAC) operations, with comprehensive support for different dataflow patterns, precision modes, and post-processing operations.

### What This Project Provides

A **production-ready systolic array accelerator** with the following capabilities:

1. **Flexible Dataflow Execution**: Switch between Weight Stationary (WS) and Output Stationary (OS) modes to optimize for different workload characteristics
2. **Adaptive Precision Processing**: Support for both 4-bit (standard) and 2-bit (SIMD parallel) activation processing, enabling power and area efficiency trade-offs
3. **Autonomous Operation**: FSM-based control unit that manages the entire computation pipeline with minimal external intervention
4. **Model Optimization**: Conv-BN fusion techniques that reduce computation by 4.3% and hardware cost by 25%
5. **Extensible Activation Functions**: Hardware support for ReLU, ELU, LeakyReLU, and GELU activation functions
6. **Spatial Pooling**: Hardware-efficient MaxPool(2×2) implementation for feature map reduction

### Project Organization

The project is organized into **Parts** (implementation milestones) and **Alpha versions** (feature demonstrations):

- **Part1**: Vanilla baseline implementation
- **Part2**: 2-bit/4-bit SIMD support (Weight Stationary only)
- **Part3**: Complete reconfigurable design (OS/WS + 2-bit/4-bit)
- **Alpha1-8**: Individual feature demonstrations and optimizations

Each version builds upon previous work, with Alpha1 providing the most comprehensive feature set serving as the baseline for all other implementations.

### Key Innovations

- **Memory Efficiency**: Reduced PSUM memory from 5.0625MB to 0.25MB (4.93% of original) through intelligent address calculation
- **SIMD Parallelism**: 2-bit mode processes two activation-weight pairs in parallel per MAC unit, doubling throughput without doubling area
- **Model Fusion**: Mathematical fusion of BatchNorm parameters into convolution weights, eliminating runtime normalization overhead
- **Autonomous Control**: FSM-based SFU requiring only two external control signals (`ofifo_valid` and `readout_start`)

The design achieves significant improvements in area efficiency, power consumption, and parameter storage through these optimization techniques, making it suitable for edge AI inference applications.

## Key Features

### Core Capabilities

- ✅ **8×8 Systolic Array**: Hierarchical MAC array architecture (array → row → tile → PE) - All versions
- ✅ **Dual Dataflow Modes**: Weight Stationary (WS) and Output Stationary (OS) with runtime switching - Part3, Alpha1
- ✅ **SIMD Reconfiguration**: 2-bit/4-bit activation processing modes - Part2, Part3, Alpha1, Alpha6
  - 2-bit mode: Parallel processing of two activation-weight pairs per MAC unit
  - Power and area efficient design through SIMD parallelism
- ✅ **Autonomous SFU**: FSM-based Summation and Function Unit with minimal control overhead - All versions (documented in Alpha7)
- ✅ **Memory Optimization**: Reduced PSUM memory from 5.0625MB to 0.25MB (4.93% of original) - All versions with onij_calculator
- ✅ **Model Fusion**: Conv-BN fusion reducing computation by 4.3% and hardware cost by 25% - Alpha5
- ✅ **Flexible Activations**: Support for ReLU, ELU, LeakyReLU, and GELU - Alpha6
- ✅ **Spatial Pooling**: Hardware-efficient MaxPool(2×2, stride=2) with minimal overhead - Alpha4, Alpha5

### Performance Characteristics

- **Throughput**: One MAC operation per cycle per PE (4-bit mode), or two parallel operations per cycle per PE (2-bit mode)
- **Latency**: Pipeline-optimized with minimal bubble cycles
- **Area Efficiency**: Shared hardware resources for parallel operations in 2-bit SIMD mode
- **Power Efficiency**: Reduced switching activity through SIMD parallelism in 2-bit mode

## Project Structure

```
ECE284-Final-VLSI-Scream/
├── Part1_Vanilla/           # Part 1: Vanilla implementation
│   ├── hardware/            # Hardware implementation
│   └── software/            # Software tools and golden data generation
├── Part2_SIMD/              # Part 2: 2-bit/4-bit SIMD (Weight Stationary only)
│   ├── hardware/            # Hardware with SIMD support, WS mode only
│   └── software/            # Quantization models and tools
├── Part3_Reconfigurable/    # Part 3: OS/WS reconfigurable + 2-bit/4-bit SIMD
│   ├── hardware/            # Full reconfigurable hardware (OS/WS + 2-bit/4-bit)
│   └── software/            # Quantization models and tools
├── Part4_Poster/            # Project poster and documentation
│   └── VLSI_Scream_Final_Poster.pdf
├── Part5_Alpha/             # Alpha versions with innovations
│   ├── Alpha1/              # MAC Implementation (baseline, full features)
│   │   └── hardware/        # Complete hardware with OS/WS and 2-bit/4-bit
│   ├── Alpha2/              # Basic implementation
│   │   ├── hardware/
│   │   └── software/
│   ├── Alpha3/              # Huffman Decoder
│   │   └── hardware/
│   ├── Alpha4/              # MaxPool implementation
│   │   ├── src/             # SFU with MaxPool
│   │   └── golden/
│   ├── Alpha5/              # Conv-BN Fusion + MaxPool
│   │   ├── src/             # SFU with Conv-BN fusion and MaxPool
│   │   └── golden/
│   ├── Alpha6/              # Flexible Activation Functions
│   │   ├── src/             # Hardware with multiple activation functions
│   │   └── golden/
│   ├── Alpha7/              # SFU FSM Implementation
│   │   └── SFU/             # SFU module with autonomous FSM
│   └── Alpha8/              # Quantization models
│       ├── models/          # Quantized model implementations
│       └── *.py             # Training and quantization scripts
├── Part6_Report/            # Project report
├── Part7_ProgressReport/    # Progress reports
│   └── ECE284 Progress Report.pdf
└── README.md                # This file
```

## Alpha Versions

The project includes multiple Alpha versions, each introducing specific innovations:

### Alpha 1: MAC Implementation (Baseline)

**Foundation for all other versions - Complete feature set**

- **Purpose**: Comprehensive MAC unit implementations for the 2D systolic array with full feature support
- **Features**:
  - Vanilla and flexible MAC designs
  - Support for both WS and OS dataflow modes with runtime switching
  - 2-bit/4-bit SIMD reconfiguration
  - Hierarchical architecture (array → row → tile → PE)
  - Complete test infrastructure for all mode combinations
- **Key Innovation**: Comprehensive MAC architecture supporting multiple dataflow and precision modes in a unified design
- **Note**: Alpha1 provides the most complete implementation, serving as the baseline for all other versions
- **Directory**: `Part5_Alpha/Alpha1/`
- **Documentation**: [Alpha1 README](Part5_Alpha/Alpha1/README.md)

### Alpha 2: Basic Implementation

- **Purpose**: Basic accelerator implementation
- **Features**: Core functionality for convolution operations
- **Directory**: `Part5_Alpha/Alpha2/`

### Alpha 3: Huffman Decoder

- **Purpose**: Data compression/decompression for neural network weights
- **Features**:
  - Serial bit input processing
  - State machine-based Huffman tree traversal
  - 8-bit symbol output
- **Key Innovation**: Hardware decoder for compressed weight storage
- **Directory**: `Part5_Alpha/Alpha3/`
- **Documentation**: [Alpha3 README](Part5_Alpha/Alpha3/README.md)

### Alpha 4: MaxPool Implementation

- **Purpose**: Spatial pooling for feature map reduction
- **Features**:
  - MaxPool(2×2, stride=2) module
  - Hardware-efficient implementation requiring only 1 additional flip-flop per SIMD lane
  - Integrated with SFU for post-processing
- **Key Innovation**: Minimal hardware overhead MaxPool implementation
- **Directory**: `Part5_Alpha/Alpha4/`
- **Documentation**: [Alpha4 README](Part5_Alpha/Alpha4/README.md)

### Alpha 5: Conv-BN Fusion + MaxPool

- **Purpose**: Model optimization and spatial pooling
- **Features**:
  - Conv-BN fusion reducing parameters and computation
  - MaxPool(2×2, stride=2) with minimal hardware overhead
  - Bias handling in SFU first accumulation cycle
- **Key Innovation**: Mathematical fusion of BatchNorm into convolution weights
  - Computation reduction: **4.3%**
  - Hardware cost reduction: **25%**
  - Parameter reduction through offline fusion
- **Mathematical Formulation**:
  \[
  \begin{cases}
  w' = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot w_{\text{conv}} \\
  b' = \frac{-\mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
  \end{cases}
  \]
- **Directory**: `Part5_Alpha/Alpha5/`
- **Documentation**: [Alpha5 README](Part5_Alpha/Alpha5/README.md)

### Alpha 6: Flexible Activation Functions

- **Purpose**: Runtime-selectable activation functions
- **Features**:
  - Unified activation function module
  - Support for ReLU, ELU, LeakyReLU, and GELU
  - Hardware-efficient approximations
  - Compatible with all dataflow and SIMD modes
- **Key Innovation**: Single hardware module supporting multiple activation functions via mode selection
- **Activation Functions**:
  - ReLU: Direct hardware implementation
  - LeakyReLU: Bit-shift approximation (α ≈ 2⁻⁶)
  - ELU: Piecewise linear approximation
  - GELU: Simplified hardware approximation
- **Directory**: `Part5_Alpha/Alpha6/`
- **Documentation**: [Alpha6 README](Part5_Alpha/Alpha6/README.md)

### Alpha 7: SFU FSM Implementation

- **Purpose**: Documentation of autonomous control for Summation and Function Unit
- **Features**:
  - FSM-based autonomous operation
  - Minimal external control signals (only `ofifo_valid` and `readout_start`)
  - Memory size reduction through onij_calculator (5.0625MB → 0.25MB, 4.93% of original)
  - Readout mechanism avoiding SRAM race conditions by preventing direct testbench access to PSUM memory
- **Key Innovation**: FSM enabling SFU to operate autonomously with minimal control
  - **Important Note**: This FSM implementation innovation already exists in all versions of this project. Alpha7 serves only as a marker point to document this design characteristic.
- **FSM States**:
  - `S_Init`: Wait for OFIFO valid signal
  - `S_Acc`: Automatically accumulate PSUMs (manages nij counter 0-35)
  - `S_ReLU`: Automatically apply ReLU activation (manages o_nij counter 0-15)
  - `S_Idle`: Wait for readout_start signal (only external control point)
  - `S_Readout`: Automatically read all output channel data (manages o_nij counter 0-15)
- **Directory**: `Part5_Alpha/Alpha7/`
- **Documentation**: [Alpha7 README](Part5_Alpha/Alpha7/README.md)

### Alpha 8: Quantization Models

- **Purpose**: Quantized model implementations
- **Features**:
  - 4-bit quantized models (91.53% accuracy)
  - 2-bit quantized models (90.67% accuracy)
  - VGG16 with BatchNorm (92.13% accuracy baseline)
- **Key Innovation**: Hardware-compatible quantized models for deployment
- **Directory**: `Part5_Alpha/Alpha8/`
- **Documentation**: [Alpha8 README](Part5_Alpha/Alpha8/README.md)

## Architecture

### System Overview

```
┌─────────────┐
│   X_MEM     │  Activation & Weight Storage
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  L0 FIFO    │  Activation Buffer
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   8×8 MAC       │  Systolic Array
│   Array         │  - Weight Stationary (WS)
└──────┬──────────┘  - Output Stationary (OS)
       │              - 2-bit/4-bit SIMD
       ▼
┌─────────────┐
│   OFIFO     │  Output FIFO
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     SFU     │  Summation & Function Unit
│  (FSM)      │  - Accumulation
└──────┬──────┘  - Activation (ReLU/ELU/LeakyReLU/GELU)
       │         - MaxPool (Alpha5)
       │         - Autonomous FSM control
       ▼
┌─────────────┐
│   PSUM_MEM  │  Output Memory
└─────────────┘
```

### MAC Hierarchy

```
mac_array (8×8)
  └── mac_row[0:7] (8 rows)
      └── mac_tile[0:7] (8 tiles per row)
          └── mac (single PE)
```

### Dataflow Modes

#### Weight Stationary (WS)
- Weights loaded and stored in each PE
- Activations flow left to right
- Partial sums flow top to bottom
- Efficient for weight reuse

#### Output Stationary (OS)
- Partial sums accumulated in each PE
- Weights flow top to bottom
- Activations flow left to right
- Different instruction encoding

### SIMD Modes

#### 4-bit Mode (Default)
- 4 bits per activation value
- One activation-weight pair per cycle per MAC

#### 2-bit Mode
- 2 bits per activation value
- **Two activation-weight pairs processed in parallel** per MAC
- Power and area efficient through SIMD parallelism

## Quick Start

### Prerequisites

- **Icarus Verilog**: 10.0+ (for simulation)
- **GTKWave**: (optional, for waveform viewing)
- **Make**: (for build automation)
- **Python**: 3.7+ (for data transformation scripts)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ECE284-Final-VLSI-Scream

# Install Verilog tools (Ubuntu/Debian)
sudo apt-get install iverilog gtkwave

# Install Verilog tools (macOS)
brew install icarus-verilog gtkwave

# Install Python dependencies
pip install numpy
```

### Running Simulations

#### Alpha 1 (Baseline)

```bash
cd Part5_Alpha/Alpha1/hardware

# Run vanilla mode (4-bit, WS)
make vanilla

# Run 2-bit mode (2-bit, WS)
make act_2b

# Run OS mode (4-bit, OS)
make os_vanilla

# Run OS 2-bit mode
make os_2b
```

#### Alpha 5 (Conv-BN Fusion)

```bash
cd Part5_Alpha/Alpha5
iverilog -f filelist -o compiled
vvp compiled
```

#### Alpha 6 (Flexible Activations)

```bash
cd Part5_Alpha/Alpha6

# Run with different activation functions
make vanilla          # ReLU
make vanilla_elu      # ELU
make vanilla_leaky    # LeakyReLU
make vanilla_gelu     # GELU
```

## Usage

### Basic Workflow

1. **Navigate to Target Directory**: Choose the part or Alpha version you want to run
2. **Select Mode**: Use Makefile targets to configure dataflow (WS/OS) and SIMD (2-bit/4-bit) modes
3. **Run Simulation**: Execute the testbench with `make <target>` or direct `iverilog` commands
4. **Verify Output**: Testbench automatically compares results against golden reference data
5. **View Waveforms**: Use `make view` or `gtkwave` to inspect signal timing (optional)

### Example Workflows

#### Running Alpha1 (Complete Feature Set)

```bash
cd Part5_Alpha/Alpha1/hardware

# Standard 4-bit Weight Stationary mode
make vanilla

# 2-bit SIMD mode (parallel processing)
make act_2b

# Output Stationary mode
make os_vanilla

# Combined OS + 2-bit mode
make os_2b

# Run all modes sequentially
make all
```

#### Running Part2 (SIMD Only, WS Mode)

```bash
cd Part2_SIMD/hardware

# 4-bit mode
make vanilla

# 2-bit mode
make act_2b
```

#### Running Part3 (Full Reconfigurable)

```bash
cd Part3_Reconfigurable/hardware

# All combinations available
make vanilla      # WS, 4-bit
make act_2b       # WS, 2-bit
make os_vanilla   # OS, 4-bit
make os_2b        # OS, 2-bit

# Run all modes
make all
```

#### Running Alpha6 (Flexible Activations)

```bash
cd Part5_Alpha/Alpha6

# Different activation functions
make vanilla          # ReLU (default)
make vanilla_elu      # ELU
make vanilla_leaky    # LeakyReLU
make vanilla_gelu     # GELU
```

### Makefile Targets

Common targets across Alpha versions:

| Target | Description |
|--------|-------------|
| `vanilla` | 4-bit activation, Weight Stationary mode |
| `act_2b` | 2-bit activation, Weight Stationary mode |
| `os_vanilla` | 4-bit activation, Output Stationary mode |
| `os_2b` | 2-bit activation, Output Stationary mode |
| `all` | Run all test modes sequentially |
| `clean` | Remove compiled files and VCD files |
| `view` | View waveform with GTKWave |

### Data Format

- **Activations**: 
  - 4-bit mode: Binary format, 32 bits per line (8 channels × 4 bits)
  - 2-bit mode: Binary format, 16 bits per line (8 channels × 2 bits), requires two tiles (tile0 and tile1)
- **Weights**: Binary format, organized by input/output tiles and kernel iterations (kij: 0-8)
- **Outputs**: Binary format, 128 bits per line (8 channels × 16 bits) for both 2-bit and 4-bit modes

**Note**: Data format details vary by part and Alpha version. See individual README files for specific format specifications:
- [Part2 README](Part2_SIMD/README.md) - WS mode only, 2-bit/4-bit SIMD
- [Part3 README](Part3_Reconfigurable/README.md) - OS/WS + 2-bit/4-bit
- [Alpha1 README](Part5_Alpha/Alpha1/README.md) - Complete feature set

## Hardware Requirements

### Simulation
- **CPU**: Multi-core recommended for faster simulation
- **RAM**: 4GB+ recommended
- **Storage**: 1GB+ for test data and waveforms

### FPGA Deployment
- **FPGA**: Cyclone IV GX or compatible
- **Development Board**: With appropriate I/O interfaces
- **Memory**: Sufficient on-chip memory for activations and weights

## Contributing

This is an academic project. For questions or contributions, please contact the maintainers.

### Development Guidelines

- Follow Verilog coding standards
- Add tests for new features
- Update documentation as needed
- Maintain compatibility with existing test infrastructure

## License

Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department  
Please do not spread this code without permission

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{vlsi_scream_2024,
  title={ECE284 Final VLSI Scream Project: A Comprehensive Systolic Array Accelerator},
  author={VVIP Lab, UCSD},
  year={2024},
  note={University of California, San Diego}
}
```

## References

### Part Documentation
- [Part1 README](Part1_Vanilla/README.md) - Vanilla implementation
- [Part2 README](Part2_SIMD/README.md) - 2-bit/4-bit SIMD (WS only)
- [Part3 README](Part3_Reconfigurable/README.md) - OS/WS reconfigurable + 2-bit/4-bit SIMD

### Alpha Documentation
- [Alpha1 README](Part5_Alpha/Alpha1/README.md) - Baseline MAC implementation (complete feature set)
- [Alpha2 README](Part5_Alpha/Alpha2/README.md) - Basic implementation
- [Alpha3 README](Part5_Alpha/Alpha3/README.md) - Huffman decoder
- [Alpha4 README](Part5_Alpha/Alpha4/README.md) - MaxPool implementation
- [Alpha5 README](Part5_Alpha/Alpha5/README.md) - Conv-BN fusion and MaxPool
- [Alpha6 README](Part5_Alpha/Alpha6/README.md) - Flexible activation functions
- [Alpha7 README](Part5_Alpha/Alpha7/README.md) - SFU FSM implementation (documentation)
- [Alpha8 README](Part5_Alpha/Alpha8/README.md) - Quantization models

---



