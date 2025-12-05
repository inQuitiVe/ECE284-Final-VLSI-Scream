# ECE284 Final Project: VLSI Scream

SIMD and Weight-Output reconfigurable 2D systolic array-based AI accelerator and mapping on Cyclone IV GX

## Project Overview

This project implements a reconfigurable 2D systolic array-based AI accelerator supporting:
- **SIMD reconfiguration**: 2-bit and 4-bit activation modes
- **Dataflow reconfiguration**: Weight Stationary (WS) and Output Stationary (OS) modes
- **Flexible activation functions**: ReLU, ELU, LeakyReLU, and GELU

For detailed information, please refer to the [Poster](Poster/VLSI_Scream_Final_Poster.pdf).

## Project Structure

```
ECE284-Final-VLSI-Scream/
├── Part1/              # Vanilla Version
├── Part2/              # 2b/4b Reconfigurable SIMD Version
├── Part3/              # OS/WS Reconfigurable Version
├── Alpha/              # Innovation Versions
│   ├── Alpha1/         # MAC Implementation (Vanilla & OS)
│   ├── Alpha2/         # MAC Array/Row/Tile Modules
│   ├── Alpha3/         # Huffman Decoder
│   ├── Alpha4/         # Whole Conv Layer (BN, ReLU, MaxPooling)
│   ├── Alpha5/         # Model Fusion (BN fusion)
│   ├── Alpha6_FlexActFunc/  # Flexible Activation Functions
│   ├── Alpha7/         # SFU FSM Implementation
│   └── Alpha8/         # ConvNext Application
└── Poster/             # Project Poster
```

## Part Descriptions

### Part 1: Vanilla Version
Basic implementation of the 2D systolic array accelerator.

**Structure:**
- `hardware/src/`: Verilog source code including testbench
- `hardware/golden/`: Test data files (binary and human-readable `viz_` prefix files)
- `software/`: Golden pattern generator (Jupyter notebook)

### Part 2: 2b/4b Reconfigurable SIMD Version
Extends Part 1 with reconfigurable SIMD support for 2-bit and 4-bit activation modes.

**Features:**
- Switchable activation bit-width (2-bit or 4-bit)
- Weight Stationary (WS) dataflow
- Support for multiple activation functions

**Structure:**
- `hardware/src/`: Complete hardware implementation
- `hardware/golden/ws2bit/`: Test data for 2-bit mode
- `hardware/golden/ws4bit/`: Test data for 4-bit mode
- `Makefile`: Build system with multiple test modes

### Part 3: OS/WS Reconfigurable Version
Adds Output Stationary (OS) dataflow support in addition to Weight Stationary.

**Features:**
- Switchable dataflow modes (OS/WS)
- Combined with SIMD reconfiguration (2b/4b)
- Complete system integration

## Alpha Versions

The `Alpha/` directory contains various innovation versions developed during the project.

### Alpha1: MAC Implementation
Initial MAC (Multiply-Accumulate) unit implementation.

**Contents:**
- `mac_vanilla.v`: Basic MAC implementation
- `mac_flex.v`: Flexible MAC implementation
- `os/`: Output Stationary MAC implementation
- Reference hardware implementations

### Alpha2: MAC Array Components
Core MAC array building blocks.

**Modules:**
- `mac_array.v`: MAC array top-level
- `mac_row.v`: MAC row implementation
- `mac_tile.v`: MAC tile (PE) implementation

### Alpha3: Huffman Decoder
Huffman decoder implementation for data compression/decompression.

**Module:**
- `huffman_decoder.v`: Huffman decoding logic

### Alpha4: Whole Conv Layer
Complete convolutional layer implementation with post-processing layers.

**Features:**
- Batch Normalization (BN)
- ReLU activation
- Max Pooling Layer (MPL)

**Key Innovation:**
- Provides a structured approach for implementing BN, ReLU, and MPL sequentially after Conv2D computation
- Enables full VGG16 CNN processing capability

**Structure:**
- `src/SFU/`: Contains MaxPooling (`max4.v`) and related calculators
- `golden/`: Test data including intermediate psum files for each kernel iteration

### Alpha5: Model Fusion
Batch Normalization parameter fusion into adjacent convolution layers.

**Key Innovation:**
- BN parameters are fixed at inference time and fused into adjacent convolution layers
- Eliminates runtime normalization operations

**Benefits:**
- Reduces computation by 4.3%
- Reduces hardware cost by 25%
- Simplifies datapath design and improves efficiency

**Mathematical Transformation:**
The fusion process transforms:
- Original: `x_conv = w_conv * x_in + b`, then `x_bn = (x_conv - μ) / √(σ² + ε)`, then `x_out = w' * x_bn + b'`
- Fused: Direct computation with `w' = γ / √(σ² + ε)` and `b' = b - μ / √(σ² + ε) * γ + β`

**Status:** To be added

### Alpha6_FlexActFunc: Flexible Activation Functions
Flexible activation function implementation supporting multiple activation types.

**Features:**
- ReLU (default)
- ELU
- LeakyReLU
- GELU

**Implementation:**
- `src/SFU/ActivationFunction.v`: Unified activation function module with mode selection
- `Makefile`: Supports all activation function modes combined with SIMD and dataflow configurations

### Alpha7: SFU FSM Implementation
Finite State Machine implementation in SFU for autonomous operation.

**Key Innovation:**
- SFU implements a complete FSM that autonomously manages state transitions
- **Eliminates the need for external control signals** (except `readout_start`)
- States: Init → Accumulation → ReLU → Idle → Readout

**Note:** This FSM design innovation exists in all project versions. Alpha7 serves as a documentation marker.

See [Alpha7/README.md](Alpha/Alpha7/README.md) for detailed FSM documentation.

### Alpha8: ConvNext Application
ConvNext architecture implementation and evaluation.

**Key Innovation:**
- Implementation and evaluation of ConvNext, a modern architecture
- Achieves the same accuracy (~90%, 4-bit) with significantly fewer parameters compared to VGGNet

**Model Comparison:**
- **VGGNet**: 68.7 MB model size
- **ConvNext**: 33.0 MB model size (51% reduction)

**Architecture:**
- Input: 224×224×3
- Multiple ConvNext blocks with downsampling stages
- Feature map progression: 96×96×96 → 28×28×192 → 14×14×384 → 7×7×768
- Layer Normalization and Global Average Pooling
- Final classification with Softmax

**Status:** To be added

## Hardware Structure

Each part/alpha version typically contains:

```
hardware/
├── src/                # Verilog source code
│   ├── core_tb.v      # Testbench
│   ├── core.v         # Top-level core module
│   ├── corelet.v      # Corelet (MAC array + FIFOs)
│   ├── Mac/           # MAC array modules
│   ├── SFU/           # Summation and Function Unit
│   ├── FIFO/          # FIFO modules (L0, OFIFO)
│   └── SRAM/          # SRAM modules
├── golden/            # Test data files
│   ├── *.txt         # Binary format data
│   └── viz/          # Human-readable decimal format
└── Makefile          # Build system
```

## Golden Data Format

- **Binary files** (`*.txt`): Raw binary data for hardware simulation
- **Viz files** (`viz_*.txt`): Human-readable decimal format for debugging
- **Activation files**: `activation_tile*.txt`
- **Weight files**: `weight_itile*_otile*_kij*.txt`
- **Output files**: `out.txt` (ReLU output), `out_raw.txt` (pre-ReLU)

## Build and Simulation

### Using Makefile

```bash
cd Part2/hardware  # or Part3/hardware
make vanilla       # Default mode (4-bit, WS, ReLU)
make act_2b        # 2-bit activation mode
make vanilla_elu   # ELU activation
make os_vanilla    # OS mode
# See Makefile for all available targets
```

### Manual Compilation

```bash
iverilog -f filelist -o compiled
vvp compiled
```

## Key Design Features

1. **Reconfigurable SIMD**: Support for 2-bit and 4-bit activation processing
2. **Reconfigurable Dataflow**: Weight Stationary and Output Stationary modes
3. **Flexible Activation**: Multiple activation functions (ReLU, ELU, LeakyReLU, GELU)
4. **Autonomous SFU**: FSM-based SFU requiring minimal external control
5. **Complete Pipeline**: MAC array → FIFO → SFU → Output

## Documentation

- **Poster**: [Poster/VLSI_Scream_Final_Poster.pdf](Poster/VLSI_Scream_Final_Poster.pdf) - Complete project overview
- **Alpha4**: [Alpha/Alpha4/README.md](Alpha/Alpha4/README.md) - Whole Conv Layer implementation
- **Alpha7**: [Alpha/Alpha7/README.md](Alpha/Alpha7/README.md) - SFU FSM documentation

## License

Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department  
Please do not spread this code without permission
