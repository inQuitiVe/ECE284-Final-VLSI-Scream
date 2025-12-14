# Alpha 6: Flexible Activation Functions

## Overview

Alpha 6 implements flexible activation function support, allowing the accelerator to switch between multiple activation functions: ReLU, ELU, LeakyReLU, and GELU. This innovation provides hardware-level support for different activation functions without requiring separate implementations.

## Key Innovation

The `ActivationFunction` module provides a unified interface that supports all four activation functions through a mode selection signal (`act_func_mode`), eliminating the need for separate hardware modules for each activation type.

## Features

- **Multiple Activation Functions**: ReLU, ELU, LeakyReLU, and GELU
- **Hardware-Efficient Implementation**: Optimized approximations for complex functions (ELU, GELU)
- **Reconfigurable at Runtime**: Switch activation functions via `act_func_mode` signal
- **Compatible with All Modes**: Works with both 2-bit/4-bit SIMD and OS/WS dataflow modes

## Activation Functions

### ReLU (Rectified Linear Unit)
- **Mode**: `2'b00`
- **Function**: `f(x) = max(0, x)`
- **Implementation**: Direct hardware implementation

### ELU (Exponential Linear Unit)
- **Mode**: `2'b01`
- **Function**: `f(x) = x if x > 0, else α * (e^x - 1)` where α = 1.0
- **Implementation**: Piecewise linear approximation for hardware efficiency

### LeakyReLU
- **Mode**: `2'b10`
- **Function**: `f(x) = x if x > 0, else α * x` where α = 0.01
- **Implementation**: Uses bit-shift approximation (α ≈ 1/64 = 2^-6)

### GELU (Gaussian Error Linear Unit)
- **Mode**: `2'b11`
- **Function**: `f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`
- **Implementation**: Simplified hardware approximation

## Directory Structure

```
Alpha6/
├── src/
│   ├── core_tb.v              # Testbench
│   ├── core.v                 # Top-level core module
│   ├── corelet.v              # Corelet module
│   ├── Mac/                   # MAC array modules
│   ├── SFU/
│   │   ├── SFU.v              # SFU with activation function support
│   │   ├── ActivationFunction.v  # Unified activation function module
│   │   ├── ReLU.v             # Original ReLU module (for reference)
│   │   └── onij_calculator.v  # Output address calculator
│   ├── FIFO/                  # FIFO modules
│   └── SRAM/                  # SRAM modules
├── golden/
│   ├── ws2bit/                # WS mode, 2-bit test data
│   └── ws4bit/                # WS mode, 4-bit test data
├── filelist                   # Verilog file list
└── Makefile                   # Build system with activation function modes
```

## Makefile Usage

### Basic Commands

```bash
cd Alpha/Alpha6
make [target]
```

### Available Targets

The Makefile supports all combinations of:
- **Activation bit-width**: 2-bit or 4-bit
- **Dataflow mode**: Weight Stationary (WS) or Output Stationary (OS)
- **Activation function**: ReLU, ELU, LeakyReLU, or GELU

#### Vanilla Modes (4-bit activation, WS mode)

- **`vanilla`**: ReLU activation (default)
  ```bash
  make vanilla
  ```

- **`vanilla_elu`**: ELU activation
  ```bash
  make vanilla_elu
  ```

- **`vanilla_leaky`**: LeakyReLU activation
  ```bash
  make vanilla_leaky
  ```

- **`vanilla_gelu`**: GELU activation
  ```bash
  make vanilla_gelu
  ```

#### 2-bit Activation Modes (WS mode)

- **`act_2b`**: 2-bit, ReLU
- **`act_2b_elu`**: 2-bit, ELU
- **`act_2b_leaky`**: 2-bit, LeakyReLU
- **`act_2b_gelu`**: 2-bit, GELU

#### OS Vanilla Modes (4-bit activation, OS mode)

- **`os_vanilla`**: OS mode, ReLU
- **`os_vanilla_elu`**: OS mode, ELU
- **`os_vanilla_leaky`**: OS mode, LeakyReLU
- **`os_vanilla_gelu`**: OS mode, GELU

#### OS 2-bit Modes (2-bit activation, OS mode)

- **`os_2b`**: OS mode, 2-bit, ReLU
- **`os_2b_elu`**: OS mode, 2-bit, ELU
- **`os_2b_leaky`**: OS mode, 2-bit, LeakyReLU
- **`os_2b_gelu`**: OS mode, 2-bit, GELU

#### Utility Commands

- **`all`**: Run all vanilla test modes
  ```bash
  make all
  ```

- **`clean`**: Remove compiled files
  ```bash
  make clean
  ```

- **`view`**: View waveform
  ```bash
  make view
  ```

- **`help`**: Show all available targets
  ```bash
  make help
  ```

### Complete Target List

| Target | Activation | Dataflow | Activation Function |
|--------|-----------|----------|-------------------|
| `vanilla` | 4-bit | WS | ReLU |
| `vanilla_elu` | 4-bit | WS | ELU |
| `vanilla_leaky` | 4-bit | WS | LeakyReLU |
| `vanilla_gelu` | 4-bit | WS | GELU |
| `act_2b` | 2-bit | WS | ReLU |
| `act_2b_elu` | 2-bit | WS | ELU |
| `act_2b_leaky` | 2-bit | WS | LeakyReLU |
| `act_2b_gelu` | 2-bit | WS | GELU |
| `os_vanilla` | 4-bit | OS | ReLU |
| `os_vanilla_elu` | 4-bit | OS | ELU |
| `os_vanilla_leaky` | 4-bit | OS | LeakyReLU |
| `os_vanilla_gelu` | 4-bit | OS | GELU |
| `os_2b` | 2-bit | OS | ReLU |
| `os_2b_elu` | 2-bit | OS | ELU |
| `os_2b_leaky` | 2-bit | OS | LeakyReLU |
| `os_2b_gelu` | 2-bit | OS | GELU |

### Compilation Flags

- **`-DACT_2BIT`**: Enable 2-bit activation mode
- **`-DIS_OS`**: Enable Output Stationary dataflow
- **`-DACT_FUNC_ELU`**: Select ELU activation function
- **`-DACT_FUNC_LEAKY`**: Select LeakyReLU activation function
- **`-DACT_FUNC_GELU`**: Select GELU activation function

## Implementation Details

### ActivationFunction Module

The `ActivationFunction` module (`src/SFU/ActivationFunction.v`) provides:
- **Input**: `in[psum_bw-1:0]` - Input data
- **Control**: `act_func_mode[1:0]` - Activation function selection
- **Output**: `out[psum_bw-1:0]` - Activated output

### Hardware Optimizations

1. **ELU Approximation**: Uses piecewise linear segments for negative values to avoid expensive exponential computation
2. **LeakyReLU**: Uses bit-shift (right shift by 6) to approximate 0.01 multiplication
3. **GELU**: Simplified approximation suitable for hardware implementation

## Golden Data

Test data includes expected outputs for different activation functions:
- Standard ReLU outputs in `golden/ws2bit/` and `golden/ws4bit/`
- Additional activation function outputs can be generated using Python scripts in the golden directories

## Integration

The `ActivationFunction` module is integrated into the SFU:
- Replaces the original `ReLU` module instances
- Connected via `act_func_mode` signal from testbench
- Maintains the same interface as the original ReLU module for compatibility

## Notes

- All activation functions are quantized to 16-bit signed integers
- Hardware approximations may differ slightly from software implementations
- The unified module reduces area overhead compared to separate implementations
- Activation function selection is compile-time (via Makefile) in this version
