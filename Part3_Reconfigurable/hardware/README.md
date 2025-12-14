# Part 3: OS/WS Reconfigurable Version

## Overview

Part 3 extends Part 2 by adding **Output Stationary (OS) dataflow** support in addition to Weight Stationary (WS). This version provides full reconfigurability for both SIMD (2-bit/4-bit) and dataflow (OS/WS) modes.

## Features

- **Reconfigurable SIMD**: Switchable activation bit-width (2-bit or 4-bit)
- **Reconfigurable Dataflow**: Switchable between Weight Stationary (WS) and Output Stationary (OS) modes
- **ReLU Activation**: Standard ReLU activation function
- **Complete System Integration**: Full pipeline with all reconfiguration options

## Directory Structure

```
Part3/
├── src/                 # Verilog source code
│   ├── core_tb.v       # Testbench
│   ├── core.v          # Top-level core module
│   ├── corelet.v       # Corelet (MAC array + FIFOs)
│   ├── mac/            # MAC array modules
│   ├── SFU/            # Summation and Function Unit
│   ├── fifo/           # FIFO modules (L0, OFIFO)
│   └── sram/           # SRAM modules
├── golden/             # Test data files
│   ├── ws2bit/         # WS mode, 2-bit activation data
│   ├── ws4bit/         # WS mode, 4-bit activation data
│   ├── os2bit/         # OS mode, 2-bit activation data
│   └── os4bit/         # OS mode, 4-bit activation data
├── sim/
│   ├── filelist        # OS mode file list (default)
│   └── filelist_ws     # WS mode file list
└── Makefile            # Build system
```

## Golden Data Format

- **Binary files** (`*.txt`): Raw binary data for hardware simulation
- **Viz files** (`viz/*.txt`): Human-readable decimal format for debugging
- **Activation files**: `activation_tile*.txt`
- **Weight files**: `weight_itile*_otile*_kij*.txt`
- **Output files**: 
  - `output.txt`: Final output after ReLU activation

## Makefile Usage

### Basic Commands

```bash
cd Part3
make [target]
```

### Available Targets

#### Vanilla Modes (4-bit activation, WS mode)

- **`vanilla`** (default): 4-bit activation, Weight Stationary, ReLU
  ```bash
  make vanilla
  ```

- **`vanilla_elu`**: 4-bit activation, Weight Stationary, ELU
  ```bash
  make vanilla_elu
  ```

- **`vanilla_leaky`**: 4-bit activation, Weight Stationary, LeakyReLU
  ```bash
  make vanilla_leaky
  ```

- **`vanilla_gelu`**: 4-bit activation, Weight Stationary, GELU
  ```bash
  make vanilla_gelu
  ```

#### 2-bit Activation Modes (WS mode)

- **`act_2b`**: 2-bit activation, Weight Stationary, ReLU
  ```bash
  make act_2b
  ```

- **`act_2b_elu`**: 2-bit activation, Weight Stationary, ELU
  ```bash
  make act_2b_elu
  ```

- **`act_2b_leaky`**: 2-bit activation, Weight Stationary, LeakyReLU
  ```bash
  make act_2b_leaky
  ```

- **`act_2b_gelu`**: 2-bit activation, Weight Stationary, GELU
  ```bash
  make act_2b_gelu
  ```

#### OS Vanilla Modes (4-bit activation, OS mode)

- **`os_vanilla`**: 4-bit activation, Output Stationary, ReLU
  ```bash
  make os_vanilla
  ```

- **`os_vanilla_elu`**: 4-bit activation, Output Stationary, ELU
  ```bash
  make os_vanilla_elu
  ```

- **`os_vanilla_leaky`**: 4-bit activation, Output Stationary, LeakyReLU
  ```bash
  make os_vanilla_leaky
  ```

- **`os_vanilla_gelu`**: 4-bit activation, Output Stationary, GELU
  ```bash
  make os_vanilla_gelu
  ```

#### OS 2-bit Modes (2-bit activation, OS mode)

- **`os_2b`**: 2-bit activation, Output Stationary, ReLU
  ```bash
  make os_2b
  ```

- **`os_2b_elu`**: 2-bit activation, Output Stationary, ELU
  ```bash
  make os_2b_elu
  ```

- **`os_2b_leaky`**: 2-bit activation, Output Stationary, LeakyReLU
  ```bash
  make os_2b_leaky
  ```

- **`os_2b_gelu`**: 2-bit activation, Output Stationary, GELU
  ```bash
  make os_2b_gelu
  ```

#### Utility Commands

- **`all`**: Run all vanilla test modes sequentially
  ```bash
  make all
  ```

- **`clean`**: Remove compiled files and VCD waveforms
  ```bash
  make clean
  ```

- **`view`**: View waveform with gtkwave (requires VCD file)
  ```bash
  make view
  ```

- **`help`**: Show all available targets
  ```bash
  make help
  ```

### Mode Summary Table

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

The Makefile uses the following compilation flags:
- **`-DACT_2BIT`**: Enable 2-bit activation mode
- **`-DIS_OS`**: Enable Output Stationary dataflow mode
- **`-DACT_FUNC_ELU`**: Use ELU activation function
- **`-DACT_FUNC_LEAKY`**: Use LeakyReLU activation function
- **`-DACT_FUNC_GELU`**: Use GELU activation function

## Manual Compilation

Example manual compilation commands:

```bash
# 4-bit, WS, ReLU (vanilla)
iverilog -f sim/filelist_ws -o compiled
vvp compiled

# 2-bit, WS, ReLU
iverilog -DACT_2BIT -f sim/filelist_ws -o compiled
vvp compiled

# 4-bit, OS, ReLU
iverilog -DIS_OS -f sim/filelist -o compiled
vvp compiled

# 2-bit, OS, ELU
iverilog -DACT_2BIT -DIS_OS -DACT_FUNC_ELU -f sim/filelist -o compiled
vvp compiled
```

## Test Data Organization

Test data is organized by mode:
- **`golden/ws4bit/`**: Weight Stationary, 4-bit activation
- **`golden/ws2bit/`**: Weight Stationary, 2-bit activation
- **`golden/os4bit/`**: Output Stationary, 4-bit activation
- **`golden/os2bit/`**: Output Stationary, 2-bit activation

The testbench automatically selects the correct golden data based on the compilation flags.

## Dataflow Modes

### Weight Stationary (WS)
- Weights are stationary in the MAC array
- Activations flow through the array
- Efficient for weight reuse scenarios

### Output Stationary (OS)
- Output partial sums are stationary
- Weights and activations flow through the array
- Efficient for different access patterns

## Notes

- Part 3 provides full reconfigurability for both SIMD and dataflow modes
- All activation functions (ReLU, ELU, LeakyReLU, GELU) are available in all modes
- The testbench automatically configures based on Makefile target selection

