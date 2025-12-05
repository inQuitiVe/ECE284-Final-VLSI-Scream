# Part 2: 2b/4b Reconfigurable SIMD Version

## Overview

Part 2 extends the vanilla version with reconfigurable SIMD support, allowing the accelerator to switch between 2-bit and 4-bit activation processing modes. This version operates in **Weight Stationary (WS) dataflow mode only**.

## Features

- **Reconfigurable SIMD**: Switchable activation bit-width (2-bit or 4-bit)
- **Weight Stationary Dataflow**: Fixed WS mode for efficient weight reuse
- **ReLU Activation**: Standard ReLU activation function
- **Complete Pipeline**: MAC array → FIFO → SFU → Output

## Directory Structure

```
Part2/
├── hardware/
│   ├── src/              # Verilog source code
│   │   ├── core_tb.v    # Testbench
│   │   ├── core.v       # Top-level core module
│   │   ├── corelet.v    # Corelet (MAC array + FIFOs)
│   │   ├── Mac/         # MAC array modules
│   │   ├── SFU/         # Summation and Function Unit
│   │   ├── FIFO/        # FIFO modules (L0, OFIFO)
│   │   └── SRAM/        # SRAM modules
│   ├── golden/          # Test data files
│   │   ├── ws2bit/      # Test data for 2-bit mode
│   │   └── ws4bit/      # Test data for 4-bit mode
│   ├── filelist         # Verilog file list
│   └── Makefile         # Build system
└── software/
    └── Part2_golden_gen.ipynb  # Golden pattern generator
```

## Golden Data Format

- **Binary files** (`*.txt`): Raw binary data for hardware simulation
- **Viz files** (`viz/*.txt`): Human-readable decimal format for debugging
- **Activation files**: `activation_tile*.txt`
- **Weight files**: `weight_itile*_otile*_kij*.txt`
- **Output files**: 
  - `out.txt`: ReLU output (used by testbench)
  - `out_raw.txt`: Pre-ReLU output (convolution layer output)

## Makefile Usage

### Basic Commands

```bash
cd Part2/hardware
make [target]
```

### Available Targets

#### Simulation Modes

- **`vanilla`** (default): 4-bit activation mode, Weight Stationary
  ```bash
  make vanilla
  ```

- **`act_2b`**: 2-bit activation mode, Weight Stationary
  ```bash
  make act_2b
  ```

#### Utility Commands

- **`all`**: Run all test modes sequentially
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

### Mode Descriptions

| Target | Activation Bits | Dataflow | Description |
|--------|----------------|----------|-------------|
| `vanilla` | 4-bit | WS | Default mode with 4-bit activations |
| `act_2b` | 2-bit | WS | 2-bit activation mode for higher throughput |

### Compilation Details

The Makefile uses `iverilog` for compilation:
- **Vanilla mode**: No special defines
- **2-bit mode**: Uses `-DACT_2BIT` flag

Output files:
- `compiled`: Compiled simulation executable
- `core_tb_2bit.vcd` or `core_tb_vanilla.vcd`: Waveform dump file

## Manual Compilation

If you need to compile manually:

```bash
# 4-bit mode (vanilla)
iverilog -f filelist -o compiled
vvp compiled

# 2-bit mode
iverilog -DACT_2BIT -f filelist -o compiled
vvp compiled
```

## Test Data

Test data is organized by mode:
- **`golden/ws4bit/`**: Contains activation, weight, and output files for 4-bit mode
- **`golden/ws2bit/`**: Contains activation, weight, and output files for 2-bit mode

Each directory includes:
- Activation tile files
- Weight files for each input/output tile and kernel iteration
- Expected output files
- Visualization files in `viz/` subdirectory

## Notes

- Part 2 only supports **Weight Stationary (WS)** dataflow mode
- Output Stationary (OS) mode is available in Part 3
- The testbench automatically selects the correct golden data based on the activation mode
