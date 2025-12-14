# Part 1: Vanilla Version with Alpha 7 (SFU as controller)

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
Part1_Vanilla/
├── hardware/              # Hardware implementation
│   ├── verilog/           # All HDL sources
│   │   ├── core_tb.v      # Testbench
│   │   ├── core.v         # Top-level core module
│   │   ├── corelet.v      # Corelet module
│   │   ├── SFU/           # Summation and Function Unit
│   │   │   ├── SFU.v      # SFU main module
│   │   │   ├── ReLU.v     # ReLU activation function
│   │   │   └── onij_calculator.v  # Output address calculator
│   │   ├── Mac/           # MAC array modules
│   │   ├── FIFO/          # FIFO modules
│   │   └── SRAM/          # SRAM modules
│   ├── datafiles/         # Input files used by the testbench
│   │   ├── activation_tile0.txt
│   │   ├── weight_itile0_otile0_kij*.txt
│   │   ├── out.txt        # Expected output
│   │   └── viz/           # Human-readable format
│   │       └── viz_out.txt
│   └── sim/               # Simulation files and the runtime filelist
│       └── filelist       # REQUIRED: Plain text file with relative paths (../verilog/) to design files
└── software/
    └── Part1_golden_gen.ipynb  # Golden pattern generator
```


### SFU (Special Function Unit)
For the key features and design concept for Alpha 7, see Alpha 7 folder.
- Accumulates partial sums from MAC array
- Applies ReLU activation function
- Manages PSUM memory read/write operations


## Testbench Usage

### Using Makefile (Recommended)

```bash
cd Part1_Vanilla/hardware
make vanilla
```

### Manual Compilation

```bash
cd Part1_Vanilla/hardware
iverilog -f sim/filelist -o compiled
vvp compiled
```

The correct results will be printed out here.


## Testbench Design
1. tb writes activation.txt in SRAM(xmem), starting from address `11'b00000000000`. 
2. tb writes kij0 weight.txt in SRAM(xmem), starting from address `11'b10000000000`. 
3. tb feeds SRAM read address/enable and instruction(kernel load/execution) to `core.v`. Note that the instructions are delayed 1 cycle from the read control signals, in order to align with the 1-cycle latency of SRAM.
4. SFU do the calculation snd accumulation. After that, core.v enters IDLE state, waiting for the reset signal.
5. Repeat 2.~4. for kij=1,2...8
6. Note that after accumulation state of kij=8, SFU performs ReLU, then enters IDLE state to wait for tb readout signal.
7. Finally, tb compares output with golden data, and report any mismatches


## Data Format

The test data files are located in `datafiles/`:
- **`activation_tile0.txt`**: Input activation data in binary format
- **`weight_itile0_otile0_kij*.txt`**: Weight data files (kij: 0-8)
- **`out.txt`**: Expected output in binary format
- **`viz/viz_out.txt`**: Human-readable decimal format for verification



## Notes
- This is the baseline version without reconfiguration features
- All subsequent parts (Part2, Part3) extend this basic implementation
- Used as reference for understanding the core architecture




























