# Makefile for ECE284 Final VLSI Scream Project
# Usage: make [target]
#   vanilla      - 4-bit activation mode (WS only)
#   act_2b       - 2-bit activation mode (WS only)
#   all          - Run all test modes (default)

# Compiler and flags
IVERILOG = iverilog
VVP = vvp
VIEWER = gtkwave

# File list (relative to sim/ directory)
FILELIST = filelist

# Default target
.DEFAULT_GOAL := all

# Vanilla mode (no special defines)
vanilla:
	@echo "=========================================="
	@echo "Running VANILLA mode (default)"
	@echo "=========================================="
	cd sim && $(IVERILOG) -f $(FILELIST) -o compiled
	cd sim && $(VVP) compiled

# 2-bit activation mode only
act_2b:
	@echo "=========================================="
	@echo "Running ACT_2B mode (2-bit activation)"
	@echo "=========================================="
	cd sim && $(IVERILOG) -DACT_2BIT -f $(FILELIST) -o compiled
	cd sim && $(VVP) compiled

# Run all test modes sequentially
all: vanilla act_2b
	@echo "=========================================="
	@echo "All test modes completed!"
	@echo "=========================================="

# Clean compiled files
clean:
	@echo "Cleaning compiled files..."
	rm -f sim/compiled sim/*.vcd
	@echo "Clean completed!"

# View waveform (requires gtkwave)
view:
	@if [ -f sim/core_tb_vanilla.vcd ]; then \
		$(VIEWER) sim/core_tb_vanilla.vcd & \
	elif [ -f sim/core_tb_2bit.vcd ]; then \
		$(VIEWER) sim/core_tb_2bit.vcd & \
	else \
		echo "Error: VCD file not found. Please run a test first."; \
	fi

# Help message
help:
	@echo "Available targets:"
	@echo "  vanilla      - 4-bit activation mode (WS only)"
	@echo "  act_2b       - 2-bit activation mode (WS only)"
	@echo "  all          - Run all test modes sequentially (default)"
	@echo "  clean        - Remove compiled files and VCD"
	@echo "  view         - View waveform with gtkwave"
	@echo "  help         - Show this help message"

.PHONY: vanilla act_2b all clean view help
