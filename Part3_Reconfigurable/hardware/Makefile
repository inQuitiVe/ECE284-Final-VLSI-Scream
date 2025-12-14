# Makefile for ECE284 Final VLSI Scream Project
# Usage: make [target]
#   vanilla      - 4-bit activation mode (WS)
#   act_2b       - 2-bit activation mode (WS)
#   os_vanilla   - OS mode with vanilla settings
#   os_2b        - OS mode with 2-bit activation
#   all          - Run all test modes (default)

# Compiler and flags
IVERILOG = iverilog
VVP = vvp
VIEWER = gtkwave

# File list (relative to sim/ directory)
FILELIST = filelist
FILELIST_WS = filelist_ws

# Default target
.DEFAULT_GOAL := all

# Vanilla mode (no special defines) - WS mode
vanilla:
	@echo "=========================================="
	@echo "Running VANILLA mode (WS)"
	@echo "=========================================="
	cd sim && $(IVERILOG) -f $(FILELIST_WS) -o compiled
	cd sim && $(VVP) compiled

# 2-bit activation mode only - WS mode
act_2b:
	@echo "=========================================="
	@echo "Running ACT_2B mode (WS, 2-bit activation)"
	@echo "=========================================="
	cd sim && $(IVERILOG) -DACT_2BIT -f $(FILELIST_WS) -o compiled
	cd sim && $(VVP) compiled

# OS mode with vanilla settings
os_vanilla:
	@echo "=========================================="
	@echo "Running OS_VANILLA mode (OS + vanilla)"
	@echo "=========================================="
	cd sim && $(IVERILOG) -DIS_OS -f $(FILELIST) -o compiled
	cd sim && $(VVP) compiled

# OS mode with 2-bit activation
os_2b:
	@echo "=========================================="
	@echo "Running OS_2B mode (OS + 2-bit activation)"
	@echo "=========================================="
	cd sim && $(IVERILOG) -DIS_OS -DACT_2BIT -f $(FILELIST) -o compiled
	cd sim && $(VVP) compiled

# Run all test modes sequentially
all: vanilla act_2b os_vanilla os_2b
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
	elif [ -f sim/core_tb_os_vanilla.vcd ]; then \
		$(VIEWER) sim/core_tb_os_vanilla.vcd & \
	elif [ -f sim/core_tb_os_2bit.vcd ]; then \
		$(VIEWER) sim/core_tb_os_2bit.vcd & \
	else \
		echo "Error: VCD file not found. Please run a test first."; \
	fi

# Help message
help:
	@echo "Available targets:"
	@echo "  vanilla      - 4-bit activation mode (WS)"
	@echo "  act_2b       - 2-bit activation mode (WS)"
	@echo "  os_vanilla   - OS mode with vanilla settings"
	@echo "  os_2b        - OS mode with 2-bit activation"
	@echo "  all          - Run all test modes sequentially (default)"
	@echo "  clean        - Remove compiled files and VCD"
	@echo "  view         - View waveform with gtkwave"
	@echo "  help         - Show this help message"

.PHONY: vanilla act_2b os_vanilla os_2b all clean view help
