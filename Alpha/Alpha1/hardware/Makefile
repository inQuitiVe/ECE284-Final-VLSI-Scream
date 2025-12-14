# Makefile for ECE284 Final VLSI Scream Project
# Usage: make [target]
#   vanilla      - Default mode (no special defines)
#   act_2b       - 2-bit activation mode only
#   os_vanilla   - OS mode with vanilla settings
#   os_2b        - OS mode with 2-bit activation
#   all          - Run all test modes

# Compiler and flags
IVERILOG = iverilog
VVP = vvp
VIEWER = gtkwave

# File list
FILELIST = filelist
FILELIST_OS = filelist_os

# Default target
.DEFAULT_GOAL := vanilla

# Vanilla mode (no special defines)
vanilla:
	@echo "=========================================="
	@echo "Running VANILLA mode (default)"
	@echo "=========================================="
	$(IVERILOG) -f $(FILELIST) -o compiled
	$(VVP) compiled

# 2-bit activation mode only
act_2b:
	@echo "=========================================="
	@echo "Running ACT_2B mode (2-bit activation)"
	@echo "=========================================="
	$(IVERILOG) -DACT_2BIT -f $(FILELIST) -o compiled
	$(VVP) compiled

# OS mode with vanilla settings
os_vanilla:
	@echo "=========================================="
	@echo "Running OS_VANILLA mode (OS + vanilla)"
	@echo "=========================================="
	$(IVERILOG) -DIS_OS -f $(FILELIST_OS) -o compiled
	$(VVP) compiled

# OS mode with 2-bit activation
os_2b:
	@echo "=========================================="
	@echo "Running OS_2B mode (OS + 2-bit activation)"
	@echo "=========================================="
	$(IVERILOG) -DIS_OS -DACT_2BIT -f $(FILELIST_OS) -o compiled
	$(VVP) compiled

# Run all test modes sequentially
all: vanilla act_2b os_vanilla os_2b
	@echo "=========================================="
	@echo "All test modes completed!"
	@echo "=========================================="

# Clean compiled files
clean:
	@echo "Cleaning compiled files..."
	rm -f compiled core_tb.vcd
	@echo "Clean completed!"

# View waveform (requires gtkwave)
view:
	@if [ -f core_tb.vcd ]; then \
		$(VIEWER) core_tb.vcd & \
	else \
		echo "Error: core_tb.vcd not found. Please run a test first."; \
	fi

# Help message
help:
	@echo "Available targets:"
	@echo "  vanilla      - Default mode (no special defines)"
	@echo "  act_2b       - 2-bit activation mode only"
	@echo "  os_vanilla   - OS mode with vanilla settings"
	@echo "  os_2b        - OS mode with 2-bit activation"
	@echo "  all          - Run all test modes sequentially"
	@echo "  clean        - Remove compiled files and VCD"
	@echo "  view         - View waveform with gtkwave"
	@echo "  help         - Show this help message"

.PHONY: vanilla act_2b os_vanilla os_2b all clean view help
