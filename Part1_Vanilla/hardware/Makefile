# Makefile for ECE284 Final VLSI Scream Project - Part1
# Usage: make [target]

# Compiler and flags
IVERILOG = iverilog
VVP = vvp
VIEWER = gtkwave

# File list
FILELIST = sim/filelist

# Default target
.DEFAULT_GOAL := vanilla

# Vanilla mode (no special defines)
vanilla:
	@echo "=========================================="
	@echo "Running VANILLA mode (default)"
	@echo "=========================================="
	$(IVERILOG) -f $(FILELIST) -o compiled
	$(VVP) compiled

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
	@echo "  vanilla      - Default mode (4-bit, WS)"
	@echo "  clean        - Remove compiled files and VCD"
	@echo "  view         - View waveform with gtkwave"
	@echo "  help         - Show this help message"

.PHONY: vanilla clean view help


