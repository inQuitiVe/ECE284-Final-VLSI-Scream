# Makefile for ECE284 Final VLSI Scream Project
# Usage: make [target]
#   vanilla         - Default mode (ReLU activation, no special defines)
#   vanilla_elu     - Vanilla mode with ELU activation
#   vanilla_leaky   - Vanilla mode with LeakyReLU activation
#   vanilla_gelu    - Vanilla mode with GELU activation
#   act_2b          - 2-bit activation mode only (ReLU)
#   act_2b_elu      - 2-bit activation mode with ELU
#   act_2b_leaky    - 2-bit activation mode with LeakyReLU
#   act_2b_gelu     - 2-bit activation mode with GELU
#   os_vanilla      - OS mode with vanilla settings (ReLU)
#   os_vanilla_elu  - OS mode with ELU activation
#   os_vanilla_leaky- OS mode with LeakyReLU activation
#   os_vanilla_gelu - OS mode with GELU activation
#   os_2b           - OS mode with 2-bit activation (ReLU)
#   os_2b_elu       - OS mode with 2-bit activation and ELU
#   os_2b_leaky     - OS mode with 2-bit activation and LeakyReLU
#   os_2b_gelu      - OS mode with 2-bit activation and GELU
#   all             - Run all vanilla test modes

# Compiler and flags
IVERILOG = iverilog
VVP = vvp
VIEWER = gtkwave

# File list
FILELIST = filelist

# Default target
.DEFAULT_GOAL := vanilla

# Vanilla mode (ReLU activation, no special defines)
vanilla:
	@echo "=========================================="
	@echo "Running VANILLA mode (ReLU, default)"
	@echo "=========================================="
	$(IVERILOG) -f $(FILELIST) -o compiled
	$(VVP) compiled

# Vanilla mode with ELU activation
vanilla_elu:
	@echo "=========================================="
	@echo "Running VANILLA_ELU mode (ELU activation)"
	@echo "=========================================="
	$(IVERILOG) -DACT_FUNC_ELU -f $(FILELIST) -o compiled
	$(VVP) compiled

# Vanilla mode with LeakyReLU activation
vanilla_leaky:
	@echo "=========================================="
	@echo "Running VANILLA_LEAKY mode (LeakyReLU activation)"
	@echo "=========================================="
	$(IVERILOG) -DACT_FUNC_LEAKY -f $(FILELIST) -o compiled
	$(VVP) compiled

# Vanilla mode with GELU activation
vanilla_gelu:
	@echo "=========================================="
	@echo "Running VANILLA_GELU mode (GELU activation)"
	@echo "=========================================="
	$(IVERILOG) -DACT_FUNC_GELU -f $(FILELIST) -o compiled
	$(VVP) compiled

# 2-bit activation mode only (ReLU)
act_2b:
	@echo "=========================================="
	@echo "Running ACT_2B mode (2-bit activation, ReLU)"
	@echo "=========================================="
	$(IVERILOG) -DACT_2BIT -f $(FILELIST) -o compiled
	$(VVP) compiled

# 2-bit activation mode with ELU
act_2b_elu:
	@echo "=========================================="
	@echo "Running ACT_2B_ELU mode (2-bit activation, ELU)"
	@echo "=========================================="
	$(IVERILOG) -DACT_2BIT -DACT_FUNC_ELU -f $(FILELIST) -o compiled
	$(VVP) compiled

# 2-bit activation mode with LeakyReLU
act_2b_leaky:
	@echo "=========================================="
	@echo "Running ACT_2B_LEAKY mode (2-bit activation, LeakyReLU)"
	@echo "=========================================="
	$(IVERILOG) -DACT_2BIT -DACT_FUNC_LEAKY -f $(FILELIST) -o compiled
	$(VVP) compiled

# 2-bit activation mode with GELU
act_2b_gelu:
	@echo "=========================================="
	@echo "Running ACT_2B_GELU mode (2-bit activation, GELU)"
	@echo "=========================================="
	$(IVERILOG) -DACT_2BIT -DACT_FUNC_GELU -f $(FILELIST) -o compiled
	$(VVP) compiled

# OS mode with vanilla settings (ReLU)
os_vanilla:
	@echo "=========================================="
	@echo "Running OS_VANILLA mode (OS + vanilla, ReLU)"
	@echo "=========================================="
	$(IVERILOG) -DIS_OS -f $(FILELIST) -o compiled
	$(VVP) compiled

# OS mode with ELU activation
os_vanilla_elu:
	@echo "=========================================="
	@echo "Running OS_VANILLA_ELU mode (OS + ELU)"
	@echo "=========================================="
	$(IVERILOG) -DIS_OS -DACT_FUNC_ELU -f $(FILELIST) -o compiled
	$(VVP) compiled

# OS mode with LeakyReLU activation
os_vanilla_leaky:
	@echo "=========================================="
	@echo "Running OS_VANILLA_LEAKY mode (OS + LeakyReLU)"
	@echo "=========================================="
	$(IVERILOG) -DIS_OS -DACT_FUNC_LEAKY -f $(FILELIST) -o compiled
	$(VVP) compiled

# OS mode with GELU activation
os_vanilla_gelu:
	@echo "=========================================="
	@echo "Running OS_VANILLA_GELU mode (OS + GELU)"
	@echo "=========================================="
	$(IVERILOG) -DIS_OS -DACT_FUNC_GELU -f $(FILELIST) -o compiled
	$(VVP) compiled

# OS mode with 2-bit activation (ReLU)
os_2b:
	@echo "=========================================="
	@echo "Running OS_2B mode (OS + 2-bit activation, ReLU)"
	@echo "=========================================="
	$(IVERILOG) -DIS_OS -DACT_2BIT -f $(FILELIST) -o compiled
	$(VVP) compiled

# OS mode with 2-bit activation and ELU
os_2b_elu:
	@echo "=========================================="
	@echo "Running OS_2B_ELU mode (OS + 2-bit activation, ELU)"
	@echo "=========================================="
	$(IVERILOG) -DIS_OS -DACT_2BIT -DACT_FUNC_ELU -f $(FILELIST) -o compiled
	$(VVP) compiled

# OS mode with 2-bit activation and LeakyReLU
os_2b_leaky:
	@echo "=========================================="
	@echo "Running OS_2B_LEAKY mode (OS + 2-bit activation, LeakyReLU)"
	@echo "=========================================="
	$(IVERILOG) -DIS_OS -DACT_2BIT -DACT_FUNC_LEAKY -f $(FILELIST) -o compiled
	$(VVP) compiled

# OS mode with 2-bit activation and GELU
os_2b_gelu:
	@echo "=========================================="
	@echo "Running OS_2B_GELU mode (OS + 2-bit activation, GELU)"
	@echo "=========================================="
	$(IVERILOG) -DIS_OS -DACT_2BIT -DACT_FUNC_GELU -f $(FILELIST) -o compiled
	$(VVP) compiled

# Run all vanilla test modes sequentially
all: vanilla vanilla_elu vanilla_leaky vanilla_gelu
	@echo "=========================================="
	@echo "All vanilla test modes completed!"
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
	@echo "Available targets (all based on vanilla):"
	@echo ""
	@echo "Vanilla modes (4-bit activation, WS mode):"
	@echo "  vanilla         - Default mode (ReLU activation)"
	@echo "  vanilla_elu     - Vanilla mode with ELU activation"
	@echo "  vanilla_leaky   - Vanilla mode with LeakyReLU activation"
	@echo "  vanilla_gelu    - Vanilla mode with GELU activation"
	@echo ""
	@echo "2-bit activation modes (WS mode):"
	@echo "  act_2b          - 2-bit activation mode (ReLU)"
	@echo "  act_2b_elu     - 2-bit activation mode with ELU"
	@echo "  act_2b_leaky   - 2-bit activation mode with LeakyReLU"
	@echo "  act_2b_gelu    - 2-bit activation mode with GELU"
	@echo ""
	@echo "OS vanilla modes (4-bit activation, OS mode):"
	@echo "  os_vanilla      - OS mode with vanilla settings (ReLU)"
	@echo "  os_vanilla_elu  - OS mode with ELU activation"
	@echo "  os_vanilla_leaky- OS mode with LeakyReLU activation"
	@echo "  os_vanilla_gelu - OS mode with GELU activation"
	@echo ""
	@echo "OS 2-bit modes (2-bit activation, OS mode):"
	@echo "  os_2b           - OS mode with 2-bit activation (ReLU)"
	@echo "  os_2b_elu       - OS mode with 2-bit activation and ELU"
	@echo "  os_2b_leaky     - OS mode with 2-bit activation and LeakyReLU"
	@echo "  os_2b_gelu      - OS mode with 2-bit activation and GELU"
	@echo ""
	@echo "Other targets:"
	@echo "  all             - Run all vanilla test modes sequentially"
	@echo "  clean           - Remove compiled files and VCD"
	@echo "  view            - View waveform with gtkwave"
	@echo "  help            - Show this help message"

.PHONY: vanilla vanilla_elu vanilla_leaky vanilla_gelu \
        act_2b act_2b_elu act_2b_leaky act_2b_gelu \
        os_vanilla os_vanilla_elu os_vanilla_leaky os_vanilla_gelu \
        os_2b os_2b_elu os_2b_leaky os_2b_gelu \
        all clean view help
