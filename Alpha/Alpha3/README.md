# Alpha 3: Huffman Decoder

## Overview

Alpha 3 implements a Huffman decoder for data compression/decompression. This module decodes Huffman-encoded bit streams back into original symbols, useful for decompressing compressed neural network weights or activations.

## Features

- **Serial Bit Input**: Processes Huffman-encoded data bit by bit
- **State Machine Based**: Uses FSM to traverse Huffman tree
- **Symbol Output**: Decodes bits into 8-bit symbols
- **Valid Signal**: Indicates when decoded symbol is ready

## Directory Structure

```
Alpha3/
└── huffman_decoder.v    # Huffman decoder module
```

## Module Description

### huffman_decoder.v

Huffman decoder that decodes serial bit streams using a state machine to traverse a Huffman tree.

**Ports:**
- `clk`: Clock signal
- `rst_n`: Active low reset
- `bit_in`: Serial input bit (Huffman-encoded)
- `data_valid`: High when `bit_in` is valid
- `char_out[7:0]`: Decoded symbol output
- `char_valid`: High for 1 cycle when `char_out` is valid

**Symbol Constants:**
- Supports symbols 0-7 (8'h00 to 8'h07)
- Can be extended for larger symbol sets

**State Machine:**
- `S_ROOT`: Root state
- `S_NODE_1` to `S_NODE_6`: Internal tree nodes
- State transitions based on input bits (0 = left, 1 = right)

## Usage

### Basic Integration

```verilog
huffman_decoder decoder (
    .clk(clk),
    .rst_n(reset_n),
    .bit_in(compressed_bit),
    .data_valid(bit_valid),
    .char_out(decoded_symbol),
    .char_valid(symbol_valid)
);
```

### Testing

```bash
# Compile and test
iverilog -o test_huffman huffman_decoder.v huffman_tb.v
vvp test_huffman
```

## Operation Flow

1. **Reset**: Initialize to root state
2. **Bit Input**: Receive Huffman-encoded bits via `bit_in`
3. **Tree Traversal**: FSM traverses Huffman tree based on input bits
4. **Symbol Decode**: When reaching a leaf node, output the corresponding symbol
5. **Valid Signal**: Assert `char_valid` for one cycle when symbol is ready
6. **Reset**: Return to root state for next symbol

## Application

This decoder can be used for:
- **Weight Decompression**: Decompress Huffman-encoded neural network weights
- **Activation Decompression**: Decompress compressed activation data
- **Data Storage Optimization**: Reduce memory requirements by storing compressed data

## Implementation Details

- **Tree Structure**: Hardcoded Huffman tree based on symbol frequencies
- **State Encoding**: Uses 4-bit state encoding for tree nodes
- **Serial Processing**: Processes one bit per clock cycle
- **Synchronous Design**: All operations synchronized to clock

## Notes

- The Huffman tree structure is fixed at design time
- Can be modified to support different symbol sets or tree structures
- Useful for reducing memory bandwidth and storage requirements
- May be integrated into the data path for on-the-fly decompression

