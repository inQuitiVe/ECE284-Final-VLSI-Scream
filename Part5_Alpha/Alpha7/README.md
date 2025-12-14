# Alpha7 - SFU FSM Implementation

## Key Features
- Implemented a Finite State Machine (FSM) in the SFU module, enabling the SFU to operate autonomously with minimal control signals.
- Designed the readout mechanism so that tb doesn't have direct access to `p_mem`. This avoids the potential control signal racing condition on SRAM.
- Reduced PSUM memory size to 4.93%\
Original: $nij \times och \times kij \times psum\_ bw$ (5.0625MB) \
Our Design: $o\_ nij \times och \times psum\_ bw$  (0.25MB)

## Important Notes
In reality, this FSM implementation innovation already exists in all versions of this project. Alpha7 serves only as a marker point to document this design characteristic.

## FSM States
The SFU FSM consists of five states:

- **S_Init**: Waits for `ofifo_valid` signal
- **S_Acc**: Automatically handles partial sum accumulation (manages `nij` counter 0~35)
- **S_ReLU**: Automatically applies ReLU activation to all output channels (manages `o_nij` counter 0~15)
- **S_Idle**: Waits for `readout_start` signal (only external control point)
- **S_Readout**: Automatically reads all output channel data (manages `o_nij` counter 0~15)

## SFU overview
### FSM states
```
S_Init    : wait for OFIFO valid
S_Acc     : accumulate PSUMs
S_SPF     : MaxPool + ReLU
S_Idle    : wait for readout
S_Readout : output PMEM contents
```
---
#### Accumulation stage (S_Acc)

- Iterates nij counter = 0..35
- Uses onij_calculator to compute output address
- Accumulates PSUM into PMEM
- Flush cycle aligns pipeline
- When kij == 8, transition to SPF stage
---
#### SPF stage (S_SPF)
- Reads PMEM using o_nij counter = 0..15
- Applies ReLU and write the result back to PMEM
---
#### Readout stage (S_Readout)
- Reads PMEM sequentially using o_nij counter = 0..15
- Output is directly wired:
    assign readout = Q_pmem;
---


## External Control Signals
The SFU requires minimal external signals:
- `ofifo_valid`: Triggers accumulation phase
- `readout_start`: Triggers readout phase (only active external control)
- `kij`: Used to determine completion of all kernel processing

All other operations (accumulation, ReLU, readout) are autonomously managed by the FSM. 

The reason that `kij` is not managed by SFU, is that the MAC array needs a reset to reload the weights. It's not resonable to design the SFU to memorize the kij internally while a module-wide reset is performed.


## onij_calculator.v
One of the key modules of `SFU.v`, it decides whether the received PSUM has to be accumulate on output, and which address to accumulate. This smart design makes the required memory size to shrink from 5.0625MB $(nij \times och \times kij \times psum\_bw)$ to 0.25MB $(o\_nij \times och \times psum\_bw)$

It has two inputs: `nij` and `kij`. The former is counted by SFU by the cycles of o_valid, while the latter is given directly by testbench. 

This module implements constant divider division by lookup table, and modulo based on the result of division. 

## Output SRAM
The special designed SRAM has the capability of read and write at the same time. It has a read enable port, read address input, write enable port and write address input. This eliminates the need of inserting bubble cycles bewteen MACs, since the SRAM can handle read and write request from SFU at the same time.


## Related Files

- `SFU/SFU.v`: Main SFU module with complete FSM implementation
- `SFU/onij_calculator.v`: Output address calculator
- `SFU/ReLU.v`: ReLU activation function module


## Testbench Usage
See Part 1.