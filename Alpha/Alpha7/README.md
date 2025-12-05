# Alpha7 - SFU FSM Implementation

## Key Innovation

**Alpha7 implements a Finite State Machine (FSM) in the SFU module, enabling the SFU to operate autonomously without requiring external control signals.**

The FSM manages all internal state transitions automatically, eliminating the need for external control signals to coordinate SFU operations.

## Important Note

**This document is for illustrative purposes only.**

In reality, this FSM implementation innovation already exists in all versions of this project. Alpha7 serves only as a marker point to document this design characteristic.

## FSM States

The SFU FSM consists of five states:

- **S_Init**: Waits for `ofifo_valid` signal
- **S_Acc**: Automatically handles partial sum accumulation (manages `nij` counter 0~35)
- **S_ReLU**: Automatically applies ReLU activation to all output channels (manages `o_nij` counter 0~15)
- **S_Idle**: Waits for `readout_start` signal (only external control point)
- **S_Readout**: Automatically reads all output channel data (manages `o_nij` counter 0~15)

## External Signals

The SFU requires minimal external signals:
- `ofifo_valid`: Triggers accumulation phase
- `readout_start`: Triggers readout phase (only active external control)
- `kij`: Used to determine completion of all kernel processing

All other operations (accumulation, ReLU, readout) are autonomously managed by the FSM.

## Related Files

- `SFU/SFU.v`: Main SFU module with complete FSM implementation
- `SFU/onij_calculator.v`: Output address calculator
- `SFU/ReLU.v`: ReLU activation function module
