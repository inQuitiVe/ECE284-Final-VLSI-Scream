# SFU for ReLU + MaxPool(2x2, stride=2)

This document describe the behavior and design of this advanced SFU.v, which is actually an extend of Alpha 7 (controller embedded SFU so that TB only need to feed data and minimal control signal)

## Key Features
- Implemented a MaxPool(2x2, stride=2) module and instantaniate it inside our SFU.v
- The MaxPool module only requires 1 additional Flip Flop to calculate the 2x2 output of the 4x4 input.

### FSM modifications for MaxPooling
MaxPooling happens right before ReLU. While the original state of ReLU operation is S_ReLU, here we redesign the purpose of the state to be "SFU of the SFU". 

In this state, MaxPooling and ReLU is performed. If there's other operation, it can be added here also. Therefore, the existence of this state provides a really good structure for potential future features.

### MaxPooling mechanism
#### High-level overview
- Spatial size before pooling: 4 × 4
- Spatial size after pooling: 2 × 2
- SIMD lanes (channels): col = 8
- Bit-width per lane: psum_bw = 16
- PMEM word width = psum_bw × col

ReLU can be applied in any `o_nij` order. However, the MaxPool hardware can be minimized by using a specially designed `o_nij` read sequence. 
```
MPL_nij0 = max{o_nij: 0, 1, 4, 5}
MPL_nij1 = max{o_nij: 2, 3, 6, 7}
MPL_nij2 = max{o_nij: 8, 9, C, D}
MPL_nij3 = max{o_nij: A, B, E, F}
```

The MPL2D unit updates the current maximum when the input value is larger and resets every 4 cycles.

Moreover, the output address of MPL_nij always corresponds to a previously accessed o_nij, allowing the result to be written back directly to PSUM memory.

### ReLU mechanism
The ReLU is cascaded after the MaxPooling Output.