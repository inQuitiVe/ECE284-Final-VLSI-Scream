# SFU for ReLU + MaxPool(2x2, stride=2)

This document describe the behavior and design of this advanced SFU.v, which is actually an extend of Alpha 7 (controller embedded SFU so that TB only need to feed data and minimal control signal)

## Key Features
- Implemented a MaxPool(2x2, stride=2) module and instantaniate it inside our SFU.v
- The MaxPool module only requires 1 additional Flip Flop to calculate the 2x2 output of the 4x4 input.
- Adds a `bias` input port. This provides hardware support for the entire Conv–BatchNorm–MaxPool–ReLU stack.
BatchNorm multiplication parameters can be folded directly into the convolution weights through model fusion, while the bias term, MaxPool and ReLU are implemented as a post-processing stage after accumulation.

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


## Testbench Usage
Ihe instructions are same as what we used in class.
```bash
cd Alpha4/
iveri filelist
irun compiled

### The correct results will be printed out here.
```

## Testbench Design
1. tb feeds kij and psum.txt to SFU.
2. SFU performs accumulation (including bias) for kij=0..8
3. After accumulation of kij=8, SFU enters a state that performs MaxPool and ReLU, and save the result in output memory
4. TB sends a readout signal.

#### Notes
The bias functionality is not verified by golden pattern, however, we believe that it's trivial enough to