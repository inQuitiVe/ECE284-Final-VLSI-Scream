
## Testbench Usage
Ihe instructions are same as what we used in class.
```bash
cd hardware/
iveri filelist
irun compiled

### The correct results will be printed out here.
```


## Testbench Design
1. tb writes activation.txt in SRAM(xmem), starting from address `11'b00000000000`. 
2. tb writes kij0 weight.txt in SRAM(xmem), starting from address `11'b10000000000`. 
3. tb feeds SRAM read address/enable and instruction(kernel load/execution) to `core.v`. Note that the instructions are delayed 1 cycle from the read control signals, in order to align with the 1-cycle latency of SRAM.
4. SFU do the calculation snd accumulation. After that, core.v enters IDLE state, waiting for the reset signal.
5. Repeat 2.~4. for kij=1,2...8
6. Note that after accumulation state of kij=8, SFU performs ReLU, then enters IDLE state to wait for tb readout signal.


## Golden Pattern Format
The pattern has the same format as the assignments.