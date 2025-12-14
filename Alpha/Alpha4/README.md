# SFU for ReLU + MaxPool(2x2, stride=2)

This document describe the behavior and design of this advanced SFU.v, which is actually an extend of Alpha 7 (controller embedded SFU so that TB only need to feed data and minimal control signal)

## Key Features
- Implemented a MaxPool(2x2, stride=2) module and instantaniate it inside our SFU.v
- 

### FSM modifications for MaxPooling
The MaxPooling happens before ReLU, so it's appropriate to

We designed a mechanism that performs MaxPooling without requiring additional Flip Flops. How




The design includes different phases
- PSUM accumulation
- 2×2 MaxPool
- ReLU
- Readout from PSUM memory (PMEM)

---

## High-level overview

- Spatial size before pooling: 4 × 4
- Spatial size after pooling: 2 × 2
- SIMD lanes (channels): col = 8
- Bit-width per lane: psum_bw = 16
- PMEM word width = psum_bw × col

---

## Data layout

### SIMD lane layout

Each PMEM word is a flat vector:

    [ lane7 | lane6 | lane5 | lane4 | lane3 | lane2 | lane1 | lane0 ]

Lane i occupies:

    word[i*psum_bw +: psum_bw]

This corresponds to:

    word[(i+1)*psum_bw-1 : i*psum_bw]

---

### Spatial indexing

- Pre-pooling map: 4 × 4  → addresses 0..15
- Post-pooling map: 2 × 2 → addresses 0..3

---

## mpl_onij_calculator

### Purpose

Performs lane-wise max pooling over a 2×2 spatial window across 4 cycles.

Behavior:
- First cycle initializes max
- Next three cycles compare and update
- Final cycle outputs pooled result

---

### Pooling schedule

order[1:0] meaning:

    00 : start pooling window
    01 : compare
    10 : compare
    11 : final compare → output valid

Address grouping:

    order  0–3  → pooled address 0
    order  4–7  → pooled address 1
    order  8–11 → pooled address 2
    order 12–15 → pooled address 3

---

### Signed comparison (IMPORTANT)

All pooling comparisons use signed arithmetic:

    if ($signed(in_slice) >= $signed(out_q_slice))

Reason:
Verilog part-selects are unsigned by default, even if the parent vector is signed.
Failing to cast will break pooling for negative values.

---

## SFU module

### FSM states

    S_Init    : wait for OFIFO valid
    S_Acc     : accumulate PSUMs
    S_SPF     : MaxPool + ReLU
    S_Idle    : wait for readout
    S_Readout : output PMEM contents

---

### Accumulation stage (S_Acc)

- Iterates nij = 0..35
- Uses onij_calculator to compute output address
- Accumulates PSUM into PMEM
- Flush cycle aligns pipeline
- When kij == 8, transition to SPF stage

---

### SPF stage (S_SPF)

- Reads PMEM using mpl_onij_calculator
- Computes max over 2×2 windows
- Applies ReLU after pooling
- Writes pooled result back to PMEM
- Write enable asserted only when MPL_valid is high

---

### Readout stage (S_Readout)

- Reads PMEM sequentially
- Current implementation reads 4 pooled entries
- Output is directly wired:

    assign readout = Q_pmem;

---

## Timing notes

- PMEM read latency assumed: 1 cycle
- Delayed signals used:
  - ofifo_data_D1 / D2
  - r_A_pmem_D1
  - ren_pmem_D1

If PMEM latency changes, delay alignment must be updated.

---

## Reset behavior

- Pooling registers reset to signed minimum:

    1000...0

This guarantees first comparison always captures real data.

---

## Dependencies

Required modules:
- onij_calculator
- ReLU (parameterized by psum_bw)

---

## Design notes

The address permutation:

    o_nij = {order[3], order[1], order[2], order[0]}

was derived via K-map simplification to minimize logic.
Although non-intuitive, it produces correct spatial mapping.

Debug lane wires are included for waveform inspection.

---

## Summary

This SFU implements a realistic hardware mapping of:
- convolution PSUM accumulation
- spatial max pooling
- activation (ReLU)
- memory-based dataflow control

The design is fully parameterized and extensible.
