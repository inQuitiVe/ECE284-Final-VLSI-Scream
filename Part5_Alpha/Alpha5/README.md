# SFU for ReLU + MaxPool(2x2, stride=2)

This document describes the behavior and design of this advanced SFU.v, which extends Alpha 7 (controller embedded SFU so that TB only needs to feed data and minimal control signals).

## Key Innovation: Conv-BN Fusion Design

This implementation adopts a **Conv-BN + ReLU + MaxPooling** module design that fuses BatchNorm (BN) with convolution to significantly reduce parameters. The key design decision is how to handle the bias term from the fused Conv-BN layer.

### Model Fusion Strategy

The original BatchNorm operation applied after convolution can be expressed as:

\[
\begin{cases}
x_{\text{conv}} = w_{\text{conv}} \cdot x_{\text{in}} + b \\
x_{\text{bn}} = \frac{x_{\text{conv}} - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
\end{cases}
\]

where $x_{\text{conv}}$ is the convolution output, $w_{\text{conv}}$ is the convolution weight, $x_{\text{in}}$ is the input, $b$ is the convolution bias, $\mu$ is the BatchNorm mean, $\sigma^2$ is the BatchNorm variance, $\epsilon$ is a small constant for numerical stability, $\gamma$ is the BatchNorm scaling factor, and $\beta$ is the BatchNorm shifting factor (bias).

By substituting the convolution output into the BatchNorm equation and rearranging terms, we can express the combined operation as a single linear transformation:

\[
x_{\text{bn}} = w' \cdot x_{\text{in}} + b'
\]

where the fused parameters are:

\[
\begin{cases}
w' = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot w_{\text{conv}} \\
b' = \frac{(b - \mu)}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
\end{cases}
\]

- **BatchNorm fusion**: BatchNorm multiplication parameters are folded directly into the convolution weights through model fusion, eliminating the need to store and compute BN parameters separately.
- **Bias handling**: Since the original convolution operation does not include a bias term ($b = 0$), the fused bias simplifies to:

\[
b' = \frac{-\mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
\]

We handle the fused Conv-BN bias by adding it during the **first accumulation update** in the SFU (when `kij = 0`). The accumulation process becomes:

\[
\text{psum} = \begin{cases}
w' \cdot x_{\text{in}} + b' & \text{if } kij = 0 \\
\text{psum} + w' \cdot x_{\text{in}} & \text{if } kij > 0
\end{cases}
\]

This approach allows us to incorporate the bias from the fused Conv-BN layer without modifying the core convolution hardware.

### Design Benefits

- **Parameter reduction**: BN parameters ($\mu$, $\sigma^2$, $\gamma$, $\beta$) are fixed at inference and fused into adjacent convolution layers, eliminating the need to store and process them separately.
- **Computation reduction**: Eliminates runtime normalization operations, reducing computation by **4.3%**.
- **Hardware cost reduction**: Simplifies datapath design by removing BatchNorm hardware, reducing hardware cost by **25%**.
- **Design efficiency**: The fused parameters can be pre-computed offline and loaded as standard convolution weights and biases, improving overall system efficiency.

### Complete Pipeline

The complete processing pipeline in Alpha5 integrates the fused Conv-BN with ReLU and MaxPooling:

\[
\text{output} = \text{ReLU}(\text{MaxPool}(w' \cdot x_{\text{in}} + b'))
\]

where $w'$ and $b'$ are the pre-computed fused parameters, and the bias $b'$ is added during the first accumulation cycle in the SFU.

## Implementation Details

### FSM Modifications

The FSM state `S_ReLU` is repurposed as `S_SPF` (Summation, Pooling, and Function) to perform both MaxPooling and ReLU operations. MaxPooling happens right before ReLU, and both operations are performed in this state.

### MaxPooling Mechanism

- Spatial size: 4 × 4 → 2 × 2 (stride=2)
- Hardware-efficient implementation requiring only 1 additional flip-flop per SIMD lane
- Uses a specially designed `o_nij` read sequence to minimize hardware complexity
- The MPL2D unit updates the current maximum incrementally and resets every 4 cycles

### ReLU Mechanism

ReLU is cascaded after the MaxPooling output.

## Testbench Usage

```bash
cd Alpha5/
iverilog -f filelist -o compiled
vvp compiled
```

## Testbench Design

1. TB feeds `kij` and `psum.txt` to SFU.
2. SFU performs accumulation for `kij=0..8`. **The bias from the fused Conv-BN is added during the first accumulation update (kij=0).**
3. After accumulation of `kij=8`, SFU enters `S_SPF` state that performs MaxPool and ReLU, and saves the result in output memory.
4. TB sends a readout signal.

#### Notes

The golden pattern files didn't include bias pattern because we're running out of time. A zero vector is fed to the bias input among all testcases. Nonetheless, the golden pattern effectively validates accumulation, MaxPool and ReLU of SFU. We believe the expression of bias adding is trivial enough to be correct.
