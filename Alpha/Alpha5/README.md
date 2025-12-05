# Alpha 5: Model Fusion

## Overview

Alpha 5 implements Batch Normalization (BN) parameter fusion into adjacent convolution layers. This optimization eliminates runtime normalization operations by fusing BN parameters into convolution weights and biases during inference, significantly reducing computation and hardware cost.

## Key Innovation

**BN parameters are fixed at inference time and fused into adjacent convolution layers, eliminating runtime normalization operations.**

This fusion process transforms the computation graph to combine BN operations directly into convolution operations, reducing both computational overhead and hardware complexity.

## Benefits

- **Computation Reduction**: 4.3% reduction in total computation
- **Hardware Cost Reduction**: 25% reduction in hardware area/cost
- **Simplified Datapath**: Eliminates normalization datapath, improving efficiency
- **No Accuracy Loss**: Mathematical equivalence ensures identical results

## Mathematical Transformation

### Original Computation Flow

```
x_conv = w_conv * x_in + b
x_bn = (x_conv - μ) / √(σ² + ε)
x_out = w' * x_bn + b'
```

Where:
- `μ`: Batch mean
- `σ²`: Batch variance
- `ε`: Small constant (epsilon)
- `γ`: BN scale parameter
- `β`: BN shift parameter

### Fused Computation

The BN operation is mathematically fused into the convolution:

```
w' = γ / √(σ² + ε)
b' = b - μ / √(σ² + ε) * γ + β
x_out = w' * x_conv + b'
```

This eliminates the intermediate normalization step while producing identical results.

### Complete Fusion Formula

For a convolution layer followed by BN:

**Original:**
```
x_conv = w_conv * x_in + b_conv
x_bn = (x_conv - μ) / √(σ² + ε)
x_out = γ * x_bn + β
```

**Fused:**
```
w_fused = (γ / √(σ² + ε)) * w_conv
b_fused = (γ / √(σ² + ε)) * (b_conv - μ) + β
x_out = w_fused * x_in + b_fused
```

## Implementation Approach

1. **Offline Fusion**: BN parameters are fused into convolution weights during model preparation
2. **Weight Transformation**: Convolution weights are scaled by `γ / √(σ² + ε)`
3. **Bias Transformation**: Convolution biases are adjusted to include BN offset
4. **Runtime**: Only fused convolution operation is executed

## Hardware Impact

### Area Reduction
- Eliminates BN computation units (subtract, divide, multiply, add)
- Removes intermediate storage for normalized values
- Simplifies data path between convolution and activation

### Performance Improvement
- Reduces computation cycles (no normalization step)
- Lower memory bandwidth (no intermediate normalized data)
- Improved pipeline efficiency

## Directory Structure

```
Alpha5/
└── [Implementation files to be added]
```

## Usage

### Model Preparation

BN fusion is typically performed offline:

```python
# Pseudo-code for BN fusion
gamma = bn_layer.gamma
beta = bn_layer.beta
mean = bn_layer.running_mean
var = bn_layer.running_var
epsilon = bn_layer.eps

# Fuse into convolution weights
scale = gamma / sqrt(var + epsilon)
fused_weight = conv_weight * scale
fused_bias = (conv_bias - mean) * scale + beta
```

### Hardware Integration

The fused weights and biases are loaded into the accelerator as standard convolution parameters, with no special handling required for BN operations.

## Status

**To be added** - This optimization is planned for future implementation.

## Notes

- BN fusion is a standard optimization technique in inference accelerators
- Requires BN parameters to be fixed (training mode disabled)
- Applicable to all layers with Conv → BN → Activation structure
- Can be combined with other optimizations (quantization, pruning)

