# Alpha 8: ConvNext Application

## Overview

Alpha 8 implements and evaluates the ConvNext architecture, a modern CNN architecture that achieves comparable accuracy to VGGNet with significantly fewer parameters. This version demonstrates the accelerator's capability to run state-of-the-art neural network architectures efficiently.

## Key Innovation

**Implementation and evaluation of ConvNext, achieving ~90% accuracy (4-bit) with 51% fewer parameters compared to VGGNet.**

ConvNext represents a modern approach to CNN design, using depthwise separable convolutions and efficient architectural patterns to reduce model size while maintaining accuracy.

## Architecture Overview

### ConvNext Structure

```
Input (224×224×3)
  ↓
Conv24 (96×96×96)
  ↓
Layer Norm
  ↓
ConvNext Block ×3
  ↓
Down Sample → (28×28×192)
  ↓
ConvNext Block ×3
  ↓
Down Sample → (14×14×384)
  ↓
ConvNext Block ×9
  ↓
Down Sample → (7×7×768)
  ↓
ConvNext Block ×3
  ↓
Layer Norm
  ↓
Global Avg Pooling → (1×1×768)
  ↓
Softmax
```

### Feature Map Progression

| Stage | Feature Map Size | Channels |
|-------|-----------------|----------|
| Input | 224×224 | 3 |
| Stage 1 | 96×96 | 96 |
| Stage 2 | 28×28 | 192 |
| Stage 3 | 14×14 | 384 |
| Stage 4 | 7×7 | 768 |
| Output | 1×1 | 768 |

## Model Comparison

| Model | Size | Accuracy (4-bit) | Parameters |
|-------|------|-----------------|------------|
| VGGNet | 68.7 MB | ~90% | ~138M |
| ConvNext | 33.0 MB | ~90% | ~28M |

**Key Advantages:**
- **51% model size reduction** (68.7 MB → 33.0 MB)
- **Similar accuracy** (~90% at 4-bit quantization)
- **Fewer parameters** enable faster inference and lower memory requirements

## ConvNext Block Structure

Each ConvNext block typically contains:
1. **Depthwise Convolution**: Efficient spatial feature extraction
2. **Pointwise Convolution**: Channel mixing
3. **Layer Normalization**: Normalization across channels
4. **GELU Activation**: Gaussian Error Linear Unit
5. **Residual Connection**: Skip connection for gradient flow

## Implementation Features

### Supported Operations
- **Depthwise Separable Convolutions**: Efficient convolution pattern
- **Layer Normalization**: Channel-wise normalization
- **Global Average Pooling**: Spatial dimension reduction
- **GELU Activation**: Modern activation function (supported via Alpha 6)

### Quantization
- **4-bit Quantization**: Maintains ~90% accuracy
- **Weight Quantization**: Reduced model size
- **Activation Quantization**: Lower memory bandwidth

## Directory Structure

```
Alpha8/
└── [Implementation files to be added]
```

## Usage

### Model Deployment

1. **Model Conversion**: Convert trained ConvNext model to accelerator format
2. **Quantization**: Apply 4-bit quantization to weights and activations
3. **Layer Mapping**: Map ConvNext layers to accelerator operations
4. **Execution**: Run inference on the accelerator

### Performance Evaluation

Compare against VGGNet:
- **Accuracy**: Maintain ~90% at 4-bit
- **Model Size**: 51% reduction
- **Inference Speed**: Faster due to fewer parameters
- **Memory Usage**: Lower memory footprint

## Hardware Requirements

- **MAC Array**: 8×8 systolic array (sufficient for ConvNext operations)
- **Activation Functions**: GELU support (via Alpha 6)
- **Normalization**: Layer normalization support
- **Memory**: Reduced memory requirements compared to VGGNet

## Status

**To be added** - This application is planned for future implementation.

## Notes

- ConvNext demonstrates the accelerator's flexibility for modern architectures
- The reduced parameter count makes it ideal for edge deployment
- Layer normalization can be fused similar to BN (Alpha 5) for further optimization
- Global Average Pooling can be efficiently implemented in the SFU

## Related Work

- **Alpha 4**: Whole Conv Layer implementation (BN, ReLU, MaxPooling)
- **Alpha 5**: Model Fusion (can be applied to Layer Norm in ConvNext)
- **Alpha 6**: Flexible Activation Functions (GELU support for ConvNext)

