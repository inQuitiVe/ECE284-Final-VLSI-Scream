# Alpha 4: Whole Conv. Layer

Implementing Batch Normalization, ReLU, and Max Pooling Layer into our proposed 2-D systolic array. 

## Selling Points
- With these layers, it is able to fully go through the whole VGG16 CNN. 
- **Alpha 7** provides a really good structure for doing BN, ReLU, then MPL. All after the conv2D computation

## Completed Part
- Built a testbench that separates the SFU and output SRAM from core.v
- I calculated the mapping from o_nij to MaxPooling2D's output(mpl_onij)
- The maxpool is implemented by 4 registers (each channel), and outputs every 4 inputs.

## In progress Part
- Golden Pattern
- Batchnorm2D, it'll be just an affine transform of all the outputs