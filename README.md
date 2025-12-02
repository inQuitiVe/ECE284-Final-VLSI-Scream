# Done

> Files for both accuracy (4,4 and 2,4) are stored in "./Files/" Folder

1. For 4bit Model 

- It achieves **91.53%** accuracy

- Outputs are 15 bits

2. 2bit Model achieved **90.67%** accuracy

- Outputs are 13 bits

- It achieves **90.67%** accuracy

- Weight shape is $C_{in}, k^2, TS$, where $TS = 8$ is **tile size**, $k=3$ is **kernel size**

3. All the `.txt` files are extracted from the **27th** layer of the Model.

- The input data is the **1st image** of the **test data** in CIFAR10

# Alphas

- Fusion is ongoing...

- Model is already trained

1. For 4-bit BN-Fused Model

- It achieves an accuracy of **92.13%**

2. For 2-bit BN-Fused Model

- It achieves an accuracy of **90.43%**
