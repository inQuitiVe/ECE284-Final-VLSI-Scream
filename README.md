# Done

> Files for **4 bits** and **2 bits** accuracies are stored in "./Files/" Folder

## 4 bits Model 

- It achieves **91.53%** accuracy

## 2 bit Model

- It achieves **90.67%** accuracy

## Shapes of txt Files

1. Output Stationary:

| Parameter | Output Stationary |
|---|---|
| Weight Number | $(2, k^2, \dfrac{C_{in}}{t})$ |
| Weight Shape | $(t, \dfrac{C_{out}}{2} )$ |
| Input Number | $(2, \dfrac{C_{in}}{t})$ |
| Input Shapes | $(t,4,w+2p)$ |

2. Weight Stationary:

| Parameter | Weight Stationary |
|---|---|
| Input Number | $\dfrac{C_{in}}{t}$ |
| Input Shape | $(t, h+2p, w+2p)$ |
| Weight Number | $(k^2, \dfrac{C_{in}}{t}, \dfrac{C_{out}}{t})$ |
| Weight Shape | $(t, t)$ |

- $h=w=4, t = \text{tile size} = 8$

- Outputs are $(t, \dfrac{C_{out}}{t} hw  )$

## Others

- All the `.txt` files are extracted from the **27th** layer of the Model.

- The input data is the **1st image** of the **test data** in CIFAR10

- Outputs are 16 bits

- `path/` files can be downloaded from `https://drive.google.com/drive/folders/1Msoyvbh17tpp8IkoSukHmSEJGndVMcX8?usp=drive_link`

# Alphas

1. For 4-bit BN-Fused Model

- It achieves an accuracy of **92.13%**

2. For 2-bit BN-Fused Model (**TODO**)

- It achieves an accuracy of ...
