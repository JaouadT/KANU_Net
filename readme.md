# KANU-Net: Kolmogorov-Arnold Networks based U-Net architecture for images segmentation

## Overview

This repository consists the code for the implementation of U-Net architecture but with Kolmogorov-Arnold Convolutions instead of regular convolutions.

### What is a KAN?
KANs are promising alternatives of Multi-Layer Perceptrons (MLPs). KANs have strong mathematical foundations just like MLPs: MLPs are based on the universal approximation theorem, while KANs are based on Kolmogorov-Arnold representation theorem. KANs and MLPs are dual: KANs have activation functions on edges, while MLPs have activation functions on nodes. KAN seems to be more parameter efficient than MLPs, but each KAN Layer has more parameters than a MLP layer. 

<img width="1163" alt="mlp_kan_compare" src="https://github.com/KindXiaoming/pykan/assets/23551623/695adc2d-0d0b-4e4b-bcff-db2c8070f841">

For more information about this novel architecture please visit:
- The official Pytorch implementation of the architecture: https://github.com/KindXiaoming/pykan
- The research paper: https://arxiv.org/abs/2404.19756


### Installation

```
git clone https://github.com/JaouadT/KANU_Net.git
cd KANU_Net
pip install -r requirements.txt
```

### Training

```
python train.py --model KANU_Net --dataset BUSI --gpu 0
```

## Results

The following are the results after training KANU_Net and regular U-Net with the same experimental setup:
<!-- results table start -->
| Model    | Accuracy | Dice  | IoU   | Sensitivity | Precision | Specificity |
| -------- | -------- | ----- | ----- | ----------- | --------- | ----------- |
| KANU_Net | 97.85    | 73.83 | 60.08 | 70.19       | 97.88     | 97.85       |
| U-Net    | 95.23    | 74.95 | 62.17 | 69.44       | 84.33     | 98.64       |
<!-- results table end -->

## Acknowledgements and credits

This model is built upon [U-Net](https://github.com/milesial/Pytorch-UNet) and the authors of the implementation of the [KA-Conv](https://github.com/XiangboGaoBarry/KA-Conv). We extend our gratitude to the creators of the original [KAN](https://github.com/KindXiaoming/pykan) for their pioneering work in the field.
