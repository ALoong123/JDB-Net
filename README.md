# JBD-Net: Joint Bidirectional inter-layer interaction and Dynamic  weighting Network for polyp image segmentation

> **Authors:** 
>
> Guangli Li , Wenhao Ai , Jingqin Lv , Boyang Liu , Yuxing Zou , Donghong Ji ,  Hongbin Zhang

## 1. Preface

- This repository provides code for "***JBD-Net: Joint Bidirectional inter-layer interaction and Dynamic  weighting Network for polyp image segmentation***".

## 2. Overview

### 2.1. Introduction

Accurate polyp segmentation plays a vital role in the early diagnosis of colorectal cancer (CRC). However, current researches need to handle many urgent issues, including weak inter-layers information interaction, loose correlations among diverse features, and the underutilization of polyp boundary information. To this end, we propose a Joint Bidirectional inter-layer interaction and Dynamic weighting Network (JBD-Net) for polyp image segmentation. First, a novel Bidirectional Inter-layer Information Interaction (BIII) module is constructed to achieve multi-scale effective information interaction between the adjacent layers in the encoder. Second, a Dynamic Weighting Fusion (DWF) module is built to enable the optimal fusion of diverse features adaptively. Finally, a Boundary Information Regression (BIR) module incorporating a deep supervision mechanism is created to realize the collaborative optimization of boundary awareness and feature refinement. Extensive experiments conducted on five benchmark datasets show that JBD-Net achieves superior segmentation performance, offering reliable support for intelligent CRC diagnosis: obtaining mDice and mIou scores of 0.927 and 0.881 for CVC-ClinicDB, 0.792 and 0.718 for CVC-ColonDB, and 0.914 and 0.849 for CVC-300, respectively. We also validate the generalization ability of JBD-Net on breast cancer and fundus image datasets, further demonstrating the versatility of our model for diversified medical image segmentation tasks. 

### 2.2. Framework Overview

<p align="center">
    <img src="imgs/JBD-Net.png"/> <br />
    <em> 
    Fig. 1. Structure of JBD-Net
    </em>
</p>

### 2.3. Qualitative Results

<p align="center">
    <img src="imgs/Qualitative Results.png"/> <br />
    <em> 
    Fig. 2. Segmentation results comparisons, where the last column is the heat map of our model. We 
use a red bounding box to locate the corresponding polyps in each image
    </em>
</p>





## 3. Proposed Baseline

### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with  a single NVIDIA RTX 3090 with 24 GB Memory.

> Note that our model also supports low memory GPU, which means you can lower the batch size

1. Configuring your environment (Prerequisites):

   Note that JDB-Net is only tested on Ubuntu OS with the following environments.  It may work on other operating systems as well but we do not guarantee that it will.

   - Creating a virtual environment in terminal: `conda create -n JDBNet python=3.10`.
   - Installing necessary packages: `pip install -r requirements.txt`

2. Downloading necessary data:

   - downloading dataset and move it into `./data/`. The above data can be downloaded from this link [(Google Drive)](https://drive.google.com/drive/folders/1swny04Dt-R_4w7Zv0q92usGFAiMBLWbh?usp=drive_link).
   - downloading pretrained weights and move it into `checkpoints/`, which can be found in this [download link (Google Drive)](https://drive.google.com/drive/folders/1YlroiBgJramtw5ZJdv58rXyjuM8aJnPT?usp=drive_link).

3. Training Configuration:

   - Assigning your costumed path, like `--model` and `--batch_size` in `opt.py`.
   - Just enjoy it!

4. Testing Configuration:

   - After you download all the pre-trained model and testing dataset, just run `test.py` to generate the final result:  replace your trained model directory (`--load_ckpt`).
   - Just enjoy it!

## 4. License


The source code is free for research and education use only. Any comercial use should get formal permission first.
