# PViC: Predicate Visual Context

__The repository is currently being cleaned up.__

This repository contains the official PyTorch implementation for the paper
> Frederic Z. Zhang, Yuhui Yuan, Dylan Campbell, Zhuoyao Zhong, Stephen Gould; _Exploring Predicate Visual Context for Detecting Human-Object Interactions_; To appear in the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

## Setup

Use the package management tool of your choice and run the following commands after creating your environment. 

```bash
# Say you are using Conda
conda create --name pvic python=3.8
conda activate pvic
# Required dependencies
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib==3.6.3 scipy==1.10.0 tqdm==4.64.1
pip install numpy==1.24.1 timm==0.6.12
pip install wandb==0.13.9
# Clone the repo and submodules
git clone https://github.com/fredzzhang/pvic.git
cd pvic
git submodule init
git submodule update
pip install -e pocket
# Build CUDA operator for MultiScaleDeformableAttention
cd h_detr/model/ops
python setup.py build install
```
