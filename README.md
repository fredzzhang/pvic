# ViC

## Setup

Use the package management tool of your choice and run the following commands after creating your environment. 

```bash
# Say you are using Conda
conda create --name vic python=3.8
conda activate vic
# Required dependencies
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib==3.6.3 scipy==1.10.0 tqdm==4.64.1
pip install numpy==1.24.1 timm==0.6.12
# Optional for logging
pip install wandb==0.13.9
# Clone the repo and submodules
git clone ...
cd vic
git submodule init
git submodule update
pip install -e pocket
# Build CUDA operator for MultiScaleDeformableAttention
cd h_detr/model/ops
python setup.py build install
```

## Wandb

Several arguments need to be specified as environment variables.

- WANDB_ENTITY
- WANDB_API_KEY
- WANDB_PROJECT
- WANDB_NAME