#!/bin/bash

# This is a template for launching training and testing scripts
# 
# Fred Zhang <frederic.zhang@anu.edu.au>
# 
# The Australian National University
# Microsoft Research Asia

# -------------------------------
# Training commands
# -------------------------------

# Train ViC-DETR-R50 on HICO-DET
DETR=base python main.py --pretrained checkpoints/detr-r50-hicodet.pth --output-dir outputs/vic-detr-r50-hicodet
# Train ViC-Defm-DETR-R50 on HICO-DET
DETR=advanced python main.py --pretrained checkpoints/defm-detr-r50-dp0-mqs-lft-iter-2stg-hicodet.pth --output-dir outputs/vic-defm-detr-r50-hicodet
# Train ViC-H-Defm-DETR-R50 on HICO-DET
DETR=advanced python main.py --num-queries-one2many 1500 --pretrained checkpoints/h-defm-detr-r50-dp0-mqs-lft-iter-2stg-hicodet.pth --output-dir outputs/vic-h-defm-detr-r50-hicodet
# Train ViC-H-Defm-DETR-SwinL on HICO-DET
DETR=advanced python main.py --backbone swin_large --drop-path-rate 0.5 --num-queries-one2one 900 --num-queries-one2many 1500 --pretrained checkpoints/h-defm-detr-swinL-dp0-mqs-lft-iter-2stg-hicodet.pth --output-dir outputs/vic-h-defm-detr-swinL-hicodet

# -------------------------------
# Testing commands
# -------------------------------

# Test ViC-DETR-R50 on HICO-DET
DETR=base python main.py --world-size 1 --batch-size 1 --eval --resume /path/to/model
# Test ViC-Defm-DETR-R50 on HICO-DET
DETR=advanced python main.py --world-size 1 --batch-size 1 --eval --resume /path/to/model
# Test ViC-H-Defm-DETR-R50 on HICO-DET
DETR=advanced python main.py --num-queries-one2many 1500 --world-size 1 --batch-size 1 --eval --resume /path/to/model
# Test ViC-H-Defm-DETR-SwinL on HICO-DET
DETR=advanced python main.py --backbone swin_large --drop-path-rate 0.5 --num-queries-one2one 900 --num-queries-one2many 1500 --world-size 1 --batch-size 1 --eval --resume /path/to/model