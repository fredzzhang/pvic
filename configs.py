"""
Configurations for object detectors

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

import argparse
import numpy as np

def base_detector_args():
    """Arguments for building the base detector DETR"""
    parser = argparse.ArgumentParser(add_help=False)
    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))

    # Transformer
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    # Training
    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--lr-drop', default=20, type=int)
    parser.add_argument('--lr-drop-factor', default=.2, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--clip-max-norm', default=.1, type=float)

    # Loss
    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # Misc.
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--data-root', default='./hicodet')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--print-interval', default=100, type=int)
    return parser

def advanced_detector_args():
    """Arguments for building advanced variants of DETR"""
    parser = argparse.ArgumentParser(add_help=False)
    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--position-embedding-scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num-feature-levels', default=4, type=int, help='number of feature levels')
    parser.add_argument("--drop-path-rate", default=0.2, type=float)
    parser.add_argument("--pretrained_backbone_path", default=None, type=str)

    # Transformer
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=.0, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument("--num-queries-one2one", default=300, type=int,
                        help="Number of query slots for one-to-one matching",)

    # Hybrid matching settings
    parser.add_argument('--num-queries-one2many', default=0, type=int,
                        help="Number of query slots for one-to-many matchining",)

    # Segmentation
    parser.add_argument('--masks', action="store_true")

    # Deformable transformer
    parser.add_argument('--dec-n-points', default=4, type=int)
    parser.add_argument('--enc-n-points', default=4, type=int)
    parser.add_argument('--no-box-refine', dest="with_box_refine",
                        default=True, action='store_false')
    parser.add_argument('--no-two-stage', dest="two_stage",
                        default=True, action='store_false')

    # Tricks
    parser.add_argument("--no-mixed-selection", dest="mixed_selection",
                        action="store_false", default=True)
    parser.add_argument("--no-look-forward-twice", dest="look_forward_twice",
                        action="store_false", default=True)

    # Training
    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--lr-backbone', default=0., type=float)
    parser.add_argument('--lr-drop', default=20, type=int)
    parser.add_argument('--lr-drop-factor', default=.2, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--clip-max-norm', default=.1, type=float)
    parser.add_argument("--use-checkpoint", default=False, action="store_true")

    # Evaluation
    parser.add_argument("--topk", default=100, type=int)

    # Loss
    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=2, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument("--mask-loss-coef", default=1, type=float)
    parser.add_argument("--dice-loss-coef", default=1, type=float)
    parser.add_argument("--cls-loss-coef", default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument("--focal-alpha", default=0.25, type=float)

    # Misc.
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--data-root', default='./hicodet')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--print-interval', default=100, type=int)
    return parser