"""
Configurations for object detectors

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

import argparse

def base_detector_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--lr-drop', default=20, type=int)
    parser.add_argument('--lr-drop-factor', default=.2, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--clip-max-norm', default=.1, type=float)

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

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