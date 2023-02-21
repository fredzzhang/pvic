"""
Opearations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

import math
import torch
import torchvision.ops.boxes as box_ops

from torch import Tensor
from typing import List, Tuple

def compute_sinusoidal_pe(pos_tensor: Tensor) -> Tensor:
    """
    Compute positional embeddings for points or bounding boxes

    Parameters:
    -----------
    pos_tensor: Tensor
        Coordinates of 2d points (x, y) normalised to (0, 1). The shape is (n_q, bs, 2).

    Returns:
    --------
    pos: Tensor
        Sinusoidal positional embeddings of shape (n_q, bs, 256).
    """
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos

def prepare_region_proposals(
    results, hidden_states, image_sizes,
    box_score_thresh, human_idx,
    min_instances, max_instances
):
    region_props = []
    for res, hs, sz in zip(results, hidden_states, image_sizes):
        sc, lb, bx = res.values()

        keep = box_ops.batched_nms(bx, sc, lb, 0.5)
        sc = sc[keep].view(-1)
        lb = lb[keep].view(-1)
        bx = bx[keep].view(-1, 4)
        hs = hs[keep].view(-1, 256)

        # Clamp boxes to image
        bx[:, :2].clamp_(min=0)
        bx[:, 2].clamp_(max=sz[1])
        bx[:, 3].clamp_(max=sz[0])

        keep = torch.nonzero(sc >= box_score_thresh).squeeze(1)

        is_human = lb == human_idx
        hum = torch.nonzero(is_human).squeeze(1)
        obj = torch.nonzero(is_human == 0).squeeze(1)
        n_human = is_human[keep].sum(); n_object = len(keep) - n_human
        # Keep the number of human and object instances in a specified interval
        if n_human < min_instances:
            keep_h = sc[hum].argsort(descending=True)[:min_instances]
            keep_h = hum[keep_h]
        elif n_human > max_instances:
            keep_h = sc[hum].argsort(descending=True)[:max_instances]
            keep_h = hum[keep_h]
        else:
            keep_h = torch.nonzero(is_human[keep]).squeeze(1)
            keep_h = keep[keep_h]

        if n_object < min_instances:
            keep_o = sc[obj].argsort(descending=True)[:min_instances]
            keep_o = obj[keep_o]
        elif n_object > max_instances:
            keep_o = sc[obj].argsort(descending=True)[:max_instances]
            keep_o = obj[keep_o]
        else:
            keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
            keep_o = keep[keep_o]

        keep = torch.cat([keep_h, keep_o])

        region_props.append(dict(
            boxes=bx[keep],
            scores=sc[keep],
            labels=lb[keep],
            hidden_states=hs[keep]
        ))

    return region_props

def associate_with_ground_truth(boxes, paired_inds, targets, num_classes, thresh=0.5):
    labels = []
    for bx, p_inds, target in zip(boxes, paired_inds, targets):
        is_match = torch.zeros(len(p_inds), num_classes, device=bx.device)

        bx_h, bx_o = bx[p_inds].unbind(1)
        gt_bx_h = recover_boxes(target["boxes_h"], target["size"])
        gt_bx_o = recover_boxes(target["boxes_o"], target["size"])

        x, y = torch.nonzero(torch.min(
            box_ops.box_iou(bx_h, gt_bx_h),
            box_ops.box_iou(bx_o, gt_bx_o)
        ) >= thresh).unbind(1)
        is_match[x, target["labels"][y]] = 1

        labels.append(is_match)
    return torch.cat(labels)

def recover_boxes(boxes, size):
    boxes = box_cxcywh_to_xyxy(boxes)
    h, w = size
    scale_fct = torch.stack([w, h, w, h])
    boxes = boxes * scale_fct
    return boxes

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def pad_queries(queries):
    b = len(queries)
    k = queries[0].shape[1]
    ns = [len(q) for q in queries]
    device = queries[0].device
    dtype = queries[0].dtype

    padded_queries = torch.zeros(b, max(ns), k, device=device, dtype=dtype)
    q_padding_mask = torch.zeros(b, max(ns), device=device, dtype=torch.bool)
    for i, n in enumerate(ns):
        padded_queries[i, :n] = queries[i]
        q_padding_mask[i, n:] = True
    return padded_queries, q_padding_mask

def compute_prior_scores(
    x: Tensor, y: Tensor,
    scores: Tensor, labels: Tensor,
    num_classes: int, is_training: bool,
    obj_cls_to_tgt_cls: list
) -> Tensor:
    prior_h = torch.zeros(len(x), num_classes, device=scores.device)
    prior_o = torch.zeros_like(prior_h)

    # Raise the power of object detection scores during inference
    p = 1.0 if is_training else 2.8
    s_h = scores[x].pow(p)
    s_o = scores[y].pow(p)

    # Map object class index to target class index
    # Object class index to target class index is a one-to-many mapping
    target_cls_idx = [obj_cls_to_tgt_cls[obj.item()]
        for obj in labels[y]]
    # Duplicate box pair indices for each target class
    pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
    # Flatten mapped target indices
    flat_target_idx = [t for tar in target_cls_idx for t in tar]

    prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
    prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

    return torch.stack([prior_h, prior_o], dim=1)

def compute_spatial_encodings(
    boxes_1: List[Tensor], boxes_2: List[Tensor],
    shapes: List[Tuple[int, int]], eps: float = 1e-10
) -> Tensor:
    """
    Parameters:
    -----------
    boxes_1: List[Tensor]
        First set of bounding boxes (M, 4)
    boxes_1: List[Tensor]
        Second set of bounding boxes (M, 4)
    shapes: List[Tuple[int, int]]
        Image shapes, heights followed by widths
    eps: float
        A small constant used for numerical stability

    Returns:
    --------
    Tensor
        Computed spatial encodings between the boxes (N, 36)
    """
    features = []
    for b1, b2, shape in zip(boxes_1, boxes_2, shapes):
        h, w = shape

        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

        iou = torch.diag(box_ops.box_iou(b1, b2))

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre
            c1_x / w, c1_y / h, c2_x / w, c2_y / h,
            # Relative box width and height
            b1_w / w, b1_h / h, b2_w / w, b2_h / h,
            # Relative box area
            b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
            b2_w * b2_h / (b1_w * b1_h + eps),
            # Box aspect ratio
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            # Intersection over union
            iou,
            # Relative distance and direction of the object w.r.t. the person
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)

        features.append(
            torch.cat([f, torch.log(f + eps)], 1)
        )
    return torch.cat(features)

def binary_focal_loss_with_logits(
    x: Tensor, y: Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
    reduction: str = 'mean',
    eps: float = 1e-6
) -> Tensor:
    """
    Focal loss by Lin et al.
    https://arxiv.org/pdf/1708.02002.pdf

    L = - |1-y-alpha| * |y-x|^{gamma} * log(|1-y-x|)

    Parameters:
    -----------
    x: Tensor[N, K]
        Post-normalisation scores
    y: Tensor[N, K]
        Binary labels
    alpha: float
        Hyper-parameter that balances between postive and negative examples
    gamma: float
        Hyper-paramter suppresses well-classified examples
    reduction: str
        Reduction methods
    eps: float
        A small constant to avoid NaN values from 'PowBackward'

    Returns:
    --------
    loss: Tensor
        Computed loss tensor
    """
    loss = (1 - y - alpha).abs() * ((y-torch.sigmoid(x)).abs() + eps) ** gamma * \
        torch.nn.functional.binary_cross_entropy_with_logits(
            x, y, reduction='none'
        )
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("Unsupported reduction method {}".format(reduction))
