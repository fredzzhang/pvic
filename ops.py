"""
Opearations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.ops.boxes as box_ops

from torch import nn, Tensor
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment

from detr.util.box_ops import generalized_box_iou

class BalancedBoxSampler:
    def __init__(self, threshold: float = .2, perc: float = .8) -> None:
        self.threshold = threshold
        self.perc = perc

    def __call__(self, scores: Tensor, number: int) -> Tensor:
        """
        Parameters:
        -----------
        scores: Tensor
            (N,) The confidence scores for a set of bounding boxes
        number: int
            The number of boxes to sample

        Returns:
        --------
        sampled_high: Tensor
            Indices of sampled high-confidence examples
        sampled_low: Tensor
            Indices of sampled low-confidence examples
        """
        idx_high = torch.nonzero(scores >= self.threshold).squeeze(1)
        idx_low = torch.nonzero(scores < self.threshold).squeeze(1)

        n_high = int(number * self.perc)
        # Protect against not enough high-confidence examples
        n_high = min(idx_high.numel(), n_high)
        n_low = number - n_high
        # Protect against not enough low-confidence examples
        n_low = min(idx_low.numel(), n_low)

        perm_high = torch.randperm(idx_high.numel(), device=idx_high.device)[:n_high]
        perm_low = torch.randperm(idx_low.numel(), device=idx_low.device)[:n_low]

        sampled_high = idx_high[perm_high]
        sampled_low = idx_low[perm_low]

        return sampled_high, sampled_low

class BoxPairCoder:
    def __init__(self,
        weights: Optional[List[float]] = None,
        bbox_xform_clip: float = math.log(1000. / 16)
    ) -> None:
        if weights is None:
            weights = [10., 10., 5., 5.]
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, props_h: Tensor, props_o: Tensor, target_h: Tensor, target_o: Tensor) -> Tensor:
        """
        Compute the regression targets based on proposed boxes pair and target box pairs.
        NOTE that all boxes are presumed to have been normalised by image width and height
        and are in (c_x, c_y, w, h) format.

        Parameters:
        -----------
        props_h: Tensor
            (N, 4) Human box proposals
        props_o: Tensor
            (N, 4) Object box proposals
        target_h: Tensor
            (N, 4) Human box targets
        target_o: Tensor
            (N, 4) Object box targets

        Returns:
        --------
        box_deltas: Tensor
            (N, 8) Regression targets for proposed box pairs
        """
        wx, wy, ww, wh = self.weights
        dx_h = wx * (target_h[:, 0] - props_h[:, 0])
        dy_h = wy * (target_h[:, 1] - props_h[:, 1])
        dw_h = ww * torch.log(target_h[:, 2] / props_h[:, 2])
        dh_h = wh * torch.log(target_h[:, 3] / props_h[:, 3])

        dx_o = wx * (target_o[:, 0] - props_o[:, 0])
        dy_o = wy * (target_o[:, 1] - props_o[:, 1])
        dw_o = ww * torch.log(target_o[:, 2] / props_o[:, 2])
        dh_o = wh * torch.log(target_o[:, 3] / props_o[:, 3])

        box_deltas = torch.stack([dx_h, dy_h, dw_h, dh_h, dx_o, dy_o, dw_o, dh_o], dim=1)

        return box_deltas

    def decode(self, props_h: Tensor, props_o: Tensor, box_deltas: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Recover the regressed box pairs based on the proposed pairs and the box deltas.
        NOTE that the proposed box pairs are presumed to have been normalised by image
        width and height and are in (c_x, c_y, w, h) format.

        Parameters:
        -----------
        props_h: Tensor
            (N, 4) Human box proposals
        props_o: Tensor
            (N, 4) Object box proposals
        box_deltas: Tensor
            (N, 8) Predicted regression values for proposed box pairs

        Returns:
        --------
        regressed_h: Tensor
            (N, 4) Regressed human boxes
        regressed_o: Tensor
            (N, 4) Regressed object boxes
        """
        weights = torch.as_tensor(self.weights).repeat(2).to(box_deltas)
        box_deltas = box_deltas / weights

        dx_h, dy_h, dw_h, dh_h, dx_o, dy_o, dw_o, dh_o = box_deltas.unbind(1)

        # # Prevent sending too large values into torch.exp()
        dw_h = torch.clamp(dw_h, max=self.bbox_xform_clip)
        dh_h = torch.clamp(dh_h, max=self.bbox_xform_clip)
        dw_o = torch.clamp(dw_o, max=self.bbox_xform_clip)
        dh_o = torch.clamp(dh_o, max=self.bbox_xform_clip)

        regressed_h = torch.stack([
            props_h[:, 0] + dx_h, props_h[:, 1] + dy_h,
            props_h[:, 2] * torch.exp(dw_h), props_h[:, 3] * torch.exp(dh_h)
        ], dim=1)

        regressed_o = torch.stack([
            props_o[:, 0] + dx_o, props_o[:, 1] + dy_o,
            props_o[:, 2] * torch.exp(dw_o), props_o[:, 3] * torch.exp(dh_o)
        ], dim=1)

        return regressed_h, regressed_o

class HungarianMatcher(nn.Module):

    def __init__(self,
        cost_object: float = 1., cost_verb: float = 1.,
        cost_bbox: float = 1., cost_giou: float = 1.
    ) -> None:
        """
        Parameters:
        ----------
        cost_object: float
            Weight on the object classification term
        cost_verb: float
            Weight on the verb classification term
        cost_bbox:
            Weight on the L1 regression error
        cost_giou:
            Weight on the GIoU term
        """
        super().__init__()
        self.cost_object = cost_object
        self.cost_verb = cost_verb
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_object + cost_verb + cost_bbox + cost_giou, \
            "At least one cost coefficient should be non zero."

    @torch.no_grad()
    def forward(self,
        bx_h: List[Tensor], bx_o: List[Tensor], objects: List[Tensor],
        prior: List[Tensor], logits: Tensor, targets: List[dict]
    ) -> List[Tensor]:
        """
        Parameters:
        ----------
        bh: List[Tensor]
            (M, 4) Human bounding boxes in detected pairs
        bo: List[Tensor]
            (M, 4) Object bounding boxes in detected pairs
        objects: List[Tensor]
            (M,) Object class indices in each pair 
        prior: List[Tensor]
            (2, M, K) Object detection scores for the human and object boxes in each pair
        logits: Tensor
            (M_, K) Classification logits for all boxes pairs
        targets: List[dict]
            Targets for each image with the following keys, `boxes_h` (G, 4), `boxes_o` (G, 4),
            `labels` (G, 117), `objects` (G,)

        Returns:
        --------
        List[Tensor]
            A list of tuples for matched indices between detected pairs and ground truth pairs.

        """
        eps = 1e-6

        # The number of box pairs in each image
        n = [len(p) for p in bx_h]

        gt_bx_h = [t['boxes_h'] for t in targets]
        gt_bx_o = [t['boxes_o'] for t in targets]

        scores = [
            torch.sigmoid(lg) * p.prod(0)
            for lg, p in zip(logits.split(n), prior)
        ]
        gt_labels = [t['labels'] for t in targets]

        cost_verb = [
            -0.5 * (
                s.matmul(l.T) / (l.sum(dim=1).unsqueeze(0) + eps) +
                (1-s).matmul(1 - l.T) / (torch.sum(1 - l, dim=1).unsqueeze(0) + eps)
            ) for s, l in zip(scores, gt_labels)
        ]

        cost_bbox = [torch.max(
            torch.cdist(h, gt_h, p=1), torch.cdist(o, gt_o, p=1)
        ) for h, o, gt_h, gt_o in zip(bx_h, bx_o, gt_bx_h, gt_bx_o)]

        cost_giou = [torch.max(
            -generalized_box_iou(box_cxcywh_to_xyxy(h), box_cxcywh_to_xyxy(gt_h)),
            -generalized_box_iou(box_cxcywh_to_xyxy(o), box_cxcywh_to_xyxy(gt_o))
        ) for h, o, gt_h, gt_o in zip(bx_h, bx_o, gt_bx_h, gt_bx_o)]

        cost_object = [
            -torch.log(                                 # Log barrier
                obj.unsqueeze(1).eq(t['object'])        # Binary mask
                * p[0].max(-1)[0].unsqueeze(1) + eps    # Object classification score
            ) for obj, p, t in zip(objects, prior, targets)
        ]

        # Final cost matrix
        C = [
            c_v * self.cost_verb + c_b * self.cost_bbox +
            c_g * self.cost_giou + c_o * self.cost_object
            for c_v, c_b, c_g, c_o in zip(cost_verb, cost_bbox, cost_giou, cost_object)
        ]

        indices = [linear_sum_assignment(c.cpu()) for c in C]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.matcher = HungarianMatcher(
            cost_object=args.set_cost_object,
            cost_verb=args.set_cost_verb,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou
        )
        self.box_pair_coder = BoxPairCoder()

    def focal_loss(self,
        bx_h: List[Tensor], bx_o: List[Tensor], indices: List[Tensor],
        prior: List[Tensor], logits: Tensor, targets: List[dict]
    ) -> Tensor:
        collated_labels = []
        for bh, bo, idx, tgt in zip(bx_h, bx_o, indices, targets):
            idx_h, idx_o = idx

            mask = torch.diag(torch.min(
                box_ops.box_iou(
                    box_cxcywh_to_xyxy(bh[idx_h]),
                    box_cxcywh_to_xyxy(tgt['boxes_h'][idx_o])
                ), box_ops.box_iou(
                    box_cxcywh_to_xyxy(bo[idx_h]),
                    box_cxcywh_to_xyxy(tgt['boxes_o'][idx_o])
                )
            ) > 0.5).unsqueeze(1)
            matched_labels = tgt['labels'][idx_o] * mask
            labels = torch.zeros(
                len(bh), self.args.num_classes,
                device=matched_labels.device
            )
            labels[idx_h] = matched_labels
            collated_labels.append(labels)

        collated_labels = torch.cat(collated_labels)
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        logits = logits[x, y]; prior = prior[x, y]; labels = collated_labels[x, y]

        n_p = len(torch.nonzero(labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_focal_loss_with_logits(
            torch.log(
                (prior + 1e-8) / (1 + torch.exp(-logits) - prior)
            ), labels, reduction='sum', alpha=self.args.alpha, gamma=self.args.gamma
        )

        return loss / n_p

    def regression_loss(self,
        props_h: List[Tensor], props_o: List[Tensor],
        reg_h: List[Tensor], reg_o: List[Tensor], indices: List[Tensor],
        targets: List[dict], bbox_deltas: List[Tensor],
    ) -> Tensor:
        props_h = torch.cat([b[i].view(-1, 4) for (i, _), b in zip(indices, props_h)])
        props_o = torch.cat([b[i].view(-1, 4) for (i, _), b in zip(indices, props_o)])
        reg_h = torch.cat([b[i].view(-1, 4) for (i, _), b in zip(indices, reg_h)])
        reg_o = torch.cat([b[i].view(-1, 4) for (i, _), b in zip(indices, reg_o)])

        tgt_h = torch.cat([t['boxes_h'][j].view(-1, 4) for (_, j), t in zip(indices, targets)])
        tgt_o = torch.cat([t['boxes_o'][j].view(-1, 4) for (_, j), t in zip(indices, targets)])

        bbox_deltas = torch.cat([d[i].view(-1, 8) for (i, _), d in zip(indices, bbox_deltas)])
        reg_targets = self.box_pair_coder.encode(
            props_h, props_o, tgt_h, tgt_o
        )

        huber_loss = F.smooth_l1_loss(
            bbox_deltas, reg_targets,
            beta=1 / 9, reduction='sum'
        )
        huber_loss = huber_loss / len(bbox_deltas)

        giou_loss = 2 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(reg_h),
            box_cxcywh_to_xyxy(tgt_h)
        )) - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(reg_o),
            box_cxcywh_to_xyxy(tgt_o)
        ))

        giou_loss = giou_loss.sum() / len(bbox_deltas)

        return dict(huber_loss=huber_loss, giou_loss=giou_loss)


    def forward(self,
        boxes: List[Tensor], bh: List[Tensor], bo: List[Tensor], objects: List[Tensor],
        prior: List[Tensor], logits: Tensor, bbox_deltas: Tensor, targets: List[dict]
    ) -> Dict[str, Tensor]:
        # n = [len(b) for b in bh]

        bx_h = [b[h] for b, h in zip(boxes, bh)]
        bx_o = [b[o] for b, o in zip(boxes, bo)]

        # bx_h_post, bx_o_post = self.box_pair_coder.decode(torch.cat(bx_h), torch.cat(bx_o), bbox_deltas)
        # bx_h_post = bx_h_post.split(n); bx_o_post = bx_o_post.split(n)

        indices = self.matcher(bx_h, bx_o, objects, prior, logits, targets)

        loss_dict = {"focal_loss": self.focal_loss(bx_h, bx_o, indices, prior, logits, targets)}
        # loss_dict.update(self.regression_loss(
        #     bx_h, bx_o, bx_h_post, bx_o_post, indices, targets, bbox_deltas.split(n)
        # ))

        return loss_dict

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
