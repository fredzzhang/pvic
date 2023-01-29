"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

import os
import copy
import torch
import torch.nn.functional as F
import torch.distributed as dist


from torch import nn, Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou

from ops import (
    binary_focal_loss_with_logits,
    compute_spatial_encodings
)
from interaction_head import InteractionHead

import sys
sys.path.append('detr')
from models import build_model
from util import box_ops
from util.misc import nested_tensor_from_tensor_list


class MultiBranchFusion(nn.Module):
    def __init__(self, fst_mod_size, scd_mod_size, repr_size, cardinality):
        super().__init__()
        self.cardinality = cardinality

        sub_repr_size = int(repr_size / cardinality)
        assert sub_repr_size * cardinality == repr_size, \
            "The given representation size should be divisible by cardinality."

        self.fc_1 = nn.ModuleList([
            nn.Linear(fst_mod_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_2 = nn.ModuleList([
            nn.Linear(scd_mod_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_3 = nn.ModuleList([
            nn.Linear(sub_repr_size, repr_size)
            for _ in range(cardinality)
        ])
    def forward(self, fst_mod: Tensor, scd_mod: Tensor) -> Tensor:
        return F.relu(torch.stack([
            fc_3(F.relu(fc_1(fst_mod) * fc_2(scd_mod)))
            for fc_1, fc_2, fc_3
            in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0))

class HumanObjectMatcher(nn.Module):
    def __init__(self, repr_size, num_verbs, obj_to_verb, human_idx=0):
        super().__init__()
        self.repr_size = repr_size
        self.num_verbs = num_verbs
        self.human_idx = human_idx
        self.obj_to_verb = obj_to_verb

        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, repr_size), nn.ReLU(),
        )
        self.mbf = MultiBranchFusion(512, repr_size, repr_size, cardinality=8)

    def check_human_instances(self, labels):
        is_human = labels == self.human_idx
        n_h = torch.sum(is_human)
        if not torch.all(labels[:n_h]==self.human_idx):
            raise AssertionError("Human instances are not permuted to the top!")
        return n_h

    def forward(self, region_props, image_sizes, device=None):
        if device is None:
            device = region_props[0]["hidden_states"].device

        paired_indices = []
        prior_scores = []
        object_types = []
        ho_queries = []
        for i, rp in enumerate(region_props):
            boxes, scores, labels, embeds = rp.values()
            nh = self.check_human_instances(labels)
            n = len(boxes)
            # Enumerate instance pairs
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < nh)).unbind(1)
            # Skip image when there are no valid human-object pairs
            if len(x_keep) == 0:
                paired_indices.append(torch.zeros(0, 2, device=device, dtype=torch.int64))
                prior_scores.append(torch.zeros(0, 2, device=device))
                object_types.append(torch.zeros(0, device=device, dtype=torch.int64))
                ho_queries.append(torch.zeros(0, self.repr_size, device=device))
                continue
            # Compute spatial features
            pairwise_spatial = compute_spatial_encodings(
                boxes[x_keep], boxes[y_keep], [image_sizes[i]]
            )
            pairwise_spatial = self.spatial_head(pairwise_spatial)
            # Compute human-object queries
            ho_q = self.mbf(
                torch.cat([embeds[x_keep], embeds[y_keep]], dim=1),
                pairwise_spatial
            )
            # Append matched human-object pairs
            paired_indices.append(torch.stack([x_keep, y_keep]), dim=1)
            prior_scores.append(torch.stack([scores[x_keep], scores[y_keep]]), dim=1)
            object_types.append(labels[y_keep])
            ho_queries.append(ho_q)

        return ho_queries, paired_indices, prior_scores, object_types

class VerbMatcher(nn.Module):
    def __init__(self, repr_size, obj_to_triplet):
        super().__init__()
        self.repr_size = repr_size
        self.obj_to_triplet = obj_to_triplet

        self.mbf = MultiBranchFusion(repr_size, 1024, repr_size, cardinality=8)
    def forward(self, ho_queries, object_types, triplet_embeds):
        device = ho_queries[0].device

        mm_queries = []
        dup_indices = []
        for ho_q, objs in zip(ho_queries, object_types):
            mm_q_per_img = []
            dup_inds_per_img = []
            for i, o in enumerate(objs):
                trip = self.obj_to_triplet(o.item())
                present_trip = [t in triplet_embeds for t in trip]
                dup = torch.ones(len(present_trip), device=device) * i
                # Construct multi-modal queries
                mm_q = self.mbf(
                    ho_q[i: i+1].repeat(len(dup), 1),
                    torch.stack([triplet_embeds[t] for t in present_trip])
                )
                # Append matched human-verb-object triplets
                mm_q_per_img.append(mm_q)
                dup_inds_per_img.append(dup)
            mm_queries.append(torch.cat(mm_q_per_img))
            dup_indices.append(torch.cat(dup_inds_per_img))

        return mm_queries, dup_indices

class TransformerDecoderLayer(nn.Module):

    def __init__(self, q_size, kv_size, num_heads, ffn_interm_size=2048, dropout=0.1):
        """
        Transformer decoder layer, adapted from DETR codebase by Facebook Research
        https://github.com/facebookresearch/detr/blob/main/models/transformer.py#L187

        Parameters:
        -----------
        q_size: int
            Dimension of the interaction queries.
        kv_size: int
            Dimension of the image features.
        num_heads: int
            Number of heads used in multihead attention.
        ffn_interm_size: int, default: 2048
            The intermediate size in the feedforward network.
        dropout: float, default: 0.1
            Dropout percentage used during training.
        """
        super().__init__()
        self.q_size = q_size
        self.kv_size = kv_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.ffn_interm_size = ffn_interm_size

        self.q_attn = nn.MultiheadAttention(q_size, num_heads, dropout=dropout)
        self.qk_attn = nn.MultiheadAttention(
            q_size, num_heads,
            kdim=kv_size, vdim=kv_size,
            dropout=dropout
        )
        self.ffn = nn.Sequential([
            nn.Linear(q_size, ffn_interm_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_interm_size, q_size)
        ])
        self.ln1 = nn.LayerNorm(q_size)
        self.ln2 = nn.LayerNorm(q_size)
        self.ln3 = nn.LayerNorm(q_size)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.dp3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, queries, features,
            q_mask: Optional[Tensor] = None,
            kv_mask: Optional[Tensor] = None,
            q_padding_mask: Optional[Tensor] = None,
            kv_padding_mask: Optional[Tensor] = None,
            q_pos: Optional[Tensor] = None,
            kv_pos: Optional[Tensor] = None,
            return_q_attn_weights: Optional[bool] = False,
            return_qk_attn_weights: Optional[bool] = False
        ):
        """
        Parameters:
        -----------
        queries: Tensor
            Interaction queries of size (B, N, K).
        features: Tensor
            Image features of size (B, C, H, W).
        q_mask: Tensor, default: None
            Attention mask to be applied during the self attention of queries.
        kv_mask: Tensor, default: None
            Attention mask to be applied during the cross attention from image
            features to interaction queries.
        q_padding_mask: Tensor, default: None
            Padding mask for interaction queries of size (B, N). Values of `True`
            indicate the corresponding query was padded and to be ignored.
        kv_padding_mask: Tensor, default: None
            Padding mask for image features of size (B, HW).
        q_pos: Tensor, default: None
            Positional encodings for the interaction queries.
        kv_pos: Tensor, default: None
            Positional encodings for the image features.
        return_q_attn_weights: bool, default: False
            If `True`, return the self attention weights.
        return_qk_attn_weights: bool, default: False
            If `True`, return the cross attention weights.

        Returns:
        --------
        outputs: list
            A list with the order [queries, q_attn_w, qk_attn_w], if both weights are
            to be returned.
        """
        q = k = self.with_pos_embed(queries, q_pos)
        # Perform self attention amongst queries
        q_attn, q_attn_weights = self.q_attn(
            q, k, value=queries, attn_mask=q_mask,
            key_padding_mask=q_padding_mask
        )
        queries = self.ln1(queries + self.dp1(q_attn))
        # Perform cross attention from memory features to queries
        qk_attn, qk_attn_weights = self.qk_attn(
            query=self.with_pos_embed(queries, q_pos),
            key=self.with_pos_embed(features, kv_pos),
            value=features, attn_mask=kv_mask,
            key_padding_mask=kv_padding_mask
        )
        queries = self.ln2(queries + self.dp2(qk_attn))
        queries = self.ln3(queries + self.dp3(self.ffn(queries)))

        outputs = [queries,]
        if return_q_attn_weights:
            outputs.append(q_attn_weights)
        if return_qk_attn_weights:
            outputs.append(qk_attn_weights)
        return outputs

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(decoder_layer.q_size)
        self.return_intermediate = return_intermediate

    def forward(self, queries, features,
            q_mask: Optional[Tensor] = None,
            kv_mask: Optional[Tensor] = None,
            q_padding_mask: Optional[Tensor] = None,
            kv_padding_mask: Optional[Tensor] = None,
            q_pos: Optional[Tensor] = None,
            kv_pos: Optional[Tensor] = None,
            return_q_attn_weights: Optional[bool] = False,
            return_qk_attn_weights: Optional[bool] = False
        ):
        # Add support for zero layers
        if self.num_layers == 0:
            return queries.unsqueeze(0)

        outputs = [queries,]
        intermediate = []
        q_attn_w = []
        qk_attn_w = []
        for layer in self.layers:
            outputs = layer(
                outputs[0], features,
                q_mask=q_mask, kv_mask=kv_mask,
                q_padding_mask=q_padding_mask,
                kv_padding_mask=kv_padding_mask,
                q_pos=q_pos, kv_pos=kv_pos,
                return_q_attn_weights=return_q_attn_weights,
                return_qk_attn_weights=return_qk_attn_weights
            )
            if self.return_intermediate:
                intermediate.append(self.norm(outputs[0]))
            if return_q_attn_weights:
                q_attn_w.append(outputs[1])
            if return_qk_attn_weights:
                qk_attn_w.append(outputs[2])

        if self.return_intermediate:
            return torch.stack(intermediate)

        outputs = [self.norm(outputs[0]).unsqueeze(0),]
        if return_q_attn_weights:
            q_attn_w = torch.stack(q_attn_w)
            outputs.append(q_attn_w)
        if return_qk_attn_weights:
            qk_attn_w = torch.stack(qk_attn_w)
            outputs.append(qk_attn_w)
        return outputs

class UPT(nn.Module):
    """
    Unary-pairwise transformer

    Parameters:
    -----------
    detector: nn.Module
        Object detector (DETR)
    postprocessor: nn.Module
        Postprocessor for the object detector
    interaction_head: nn.Module
        Interaction head of the network
    human_idx: int
        Index of the human class
    num_classes: int
        Number of action classes
    alpha: float
        Hyper-parameter in the focal loss
    gamma: float
        Hyper-parameter in the focal loss
    box_score_thresh: float
        Threshold used to eliminate low-confidence objects
    fg_iou_thresh: float
        Threshold used to associate detections with ground truth
    min_instances: float
        Minimum number of instances (human or object) to sample
    max_instances: float
        Maximum number of instances (human or object) to sample
    """
    def __init__(self,
        detector: nn.Module,
        postprocessor: nn.Module,
        interaction_head: nn.Module,
        human_idx: int, num_classes: int,
        alpha: float = 0.5, gamma: float = 2.0,
        box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
        min_instances: int = 3, max_instances: int = 15,
    ) -> None:
        super().__init__()
        self.backbone = detector.backbone
        self.transformer = detector.transformer
        self.class_embed = detector.class_embed
        self.bbox_embed = detector.bbox_embed
        self.input_proj = detector.input_proj
        self.query_embed = detector.query_embed

        self.postprocessor = postprocessor
        self.interaction_head = interaction_head

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh

        self.min_instances = min_instances
        self.max_instances = max_instances

    def recover_boxes(self, boxes, size):
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets):
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)

        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])

        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)

        labels[x, targets['labels'][y]] = 1

        return labels

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets):
        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]

        n_p = len(torch.nonzero(labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
        )

        return loss / n_p

    def prepare_region_proposals(self, results, hidden_states):
        region_props = []
        for res, hs in zip(results, hidden_states):
            sc, lb, bx = res.values()

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            hs = hs[keep].view(-1, 256)

            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum(); n_object = len(keep) - n_human
            # Keep the number of human and object instances in a specified interval
            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
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

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, attn_maps, image_sizes):
        n = [len(b) for b in bh]
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, attn, size in zip(
            boxes, bh, bo, logits, prior, objects, attn_maps, image_sizes
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            scores = torch.sigmoid(lg[x, y])
            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                objects=obj[x], attn_maps=attn, size=size
            ))

        return detections

    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (2, M) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `attn_maps`: list
                Attention weights in the cooperative and competitive layers
            `size`: torch.Tensor
                (2,) Image height and width
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images
        ], device=images[0].device)

        # --------------------------- #
        # Stage one: object detection #
        # --------------------------- #
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        features, pos = self.backbone(images)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        results = self.postprocessor(results, image_sizes)

        # ---------------------------------- #
        # Stage two: interaction recognition #
        # ---------------------------------- #
        region_props = self.prepare_region_proposals(results, hs[-1])

        logits, prior, bh, bo, objects, attn_maps = self.interaction_head(
            memory, image_sizes, region_props
        )
        boxes = [r['boxes'] for r in region_props]

        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets)
            loss_dict = dict(
                interaction_loss=interaction_loss
            )
            return loss_dict

        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, attn_maps, image_sizes)
        return detections

def build_detector(args, class_corr):
    detr, _, postprocessors = build_model(args)
    if os.path.exists(args.pretrained):
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
        detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])
    predictor = torch.nn.Linear(args.repr_dim * 2, args.num_classes)
    interaction_head = InteractionHead(
        predictor, args.hidden_dim, args.repr_dim,
        detr.backbone[0].num_channels,
        args.num_classes, args.human_idx, class_corr
    )
    detector = UPT(
        detr, postprocessors['bbox'], interaction_head,
        human_idx=args.human_idx, num_classes=args.num_classes,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        fg_iou_thresh=args.fg_iou_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
    )
    return detector
