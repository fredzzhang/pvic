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
import torchvision.transforms.functional as T

from torch import nn, Tensor
from torch.nn.init import normal_
from typing import Optional, List

from ops import (
    binary_focal_loss_with_logits,
    compute_spatial_encodings,
    prepare_region_proposals,
    associate_with_ground_truth,
    compute_prior_scores,
    pad_queries
)

import d_detr.models.deformable_transformer as M

import sys
sys.path.append('detr')
from models import build_model
from util.misc import NestedTensor, nested_tensor_from_tensor_list


class MultiModalFusion(nn.Module):
    def __init__(self, fst_mod_size, scd_mod_size, repr_size):
        super().__init__()
        self.fc1 = nn.Linear(fst_mod_size, repr_size)
        self.fc2 = nn.Linear(scd_mod_size, repr_size)
        self.ln1 = nn.LayerNorm(repr_size)
        self.ln2 = nn.LayerNorm(repr_size)

        mlp = []
        repr_size = [2 * repr_size, int(repr_size * 1.5), repr_size]
        for d_in, d_out in zip(repr_size[:-1], repr_size[1:]):
            mlp.append(nn.Linear(d_in, d_out))
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.ln1(self.fc1(x))
        y = self.ln2(self.fc2(y))
        z = F.relu(torch.cat([x, y], dim=-1))
        z = self.mlp(z)
        return z

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
        self.mbf = MultiModalFusion(512, repr_size, repr_size)

    def check_human_instances(self, labels):
        is_human = labels == self.human_idx
        n_h = torch.sum(is_human)
        if not torch.all(labels[:n_h]==self.human_idx):
            raise AssertionError("Human instances are not permuted to the top!")
        return n_h

    def forward(self, region_props, image_sizes, device=None):
        if device is None:
            device = region_props[0]["hidden_states"].device

        ho_queries = []
        paired_indices = []
        prior_scores = []
        object_types = []
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
                ho_queries.append(torch.zeros(0, self.repr_size, device=device))
                paired_indices.append(torch.zeros(0, 2, device=device, dtype=torch.int64))
                prior_scores.append(torch.zeros(0, 2, self.num_verbs, device=device))
                object_types.append(torch.zeros(0, device=device, dtype=torch.int64))
                continue
            # Compute spatial features
            pairwise_spatial = compute_spatial_encodings(
                [boxes[x_keep],], [boxes[y_keep],], [image_sizes[i],]
            )
            pairwise_spatial = self.spatial_head(pairwise_spatial)
            # Compute human-object queries
            ho_q = self.mbf(
                torch.cat([embeds[x_keep], embeds[y_keep]], dim=1),
                pairwise_spatial
            )
            # Append matched human-object pairs
            ho_queries.append(ho_q)
            paired_indices.append(torch.stack([x_keep, y_keep], dim=1))
            prior_scores.append(compute_prior_scores(
                x_keep, y_keep, scores, labels, self.num_verbs, self.training,
                self.obj_to_verb
            ))
            object_types.append(labels[y_keep])

        return ho_queries, paired_indices, prior_scores, object_types

class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims
    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)

class FeatureHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.num_feature_levels = 4
        self.feature_dim = [512, 1024, 2048, 2048]

        input_proj_list = []
        for i, fd in enumerate(self.feature_dim):
            if i == 3:
                input_proj_list.append(nn.Sequential(
                    # Stride 2 conv. to create C6.
                    nn.Conv2d(fd, dim, 3, stride=2, padding=1),
                    nn.GroupNorm(32, dim),
                ))
            else:
                input_proj_list.append(nn.Sequential(
                    # Only align feature dimensions for C3-C5.
                    nn.Conv2d(fd, dim, 1),
                    nn.GroupNorm(32, dim)
                ))
        self.input_proj = nn.ModuleList(input_proj_list)

        encoder_layer = M.DeformableTransformerEncoderLayer(
            d_model=dim, d_ffn=dim * 4
        )
        self.encoder = M.DeformableTransformerEncoder(
            encoder_layer, 6
        )
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, dim)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, M.MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, features, image_padding_mask, pos_embeds, pe):
        # Construct multiscale features.
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            # Skip C2 features.
            if l == 0:
                continue
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l-1](src))
            masks.append(mask)
        # Construct additional C6 (1/64) level.
        feat_c6 = self.input_proj[-1](features[-1].tensors)
        mask_c6 = F.interpolate(
            image_padding_mask[None].float(),
            size=feat_c6.shape[-2:]
        ).to(torch.bool)[0]
        pos_c6 = pe(NestedTensor(feat_c6, mask_c6)).to(feat_c6.dtype)
        srcs.append(feat_c6)
        masks.append(mask_c6)
        pos_embeds.append(pos_c6)

        # Prepare inputs for the encoder.
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            h, w = src.shape[2:]
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(
            src_flatten, spatial_shapes, level_start_index,
            valid_ratios, lvl_pos_embed_flatten, mask_flatten
        )

        return memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten

class MSDeformAttn(M.MSDeformAttn):
    def __init__(self, q_dim, kv_dim, n_levels=4, n_heads=8, n_points=4):
        """
        A wrapper of the multiscale deformable attention module to allow
        keys and queries to have different dimensions.
        """
        # Still have to call this method for nn.Module to be properly
        # initialised, although certain modules will be overriden.
        super().__init__(q_dim, n_levels, n_heads, n_points)
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.value_proj = nn.Linear(kv_dim, q_dim)

        self._reset_parameters()

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self,
        q_dim, kv_dim, d_ffn,
        dropout=0.1, n_levels=4, n_heads=8, n_points=4
    ):
        super().__init__()
        self.q_attn = nn.MultiheadAttention(q_dim, n_heads, dropout=dropout)
        self.qk_attn = MSDeformAttn(
            q_dim=q_dim, kv_dim=kv_dim,
            n_levels=n_levels, n_heads=n_heads, n_points=n_points
        )
        self.ffn = nn.Sequential(
            nn.Linear(q_dim, d_ffn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, q_dim)
        )
        self.ln1 = nn.LayerNorm(q_dim)
        self.ln2 = nn.LayerNorm(q_dim)
        self.ln3 = nn.LayerNorm(q_dim)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.dp3 = nn.Dropout(dropout)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self,
        queries, features, reference_points,
        kv_spatial_shapes, level_start_index,
        q_pos: Optional[Tensor] = None,
        kv_padding_mask: Optional[Tensor] = None
    ):
        q = k = self.with_pos_embed(queries, q_pos)
        # Perform self attention amongst queries.
        q_attn = self.q_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            queries.transpose(0, 1)
        )[0].transpose(0, 1)
        queries = self.ln1(queries + self.dp1(q_attn))
        # Perform cross attention from memory features to queries.
        qk_attn = self.qk_attn(
            self.with_pos_embed(queries, q_pos),
            reference_points, features,
            kv_spatial_shapes, level_start_index, kv_padding_mask
        )
        queries = self.ln2(queries + self.dp2(qk_attn))
        queries = self.ln3(queries + self.dp3(self.ffn(queries)))

        return queries

class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        if return_intermediate and num_layers == 0:
            raise ValueError("Zero layers are not supported when \'return_intermediate\' is set to True.")

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, M.MSDeformAttn):
                m._reset_parameters()

    def forward(self,
        queries, features, reference_points,
        kv_spatial_shapes, kv_valid_ratios, level_start_index,
        q_pos: Optional[Tensor] = None,
        kv_padding_mask: Optional[Tensor] = None
    ):
        output = queries
        intermediate = []
        for layer in self.layers:
            # The valid ratios are not used to get the precise position of the box centres.
            reference_points_input = reference_points[:, :, None]
            # reference_points_input = reference_points[:, :, None] * kv_valid_ratios[:, None]
            output = layer(
                output, features, reference_points_input,
                kv_spatial_shapes, level_start_index,
                q_pos=q_pos, kv_padding_mask=kv_padding_mask
            )

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output[None]


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
        feature_head: nn.Module,
        triplet_decoder: nn.Module,
        # triplet_embeds: Tensor,
        # obj_to_triplet: list,
        obj_to_verb: list,
        num_verbs: int, num_triplets: int,
        repr_size: int = 384, human_idx: int = 0,
        alpha: float = 0.5, gamma: float = 2.0,
        box_score_thresh: float = 0.2,
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

        self.ho_matcher = HumanObjectMatcher(
            repr_size=repr_size,
            num_verbs=num_verbs,
            obj_to_verb=obj_to_verb,
            human_idx=human_idx,
        )
        self.feature_head = feature_head
        self.decoder = triplet_decoder
        self.binary_classifier = nn.Linear(repr_size, num_verbs)

        self.repr_size = repr_size
        self.human_idx = human_idx
        self.num_verbs = num_verbs
        self.num_triplets = num_triplets
        self.alpha = alpha
        self.gamma = gamma
        self.box_score_thresh = box_score_thresh
        self.min_instances = min_instances
        self.max_instances = max_instances

    def freeze_detector(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False
        for p in self.class_embed.parameters():
            p.requires_grad = False
        for p in self.bbox_embed.parameters():
            p.requires_grad = False
        for p in self.input_proj.parameters():
            p.requires_grad = False
        for p in self.query_embed.parameters():
            p.requires_grad = False

    def compute_classification_loss(self, logits, prior, labels):
        prior = torch.cat(prior, dim=0).prod(1)
        x, y = torch.nonzero(prior).unbind(1)

        logits = logits[:, x, y]
        prior = prior[x, y]
        labels = labels[None, x, y].repeat(len(logits), 1)

        n_p = labels.sum()
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

    def postprocessing(self,
            boxes, paired_inds, object_types,
            logits, prior, image_sizes
        ):
        n = [len(p_inds) for p_inds in paired_inds]
        logits = logits.split(n)

        detections = []
        for bx, p_inds, objs, lg, pr, size in zip(
            boxes, paired_inds, object_types,
            logits, prior, image_sizes
        ):
            pr = pr.prod(1)
            x, y = torch.nonzero(pr).unbind(1)
            scores = lg[x, y].sigmoid() * pr[x, y]
            detections.append(dict(
                boxes=bx, pairing=p_inds[x], scores=scores,
                labels=y, objects=objs[x], size=size
            ))

        return detections

    def fetch_reference_points(self, boxes, paired_inds, image_sizes):
        reference_points = []
        for bx, p_inds, sz in zip(boxes, paired_inds, image_sizes):
            h, w = sz
            cx = (bx[:, 0] + bx[:, 2]) / 2
            cy = (bx[:, 1] + bx[:, 3]) / 2
            bx_c = torch.stack([cx / w, cy / h], dim=1)
            p_bx_c = bx_c[p_inds].mean(1)
            reference_points.append(p_bx_c)
        return reference_points

    def forward(self,
        images: List[Tensor],
        triplet_cands: Optional[List[int]] = None,
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
        hs, _ = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        results = self.postprocessor(results, image_sizes)

        # ---------------------------------- #
        # Stage two: interaction recognition #
        # ---------------------------------- #
        region_props = prepare_region_proposals(
            results, hs[-1], image_sizes,
            box_score_thresh=self.box_score_thresh,
            human_idx=self.human_idx,
            min_instances=self.min_instances,
            max_instances=self.max_instances
        )
        boxes = [r['boxes'] for r in region_props]

        ho_queries, paired_inds, prior_scores, object_types = self.ho_matcher(region_props, image_sizes)

        # Compute keys/values for the triplet decoder.
        memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten = self.feature_head(features, images.mask, pos[1:], self.backbone[1])

        reference_points = self.fetch_reference_points(boxes, paired_inds, image_sizes)
        # Run decoder per image, due to the disparity in query numbers.
        ho_embeds = []
        for i, (ho_q, mem) in enumerate(zip(ho_queries, memory)):
            hs = self.decoder(
                ho_q[None], mem[None], reference_points[i][None],
                spatial_shapes, valid_ratios[i: i+1], level_start_index,
                None, mask_flatten[i: i+1]
            )
            ho_embeds.append(hs[:, 0])
        ho_embeds = torch.cat(ho_embeds, dim=1)
        logits = self.binary_classifier(ho_embeds)

        if self.training:
            labels = associate_with_ground_truth(
                boxes, paired_inds, targets, self.num_verbs
            )
            cls_loss = self.compute_classification_loss(logits, prior_scores, labels)
            loss_dict = dict(
                cls_loss=cls_loss
            )
            return loss_dict

        detections = self.postprocessing(
            boxes, paired_inds, object_types,
            logits[-1], prior_scores, image_sizes
        )
        return detections

def build_detector(args, obj_to_verb):
    detr, _, postprocessors = build_model(args)
    if os.path.exists(args.pretrained):
        if dist.is_initialized():
            print(f"Rank {dist.get_rank()}: Load weights for the object detector from {args.pretrained}")
        else:
            print(f"Load weights for the object detector from {args.pretrained}")
        detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])

    # if os.path.exists(args.triplet_embeds):
    #     triplet_embeds = torch.load(args.triplet_embeds)
    # else:
    #     raise ValueError(f"Language embeddings for triplets do not exist at {args.triplet_embeds}.")

    decoder_layer = DeformableTransformerDecoderLayer(
        q_dim=args.repr_dim, kv_dim=args.hidden_dim,
        d_ffn=args.repr_dim * 4
    )
    triplet_decoder = DeformableTransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=args.triplet_dec_layers,
        return_intermediate=args.triplet_aux_loss
    )
    feature_head = FeatureHead(args.hidden_dim)
    detector = UPT(
        detr, postprocessors['bbox'],
        feature_head=feature_head,
        triplet_decoder=triplet_decoder,
        obj_to_verb=obj_to_verb,
        num_verbs=args.num_verbs,
        num_triplets=args.num_triplets,
        repr_size=args.repr_dim,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
    )
    return detector
