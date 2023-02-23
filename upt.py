"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

import os
import copy
import torch
import pocket
import torch.nn.functional as F
import torch.distributed as dist

from torch import nn, Tensor
from typing import Optional, List, Tuple
from collections import OrderedDict
from swint import SwinTransformerBlockV2
from attention import MultiheadAttention
from torchvision.ops import FeaturePyramidNetwork

from ops import (
    binary_focal_loss_with_logits,
    compute_spatial_encodings,
    prepare_region_proposals,
    associate_with_ground_truth,
    compute_prior_scores,
    compute_sinusoidal_pe
)

from detr.models import build_model
from detr.util.misc import nested_tensor_from_tensor_list

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

class ModifiedEncoderLayer(nn.Module):
    def __init__(self,
        hidden_size: int, repr_size: int,
        num_heads: int = 8, dropout_prob: float = .1, return_weights: bool = False,
    ) -> None:
        super().__init__()
        if repr_size % num_heads != 0:
            raise ValueError(
                f"The given representation size {repr_size} "
                f"should be divisible by the number of attention heads {num_heads}."
            )
        self.sub_repr_size = int(repr_size / num_heads)

        self.hidden_size = hidden_size
        self.repr_size = repr_size

        self.num_heads = num_heads
        self.return_weights = return_weights

        self.unary = nn.Linear(hidden_size, repr_size)
        self.pairwise = nn.Linear(repr_size, repr_size)
        self.attn = nn.ModuleList([nn.Linear(3 * self.sub_repr_size, 1) for _ in range(num_heads)])
        self.message = nn.ModuleList([nn.Linear(self.sub_repr_size, self.sub_repr_size) for _ in range(num_heads)])
        self.aggregate = nn.Linear(repr_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

        self.ffn = pocket.models.FeedForwardNetwork(hidden_size, hidden_size * 4, dropout_prob)

    def reshape(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.sub_repr_size
        )
        x = x.view(*new_x_shape)
        if len(new_x_shape) == 3:
            return x.permute(1, 0, 2)
        elif len(new_x_shape) == 4:
            return x.permute(2, 0, 1, 3)
        else:
            raise ValueError("Incorrect tensor shape")

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        device = x.device
        n = len(x)

        u = F.relu(self.unary(x))
        p = F.relu(self.pairwise(y))

        # Unary features (H, N, L)
        u_r = self.reshape(u)
        # Pairwise features (H, N, N, L)
        p_r = self.reshape(p)

        i, j = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device))
        # Features used to compute attention (H, N, N, 3L)
        attn_features = torch.cat([u_r[:, i], u_r[:, j], p_r], dim=-1)
        # Attention weights (H,) (N, N, 1)
        weights = [
            F.softmax(l(f), dim=0) for f, l
            in zip(attn_features, self.attn)
        ]
        # Repeated unary feaures along the third dimension (H, N, N, L)
        u_r_repeat = u_r.unsqueeze(dim=2).repeat(1, 1, n, 1)
        messages = [
            l(f_1 * f_2) for f_1, f_2, l
            in zip(u_r_repeat, p_r, self.message)
        ]
        aggregated_messages = self.aggregate(F.relu(
            torch.cat([
                (w * m).sum(dim=0) for w, m
                in zip(weights, messages)
            ], dim=-1)
        ))
        aggregated_messages = self.dropout(aggregated_messages)
        x = self.norm(x + aggregated_messages)
        x = self.ffn(x)

        if self.return_weights: attn = weights
        else: attn = None

        return x, attn

class ModifiedEncoder(nn.Module):
    def __init__(self,
        hidden_size: int = 256, repr_size: int = 384,
        num_heads: int = 8, num_layers: int = 2,
        dropout_prob: float = .1, return_weights: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.mod_enc = nn.ModuleList([ModifiedEncoderLayer(
            hidden_size=hidden_size, repr_size=repr_size,
            num_heads=num_heads, dropout_prob=dropout_prob, return_weights=return_weights
        ) for _ in range(num_layers)])

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, List[Optional[Tensor]]]:
        attn_weights = []
        for layer in self.mod_enc:
            x, attn = layer(x, y)
            attn_weights.append(attn)
        return x, attn_weights

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
        self.encoder = ModifiedEncoder(256, repr_size, num_layers=2)
        self.mmf = MultiModalFusion(512, repr_size, repr_size)

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
            x = x.flatten(); y = y.flatten()
            # Compute spatial features
            pairwise_spatial = compute_spatial_encodings(
                [boxes[x],], [boxes[y],], [image_sizes[i],]
            )
            pairwise_spatial = self.spatial_head(pairwise_spatial)
            pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)

            embeds, _ = self.encoder(embeds, pairwise_spatial_reshaped)
            # Compute human-object queries
            ho_q = self.mmf(
                torch.cat([embeds[x_keep], embeds[y_keep]], dim=1),
                pairwise_spatial_reshaped[x_keep, y_keep]
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

class VerbMatcher(nn.Module):
    def __init__(self, repr_size, triplet_to_obj, triplet_embeds, num_objs=80):
        super().__init__()
        self.repr_size = repr_size
        self.triplet_to_obj = triplet_to_obj
        self.num_objs = num_objs
        self.register_buffer("triplet_embeds", triplet_embeds.type(torch.float32))

        self.mmf = MultiModalFusion(repr_size, triplet_embeds.shape[1], repr_size)
    def forward(self, ho_queries, object_types, triplet_cands):
        device = ho_queries[0].device

        mm_queries = []
        dup_indices = []
        triplet_indices = []
        for ho_q, objs, t_cands in zip(ho_queries, object_types, triplet_cands):
            obj_to_triplet = [[] for _ in range(self.num_objs)]
            # Create an object-to-triplet mapping from the candidates
            for t in t_cands:
                obj_to_triplet[self.triplet_to_obj[t.item()]].append(t.item())
            # Retrieve matched triplets and indices for duplicating ho pairs
            matched_vb = torch.as_tensor([
                (i, t) for i, o in enumerate(objs)
                for t in obj_to_triplet[o.item()]
            ], device=device)

            # Handle images without valid ho pairs
            if len(matched_vb) == 0:
                mm_queries.append(torch.zeros(0, self.repr_size, device=device))
                dup_indices.append(torch.zeros(0, device=device, dtype=torch.long))
                triplet_indices.append(torch.zeros(0, device=device, dtype=torch.long))
                continue

            dup_inds, t_inds = matched_vb.unbind(1)

            mm_queries.append(self.mmf(
                ho_q[dup_inds], self.triplet_embeds[t_inds]
            ))
            dup_indices.append(dup_inds)
            triplet_indices.append(t_inds)

        return mm_queries, dup_indices, triplet_indices

class SwinTransformer(nn.Module):

    def __init__(self, dim, num_layers):
        """
        A feature stage consisting of a series of Swin Transformer V2 blocks.

        Parameters:
        -----------
        dim: int
            Dimension of the input features.
        """
        super().__init__()
        self.dim = dim

        self.depth = num_layers
        self.num_heads = dim // 32
        self.window_size = 8
        self.base_sd_prob = 0.2

        shift_size = [
            [self.window_size // 2] * 2 if i % 2
            else [0, 0] for i in range(self.depth)
        ]
        # Use stochastic depth parameters for the third stage of Swin-T variant.
        sd_prob = (torch.linspace(0, 1, 12)[10-num_layers:10] * self.base_sd_prob).tolist()

        blocks: List[nn.Module] = []
        for i in range(self.depth):
            blocks.append(SwinTransformerBlockV2(
                dim=dim, num_heads=self.num_heads,
                window_size=[self.window_size] * 2,
                shift_size=shift_size[i],
                stochastic_depth_prob=sd_prob[i]
            ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        """
        Parameters:
        -----------
        x: Tensor
            Input features maps of shape (B, H, W, C).

        Returns:
        --------
        Tensor
            Output feature maps of shape (B, H, W, C).
        """
        return self.blocks(x)

class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims
    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)

class FeatureHead(nn.Module):
    def __init__(self, dim, dim_backbone, return_layer, num_layers):
        super().__init__()
        self.dim = dim
        self.dim_backbone = dim_backbone
        self.return_layer = return_layer

        in_channel_list = [
            int(dim_backbone * 2 ** i)
            for i in range(return_layer + 1, 1)
        ]
        self.fpn = FeaturePyramidNetwork(in_channel_list, dim)
        self.layers = nn.Sequential(
            Permute([0, 2, 3, 1]),
            SwinTransformer(dim, num_layers)
        )
    def forward(self, x):
        pyramid = OrderedDict(
            (f"{i}", x[i].tensors)
            for i in range(self.return_layer, 0)
        )
        mask = x[self.return_layer].mask
        x = self.fpn(pyramid)[f"{self.return_layer}"]
        x = self.layers(x)
        return x, mask

class TransformerDecoderLayer(nn.Module):

    def __init__(self, q_dim, kv_dim, num_heads, ffn_interm_dim, dropout=0.1):
        """
        Transformer decoder layer, adapted from DETR codebase by Facebook Research
        https://github.com/facebookresearch/detr/blob/main/models/transformer.py#L187

        Parameters:
        -----------
        q_dim: int
            Dimension of the interaction queries.
        kv_dim: int
            Dimension of the image features.
        num_heads: int
            Number of heads used in multihead attention.
        ffn_interm_dim: int
            Dimension of the intermediate representation in the feedforward network.
        dropout: float, default: 0.1
            Dropout percentage used during training.
        """
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.ffn_interm_dim = ffn_interm_dim

        self.q_attn = nn.MultiheadAttention(q_dim, num_heads, dropout=dropout)
        # Linear projects on qkv have been removed in this custom layer
        self.qk_attn = MultiheadAttention(q_dim * 2, num_heads, dropout=dropout, vdim=q_dim)
        # Add the missing linear projections
        self.qk_attn_q_proj = nn.Linear(q_dim, q_dim)
        self.qk_attn_k_proj = nn.Linear(kv_dim, q_dim)
        self.qk_attn_v_proj = nn.Linear(kv_dim, q_dim)
        self.qk_attn_kpos_proj = nn.Linear(kv_dim, q_dim)
        self.qk_attn_qpos_proj = nn.Linear(kv_dim * 2, q_dim)

        self.ffn = nn.Sequential(
            nn.Linear(q_dim, ffn_interm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_interm_dim, q_dim)
        )
        self.ln1 = nn.LayerNorm(q_dim)
        self.ln2 = nn.LayerNorm(q_dim)
        self.ln3 = nn.LayerNorm(q_dim)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.dp3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
            queries: Tensor, features: Tensor,
            q_pos: Tensor, k_pos: Tensor,
            q_mask: Optional[Tensor] = None,
            kv_mask: Optional[Tensor] = None,
            q_padding_mask: Optional[Tensor] = None,
            kv_padding_mask: Optional[Tensor] = None,
        ):
        """
        Parameters:
        -----------
        queries: Tensor
            Interaction queries of size (B, N, K).
        features: Tensor
            Image features of size (HW, B, C).
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
        k_pos: Tensor, default: None
            Positional encodings for the image features.

        Returns:
        --------
        outputs: list
            A list with the order [queries, q_attn_w, qk_attn_w].
        """
        # Remove positional embeddings in self-attention
        q = k = queries
        # Perform self attention amongst queries
        q_attn, q_attn_weights = self.q_attn(
            q, k, value=queries, attn_mask=q_mask,
            key_padding_mask=q_padding_mask
        )
        queries = self.ln1(queries + self.dp1(q_attn))
        # Perform cross attention from memory features to queries
        q = self.qk_attn_q_proj(queries)
        k = self.qk_attn_k_proj(features)
        v = self.qk_attn_v_proj(features)
        q_pos = self.qk_attn_qpos_proj(q_pos)
        k_pos = self.qk_attn_kpos_proj(k_pos)

        n_q, bs, _ = q.shape
        q = q.view(n_q, bs, self.num_heads, self.q_dim // self.num_heads)
        q_pos = q_pos.view(n_q, bs, self.num_heads, self.q_dim // self.num_heads)
        q = torch.cat([q, q_pos], dim=3).view(n_q, bs, self.q_dim * 2)

        hw, _, _ = k.shape
        k = k.view(hw, bs, self.num_heads, self.q_dim // self.num_heads)
        k_pos = k_pos.view(hw, bs, self.num_heads, self.q_dim // self.num_heads)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, self.q_dim * 2)

        qk_attn, qk_attn_weights = self.qk_attn(
            query=q, key=k, value=v,
            key_padding_mask=kv_padding_mask
        )
        queries = self.ln2(queries + self.dp2(qk_attn))
        queries = self.ln3(queries + self.dp3(self.ffn(queries)))

        outputs = [queries, q_attn_weights, qk_attn_weights]
        return outputs

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(decoder_layer.q_dim)
        self.return_intermediate = return_intermediate

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, queries, features,
            q_mask: Optional[Tensor] = None,
            kv_mask: Optional[Tensor] = None,
            q_padding_mask: Optional[Tensor] = None,
            kv_padding_mask: Optional[Tensor] = None,
            q_pos: Optional[Tensor] = None,
            k_pos: Optional[Tensor] = None,
            return_q_attn_weights: Optional[bool] = False,
            return_qk_attn_weights: Optional[bool] = False
        ):
        # Add support for zero layers
        if self.num_layers == 0:
            return [queries.unsqueeze(0),]
        # Explicitly handle zero-size queries
        if queries.numel() == 0:
            rp = self.num_layers if self.return_intermediate else 1
            return [queries.unsqueeze(0).repeat(rp, 1, 1, 1),]

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
                q_pos=q_pos, k_pos=k_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(outputs[0]))
            if return_q_attn_weights:
                q_attn_w.append(outputs[1])
            if return_qk_attn_weights:
                qk_attn_w.append(outputs[2])

        if self.return_intermediate:
            outputs = [torch.stack(intermediate),]
        else:
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
        feature_head: nn.Module,
        backbone_fusion_layer: int,
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
        self.fusion_layer = backbone_fusion_layer
        self.decoder = triplet_decoder
        self.binary_classifier = nn.Linear(repr_size, num_verbs)

        self.kv_dim = self.transformer.d_model
        self.h_ref_anchor_head = nn.Sequential(
            nn.Linear(repr_size, repr_size), nn.ReLU(),
            nn.Linear(repr_size, 2)
        )
        self.o_ref_anchor_head = nn.Sequential(
            nn.Linear(repr_size, repr_size), nn.ReLU(),
            nn.Linear(repr_size, 2)
        )

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

    def compute_query_pe(self, boxes, p_inds, ho_queries, image_sizes):
        q_pos = []
        for bx, p, ho_q, (h, w) in zip(boxes, p_inds, ho_queries, image_sizes):
            bx_c = (bx[:, :2] + bx[:, 2:]) / 2
            bx_c /= torch.stack([w, h])
            b_wh = bx[:, 2:] - bx[:, :2]
            b_wh /= torch.stack([w, h])

            bh_wh, bo_wh = b_wh[p].unbind(1)

            pe = compute_sinusoidal_pe(bx_c[:, None])
            pe = pe[p].squeeze(2)   # n_query, 2, kv_dim

            # Modulate the positional embeddings with box widths and heights by
            # applying different temperatures to x and y
            hum_ref_hw_cond = self.h_ref_anchor_head(ho_q).sigmoid()    # n_query, 2
            obj_ref_hw_cond = self.o_ref_anchor_head(ho_q).sigmoid()
            # Note that the positional embeddings are stacked as [pe(y), pe(x)]
            pe[:, 0, :self.kv_dim // 2] *= (hum_ref_hw_cond[:, 1] / bh_wh[:, 1]).unsqueeze(-1)
            pe[:, 0, self.kv_dim // 2:] *= (hum_ref_hw_cond[:, 0] / bh_wh[:, 0]).unsqueeze(-1)
            pe[:, 1, :self.kv_dim // 2] *= (obj_ref_hw_cond[:, 1] / bo_wh[:, 1]).unsqueeze(-1)
            pe[:, 1, self.kv_dim // 2:] *= (obj_ref_hw_cond[:, 0] / bo_wh[:, 0]).unsqueeze(-1)

            # Concatenate pe for human and object boxes at the last dimension
            q_pos.append(pe.view(-1, self.kv_dim * 2))                  # n_query, kv_dim * 2
        return q_pos

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
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])

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
        # if triplet_cands is None:
            # triplet_cands = [torch.arange(self.num_triplets).tolist() for _ in range(len(boxes))]
            # triplet_cands = [None for _ in range(len(boxes))]
        # mm_queries, dup_inds, triplet_inds = self.vb_matcher(ho_queries, object_types, triplet_cands)
        # padded_queries, q_padding_mask = pad_queries(mm_queries)

        memory, mask = self.feature_head(features)
        b, h, w, c = memory.shape
        memory = memory.reshape(b, h * w, c)
        kv_p_m = mask.reshape(-1, 1, h * w)
        k_pos = pos[self.fusion_layer].permute(0, 2, 3, 1).reshape(b, h * w, 1, c)
        q_pos = self.compute_query_pe(boxes, paired_inds, ho_queries, image_sizes)

        output_queries = []
        for i, (ho_q, mem) in enumerate(zip(ho_queries, memory)):
            output_queries.append(self.decoder(
                ho_q.unsqueeze(1), mem.unsqueeze(1),
                # q_padding_mask=q_padding_mask,
                kv_padding_mask=kv_p_m[i],
                q_pos=q_pos[i].unsqueeze(1),
                k_pos=k_pos[i]
            )[0].squeeze(dim=2))
        mm_queries_collate = torch.cat(output_queries, dim=1)
        # mm_queries_collate = torch.cat([
        #     mm_q[q_p_m] for mm_q, q_p_m
        #     in zip(padded_queries, q_padding_mask)
        # ])
        logits = self.binary_classifier(mm_queries_collate)

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

    decoder_layer = TransformerDecoderLayer(
        q_dim=args.repr_dim, kv_dim=args.hidden_dim,
        ffn_interm_dim=args.repr_dim * 4,
        num_heads=args.nheads, dropout=args.dropout
    )
    triplet_decoder = TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=args.triplet_dec_layers,
        return_intermediate=args.triplet_aux_loss
    )
    feature_head = FeatureHead(
        args.hidden_dim,
        detr.backbone.num_channels,
        args.backbone_fusion_layer,
        args.triplet_enc_layers
    )
    detector = UPT(
        detr, postprocessors['bbox'],
        feature_head=feature_head,
        backbone_fusion_layer=args.backbone_fusion_layer,
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
