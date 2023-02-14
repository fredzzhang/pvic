"""
Transformer decoder layer, adapted from DETR codebase by Facebook Research
https://github.com/facebookresearch/detr/blob/main/models/transformer.py#L187

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""
import copy
import torch
from torch import nn, Tensor
from typing import Optional

class TransformerDecoderLayer(nn.Module):

    def __init__(self, q_dim, kv_dim, num_heads, ffn_interm_dim, dropout=0.1):
        """
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
        self.qk_attn = nn.MultiheadAttention(
            q_dim, num_heads,
            kdim=kv_dim, vdim=kv_dim,
            dropout=dropout
        )
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
