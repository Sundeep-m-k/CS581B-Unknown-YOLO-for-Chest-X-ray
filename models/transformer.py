# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer implementation.

This version:
- Adds positional embeddings to attention
- Removes final layer normalization from encoder
- Decoder returns outputs from all intermediate layers (if enabled)
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):
    """
    Main Transformer module used by DETR / PBC.
    Contains:
    - Encoder: processes image feature maps
    - Decoder: processes object queries and generates predictions
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        # Build encoder block
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward,
            dropout, activation, normalize_before
        )

        # Optional final layer normalization for encoder
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        # Build decoder block
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward,
            dropout, activation, normalize_before
        )

        # Final normalization for decoder
        decoder_norm = nn.LayerNorm(d_model)

        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec
        )

        # Initialize weights
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """Initialize all parameters with Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        Forward pass.

        Inputs:
        - src: feature map from backbone (B, C, H, W)
        - mask: padding mask for images (B, H, W)
        - query_embed: object queries (list of variable length)
        - pos_embed: positional encodings

        Output:
        - ans: list of predictions per batch sample
        """

        # Flatten spatial dimensions: (B, C, H, W) -> (HW, B, C)
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        # Count how many queries exist per image in batch
        positive_num = [query_embed[idx].size(0) for idx in range(bs)]

        # Pad queries so all batches have same number
        padding_num = [max(positive_num) - positive_num[idx] for idx in range(bs)]

        for idx in range(bs):
            if padding_num[idx] == 0:
                continue
            query_embed[idx] = torch.cat([
                query_embed[idx],
                torch.zeros((padding_num[idx], c), device=src.device)
            ], dim=0)

        # Stack queries into (NumQueries, Batch, Dim)
        query_embed = torch.stack(query_embed, dim=1)

        # Flatten mask spatial dimensions
        mask = mask.flatten(1)

        # Initialize target tensor for decoder
        tgt = torch.zeros_like(query_embed)

        # Encoder forward pass
        memory = self.encoder(
            src,
            src_key_padding_mask=mask,
            pos=pos_embed
        )

        # Decoder forward pass
        hs = self.decoder(
            tgt, memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed
        )

        # Remove padded queries per image
        ans = []
        for idx in range(bs):
            ans.append(hs[:, :positive_num[idx], idx, :])

        return ans


class TransformerEncoder(nn.Module):
    """
    Encoder stack of N identical encoder layers.
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):

        output = src

        # Pass through encoder layers sequentially
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos
            )

        # Final normalization if enabled
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """
    Decoder stack producing object query outputs.
    Optionally returns all intermediate layer outputs.
    """

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None):

        output = tgt
        intermediate = []

        # Pass through decoder layers
        for layer in self.layers:
            output = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos
            )

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        # Apply final normalization
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        # Return stacked intermediate outputs if enabled
        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    """
    Single encoder layer:
      - Self-attention
      - Feed-forward network
      - Residual connections + normalization
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu",
                 normalize_before=False):

        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):

        # Apply positional embeddings
        q = k = self.with_pos_embed(src, pos)

        # Self-attention
        src2 = self.self_attn(
            q, k, value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]

        # Residual + norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        # Residual + norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    """
    Single decoder layer:
      - Self-attention on queries
      - Cross-attention with encoder memory
      - Feed-forward network
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu",
                 normalize_before=False):

        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None):

        # Self-attention on queries
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention between query and image memory
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


def _get_clones(module, N):
    """Create N independent copies of a module"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_transformer(args):
    """
    Helper function to create DETR Transformer from config arguments.
    """
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Map activation name to PyTorch function"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}")
