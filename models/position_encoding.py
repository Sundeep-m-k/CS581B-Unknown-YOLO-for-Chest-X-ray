# Copyright (c) Facebook, Inc. and its affiliates.
"""
Various positional encoding implementations used by the DETR / PBC transformer.

Positional encodings provide spatial information to the transformer,
which otherwise has no notion of 2D image structure.

Supported encodings:
- Sine-based absolute positional encoding
- Learned absolute positional encoding
- Learned relative positional encoding
"""

import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    Standard sine–cosine positional encoding (as in "Attention Is All You Need"),
    adapted for 2D images.

    This produces a fixed positional encoding based on sine/cosine waves
    using normalized X/Y coordinates of each pixel location.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()

        self.num_pos_feats = num_pos_feats      # Number of features per coordinate
        self.temperature = temperature          # Frequency scaling constant
        self.normalize = normalize              # Normalize coordinates to [0,1]

        # Validation for scale usage
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        # Default scaling is 2π
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale


    def forward(self, tensor_list: NestedTensor):
        """
        Input:
            tensor_list: NestedTensor containing:
                - tensors: (B, C, H, W)
                - mask: padding mask

        Output:
            pos: Positional encoding tensor of shape (B, 2*num_pos_feats, H, W)
        """

        x = tensor_list.tensors
        mask = tensor_list.mask

        # Mask must be present for valid coordinate construction
        assert mask is not None

        # Mask: True = padding, False = valid pixels
        not_mask = ~mask

        # Compute cumulative positions:
        # Y = vertical positions
        # X = horizontal positions
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        # Normalize coordinates if enabled
        if self.normalize:
            eps = 1e-6  # Avoid divide by zero
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Generate frequency scale vector
        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=x.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Divide coordinates by frequencies
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # Apply sin/cos alternating encoding
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(),
             pos_x[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)

        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(),
             pos_y[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)

        # Combine (Y, X) and rearrange to (B, C, H, W)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


    def calc_emb(self, normed_coord):
        """
        Compute positional embedding directly for point coordinates.

        Used when positional embedding must be computed on sparse annotated points.

        Input:
            normed_coord: (N,2) normalized x,y coordinates

        Output:
            pos embedding: (N, 2*num_pos_feats)
        """

        normed_coord = normed_coord.clamp(0., 1.) * self.scale
        device = normed_coord.device

        # Frequency scale vector
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Apply sine-cosine encoding same as grid version
        pos_x = normed_coord[:, 0, None] / dim_t
        pos_y = normed_coord[:, 1, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(),
             pos_x[:, 1::2].cos()),
            dim=2
        ).flatten(1)

        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(),
             pos_y[:, 1::2].cos()),
            dim=2
        ).flatten(1)

        pos = torch.cat((pos_y, pos_x), dim=1)

        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Learned absolute positional embedding.

    Uses embedding tables for row and column indices.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()

        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)


    def forward(self, tensor_list: NestedTensor):
        """
        Generates full spatial positional encoding grid from learned embeddings.
        """

        x = tensor_list.tensors
        h, w = x.shape[-2:]

        # Create row and column indices
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        # Combine row and column embeddings
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1)
        ], dim=-1)

        # Format to (B, C, H, W)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(
            x.shape[0], 1, 1, 1
        )

        return pos


class PositionEmbeddingRelativeLearned(nn.Module):
    """
    Learned relative positional encoding.

    Models position as weighted combinations of embedding bins along X and Y.
    Coordinates are normalized and interpolated across the bins.
    """

    def __init__(self, num_pos_feats=256, num_emb=51):
        super().__init__()

        self.row_embed = nn.Embedding(num_emb, num_pos_feats)
        self.col_embed = nn.Embedding(num_emb, num_pos_feats)

        self.num_emb = num_emb
        self.reset_parameters()

        # Define slicing intervals
        self.each_piece = 1 / (num_emb - 1)
        self.slices = torch.tensor(
            [_ * self.each_piece for _ in range(num_emb)]
        )


    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)


    def forward(self, tensor_list: NestedTensor):
        """
        Generates smooth relative positional embeddings using interpolation.
        """

        x = tensor_list.tensors
        mask = tensor_list.mask

        assert mask is not None

        not_mask = ~mask

        # Cumulative pixel coordinates
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        # Normalize to [0,1]
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps)
        x_embed = x_embed / (x_embed[:, :, -1:] + eps)

        # Weighted interpolation across bins
        weighted_col_emb = self.calc_weighted_emb_2d(
            x_embed,
            self.slices.to(x.device),
            self.col_embed.weight
        )

        weighted_row_emb = self.calc_weighted_emb_2d(
            y_embed,
            self.slices.to(x.device),
            self.row_embed.weight
        )

        # Combine X and Y positional embeddings
        return torch.cat(
            [weighted_col_emb, weighted_row_emb], dim=3
        ).permute(0, 3, 1, 2)


    def calc_weighted_emb_2d(self, coord, slices, embed):
        """
        Interpolates embedding weights across bins along 2D grid.
        """

        dis = abs(coord[:, :, :, None] - slices[None, :])
        weight = (-dis + self.each_piece) * (dis < self.each_piece) / self.each_piece

        return torch.matmul(weight, embed)


    def calc_emb(self, normed_coord):
        """
        Compute relative positional embedding for point coordinates only.
        """

        normed_coord = normed_coord.clamp(0., 1.)
        device = normed_coord.device

        weighted_col_emb = self.calc_weighted_emb(
            normed_coord[:, 0],
            self.slices.to(device),
            self.col_embed.weight
        )

        weighted_row_emb = self.calc_weighted_emb(
            normed_coord[:, 1],
            self.slices.to(device),
            self.row_embed.weight
        )

        return torch.cat([weighted_col_emb, weighted_row_emb], dim=1)


    def calc_weighted_emb(self, coord, slices, embed):
        """
        Interpolates 1D positional embedding for point coordinates.
        """

        dis = abs(coord[:, None] - slices[None, :])
        weight = (self.each_piece - dis) * (dis < self.each_piece) / self.each_piece

        return weight @ embed


def build_position_encoding(args):
    """
    Factory method to build the requested positional encoding
    based on user arguments.
    """

    N_steps = args.hidden_dim // 2

    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)

    elif args.position_embedding in ('v4', 'relative-learned'):
        position_embedding = PositionEmbeddingRelativeLearned(N_steps)

    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
