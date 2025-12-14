# models/backbone_swin.py

# Utility imports
from collections import OrderedDict
from typing import Dict, List

# PyTorch imports
import torch
import torch.nn.functional as F
from torch import nn

# timm library provides pretrained Swin Transformer models
import timm

# Project utilities
from util.misc import NestedTensor
from .position_encoding import build_position_encoding
from .backbone import Joiner


class SwinBackbone(nn.Module):
    """
    Swin-T (Swin Tiny) backbone implementation.

    This class is designed to behave like the standard DETR BackboneBase:
    - Takes either a NestedTensor or a plain tensor as input
    - Outputs feature maps stored in a dictionary
    - Each output is converted to a NestedTensor (feature + mask)
    - Has a `num_channels` attribute required by DETR
    """

    def __init__(self, train_backbone: bool, return_interm_layers: bool, d_model: int = 256):
        """
        Parameters:
        - train_backbone: Whether to update backbone weights during training
        - return_interm_layers: If True, use all Swin feature stages (0–3)
        - d_model: Transformer embedding dimension (DETR default: 256)
        """
        super().__init__()

        # Decide which Swin feature layers should be used
        if return_interm_layers:
            out_indices = (0, 1, 2, 3)     # Use all stages
        else:
            out_indices = (3,)            # Use only final stage

        # Load a pretrained Swin-Tiny model using timm
        # features_only=True ensures the model returns feature maps
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,
            out_indices=out_indices,
        )

        # Freeze backbone weights if required
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # Get channel size of Swin feature maps from each stage
        swin_channels = self.backbone.feature_info.channels()

        # Projection layers:
        # Convert Swin output channels to d_model size (256)
        # This ensures compatibility with the DETR transformer
        self.proj = nn.ModuleList(
            [nn.Conv2d(c, d_model, kernel_size=1) for c in swin_channels]
        )

        # Define the output channel dimension seen by the transformer
        self.num_channels = d_model

        self.return_interm_layers = return_interm_layers


    def forward(self, tensor_list):
        """
        Forward pass for Swin backbone.

        Handles both:
        - NestedTensor (images + masks) used in DETR
        - Plain torch.Tensor inputs when masks are not required
        """

        # Check if input is a NestedTensor
        is_nested = isinstance(tensor_list, NestedTensor)

        if is_nested:
            xs_in = tensor_list.tensors   # image batch (B x C x H x W)
            mask_in = tensor_list.mask   # image masks (B x H x W)
        else:
            xs_in = tensor_list
            mask_in = None

        # Extract feature maps from Swin backbone
        feats = self.backbone(xs_in)

        # Store model outputs in ordered dictionary
        out: Dict[str, NestedTensor] = OrderedDict()

        for i, x in enumerate(feats):
            # Project feature maps to d_model channels
            x_proj = self.proj[i](x)

            if is_nested:
                # Resize input mask to match feature map resolution
                assert mask_in is not None
                m = F.interpolate(
                    mask_in[None].float(), size=x_proj.shape[-2:]
                ).to(torch.bool)[0]

                # Save projected features and resized mask as NestedTensor
                out[str(i)] = NestedTensor(x_proj, m)
            else:
                # Save raw projected features for non-mask inputs
                out[str(i)] = x_proj

        return out


def build_swin_backbone(args):
    """
    Builds the complete Swin backbone module for DETR:

    Swin Backbone
        -> Position Encoding
            -> Joiner

    This mirrors the behavior of build_backbone() from backbone.py.
    """

    # Build position encoding module (sine or learned)
    position_embedding = build_position_encoding(args)

    # Determine whether backbone weights should be trained
    train_backbone = args.lr_backbone > 0

    # Whether to return feature maps at all Swin stages
    return_interm_layers = args.masks

    # DETR transformer dimension (default = 256)
    d_model = getattr(args, "hidden_dim", 256)

    # Create the Swin backbone
    backbone = SwinBackbone(
        train_backbone,
        return_interm_layers,
        d_model=d_model
    )

    # Join backbone features and positional embeddings into DETR format
    model = Joiner(backbone, position_embedding)

    # Set num_channels for transformer compatibility
    model.num_channels = backbone.num_channels

    return model
