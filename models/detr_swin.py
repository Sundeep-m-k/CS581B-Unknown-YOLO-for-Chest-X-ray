# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model with Swin Transformer backbone.

This file defines:
    - TVSwinJoiner: wrapper around torchvision Swin to work with NestedTensor
    - DETR: detection model that uses point-based queries
    - PostProcess: converts outputs to COCO-style format
    - MLP: simple feed-forward network
    - build_swin_backbone / build: factory functions for model and criterion
"""

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .point_encoder import build_point_encoder
from .label_encoder import build_label_encoder
from .point_criterion import PointCriterion
from .position_encoding import build_position_encoding


# ---------------------------------------------------------------------
# Swin wrapper:
# Converts torchvision Swin outputs into:
#   - list of NestedTensor feature maps
#   - matching positional encodings
# ---------------------------------------------------------------------
class TVSwinJoiner(torch.nn.Module):
    def __init__(self, tv_model, position_embedding):
        super().__init__()
        self.tv_model = tv_model               # torchvision Swin model
        self.position_embedding = position_embedding

    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: NestedTensor with:
                - tensors: [B, 3, H, W]
                - mask:    [B, H, W] (True for padded pixels)

        Returns:
            out: list[NestedTensor]       # feature maps at different scales
            pos: list[Tensor]            # positional encodings (same shapes)
        """
        assert isinstance(tensor_list, NestedTensor)
        x = tensor_list.tensors  # [B, 3, H, W]
        m = tensor_list.mask
        assert m is not None

        # Different torchvision Swin versions expose different APIs.
        if hasattr(self.tv_model, "features"):
            feat = self.tv_model.features(x)
        elif hasattr(self.tv_model, "forward_features"):
            feat = self.tv_model.forward_features(x)
        else:
            # Fallback: generic forward (may include classifier head)
            feat = self.tv_model(x)

        # Normalize output into a list of feature maps
        if isinstance(feat, torch.Tensor):
            xs = [feat]
        elif isinstance(feat, (list, tuple)):
            xs = list(feat)
        else:
            # Some models might return dicts
            try:
                xs = list(feat.values())
            except Exception:
                raise RuntimeError("Unsupported Swin backbone output type")

        out = []
        pos = []

        # For each feature map level from Swin
        for x in xs:
            # torchvision Swin may use channels-last layout [B, H, W, C]
            if x.dim() == 4:
                # Heuristic: check if shape looks like NHWC
                if x.shape[1] < x.shape[-1]:
                    # Convert NHWC -> NCHW
                    x = x.permute(0, 3, 1, 2).contiguous()

            # Now x is [B, C, H, W]
            # Resize mask from original image size to this feature map size
            mask = F.interpolate(
                m[None].float(), size=x.shape[-2:], mode="nearest"
            ).to(torch.bool)[0]

            nt = NestedTensor(x, mask)
            out.append(nt)

            # Compute positional encoding for this feature level
            pos.append(self.position_embedding(nt).to(x.dtype))

        return out, pos


# ---------------------------------------------------------------------
# DETR model with point-based queries and Swin backbone
# ---------------------------------------------------------------------
class DETR(nn.Module):
    """ DETR module with Swin backbone that performs object detection """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        label_encoder=None,
        point_encoder=None,
    ):
        """
        Initializes the model.

        Args:
            backbone: feature extractor module (e.g., Swin or ResNet)
            transformer: decoder/encoder transformer module
            num_classes: number of foreground object classes
            num_queries: number of query slots (max objects per image)
            aux_loss: if True, return auxiliary outputs for deep supervision
            label_encoder: module to encode class labels into embeddings
            point_encoder: module to encode supervision points into queries
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer

        hidden_dim = transformer.d_model  # transformer embedding dim

        # Classification head: predicts (num_classes + 1) including "no object"
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        # Bounding box head: predicts 4 box coordinates
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # Default query embeddings (not used directly in this point-based version)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Project backbone feature channels into transformer dimension
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        self.label_encoder = label_encoder
        self.point_encoder = point_encoder
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor, points_supervision):
        """
        Forward pass with point-based supervision.

        Args:
            samples: NestedTensor with:
                - samples.tensors: [B, 3, H, W]
                - samples.mask   : [B, H, W]
            points_supervision: list of dicts (per image), each with:
                - "points": (N, 2) normalized point coordinates
                - "object_ids": (N,) object id per point (if used)
                - "labels": (N,) class labels per point

        Returns:
            dict with:
                - "pred_boxes": predicted normalized boxes (cx, cy, w, h)
                - "gt_label": ground truth labels for each point
                - "aux_outputs": (optional) predictions from intermediate layers
        """
        # Allow raw tensors or lists to be converted into NestedTensor
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # Extract multi-scale features and positions from backbone
        features, pos = self.backbone(samples)

        # Use the last (highest-level) feature map
        src, mask = features[-1].decompose()
        assert mask is not None

        # Encode supervision points and labels into query embeddings
        query_embed = self.point_encoder(
            points_supervision,
            self.backbone.position_embedding,
            self.label_encoder,
        )
        bs = len(query_embed)  # batch size

        # Transformer takes projected features, masks, queries and pos encodings
        hs = self.transformer(self.input_proj(src), mask, query_embed, pos[-1])
        # hs: list/tensor of shape [batch_size, depth, num_points, hidden_dim] (implementation dependent)

        depth = hs[0].size(0)  # number of decoder layers / depth

        outputs_class = []   # list of per-image class predictions
        outputs_coord = []   # list of per-image box predictions

        # Process each image in the batch separately
        for idx in range(bs):
            cur_point_sup = points_supervision[idx]["points"]  # [num_points, 2]

            # hs[idx]: [depth, num_points, hidden_dim]
            # Class logits for each layer and point
            outputs_class.append(self.class_embed(hs[idx]))  # [depth, num_points, num_classes+1]

            # Box regression for each layer and point
            o_coord = self.bbox_embed(hs[idx]).sigmoid() / 2

            # The point coordinates act as centers / anchors.
            # Adjust predicted offsets around the given supervision points.
            o_coord[:, :, 0] = (-o_coord[:, :, 0] + cur_point_sup[None, :, 0]).clamp_(min=0.001)
            o_coord[:, :, 1] = (-o_coord[:, :, 1] + cur_point_sup[None, :, 1]).clamp_(min=0.001)
            o_coord[:, :, 2] = (o_coord[:, :, 2] + cur_point_sup[None, :, 0]).clamp_(max=0.999)
            o_coord[:, :, 3] = (o_coord[:, :, 3] + cur_point_sup[None, :, 1]).clamp_(max=0.999)

            # Convert from (x_min, y_min, x_max, y_max) → (center_x, center_y, w, h)
            o_coord = box_ops.box_xyxy_to_cxcywh(o_coord)
            outputs_coord.append(o_coord)  # [depth, num_points, 4]

        # Reorganize predictions by depth instead of by image
        outputs_class_depth = []
        outputs_coord_depth = []

        for dep_idx in range(depth):
            batched_cls = []
            batched_coord = []
            # Collect predictions for the same decoder layer across the batch
            for idx in range(bs):
                batched_cls.append(outputs_class[idx][dep_idx])
                batched_coord.append(outputs_coord[idx][dep_idx])
            outputs_class_depth.append(batched_cls)
            outputs_coord_depth.append(batched_coord)

        # Ground-truth labels per image (for loss computation)
        gt_label = []
        for i in range(bs):
            gt_label.append(points_supervision[i]["labels"].unsqueeze(0))

        # We only keep the last layer's box predictions for main loss
        out = {"pred_boxes": outputs_coord_depth[-1], "gt_label": gt_label}

        # Add auxiliary outputs from intermediate layers if enabled
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_coord_depth)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        """
        Prepare auxiliary outputs for intermediate decoder layers.

        TorchScript does not support dicts with mixed types (Tensor + list),
        so we repack them into a uniform structure.
        """
        # Use all layers except the last as auxiliary predictions
        return [{"pred_boxes": b} for b in outputs_coord[:-1]]


# ---------------------------------------------------------------------
# Post-processing for COCO-style evaluation
# ---------------------------------------------------------------------
class PostProcess(nn.Module):
    """ Convert model output into the format expected by the COCO API. """

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """
        Perform the computation.

        Args:
            outputs: dict from model:
                - "pred_boxes": list of [num_points, 4] in cx,cy,w,h (normalized)
                - "gt_label": list of label tensors per image
            target_sizes: [batch_size, 2] original image sizes (h, w)

        Returns:
            List[dict] with keys:
                - "boxes": tuple(tensor) of [num_points, 4] (absolute xyxy)
                - "labels": tuple(tensor) of labels (same order as boxes)
        """
        out_bbox = outputs["pred_boxes"]
        gt_label = outputs["gt_label"]

        assert target_sizes.shape[1] == 2
        bs = len(out_bbox)

        boxes = []

        # Convert all boxes from cxcywh(normalized) -> xyxy(normalized)
        for idx in range(bs):
            boxes.append(box_ops.box_cxcywh_to_xyxy(out_bbox[idx]))

        # Scale normalized boxes to absolute image coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        for idx in range(bs):
            boxes[idx] = boxes[idx] * scale_fct[idx][None]

        # Pack results in COCO-like format
        results = [
            {"boxes": (b,), "labels": (l,)}
            for b, l in zip(boxes, gt_label)
        ]

        return results


# ---------------------------------------------------------------------
# Simple multi-layer perceptron used for bounding box regression
# ---------------------------------------------------------------------
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN). """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        Args:
            input_dim: dimension of input features
            hidden_dim: hidden layer size
            output_dim: dimension of output
            num_layers: total number of linear layers
        """
        super().__init__()
        self.num_layers = num_layers

        # Hidden layers all have the same size
        h = [hidden_dim] * (num_layers - 1)

        # Create sequence of Linear layers
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        # Apply ReLU after every layer except the last
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ---------------------------------------------------------------------
# Backbone builder for Swin variants
# ---------------------------------------------------------------------
def build_swin_backbone(args):
    """Build Swin Transformer backbone with pretrained ImageNet weights."""
    from torchvision.models.swin_transformer import swin_t, swin_s, swin_b
    from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights

    # Select Swin variant and number of output channels
    if args.backbone == "swin_tiny":
        backbone_model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        num_channels = 768
    elif args.backbone == "swin_small":
        backbone_model = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
        num_channels = 768
    elif args.backbone == "swin_base":
        backbone_model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        num_channels = 1024
    else:
        raise ValueError(f"Unknown Swin backbone: {args.backbone}")

    # Create positional encoding for DETR
    position_embedding = build_position_encoding(args)

    # Wrap plain torchvision model so it accepts NestedTensor and returns (features, pos)
    joiner = TVSwinJoiner(backbone_model, position_embedding)
    joiner.num_channels = num_channels

    return joiner


# ---------------------------------------------------------------------
# Model / criterion / postprocessor builder
# ---------------------------------------------------------------------
def build(args):
    """
    Build the full model, criterion, and postprocessors given config args.
    """

    # Default number of classes depends on dataset
    num_classes = 3 if args.dataset_file != "coco" else 16  # CXR: 14+2, RSNA: 1+2

    if args.dataset_file == "cxr8":
        num_classes = 10  # 8+2
    if args.dataset_file == "coco_panoptic":
        num_classes = 250

    device = torch.device(args.device)

    # Choose Swin backbone or fallback backbone
    if args.backbone.startswith("swin_"):
        backbone = build_swin_backbone(args)
    else:
        backbone = build_backbone(args)

    # Build transformer encoder-decoder
    transformer = build_transformer(args)

    # Build encoders for labels and points (used to form queries)
    label_encoder = build_label_encoder(args.hidden_dim, num_classes)
    point_encoder = build_point_encoder()

    # Create DETR model
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        label_encoder=label_encoder,
        point_encoder=point_encoder,
    )

    # Hungarian matcher for assignment
    matcher = build_matcher(args)

    # Base loss weights
    weight_dict = {
        "loss_ce": 1,
        "loss_bbox": args.bbox_loss_coef,
    }
    weight_dict["loss_giou"] = args.giou_loss_coef

    # Consistency losses for semi-supervised training
    if args.cons_loss:
        weight_dict["loss_cons"] = args.cons_loss_coef
    if args.train_with_unlabel_imgs:
        weight_dict["loss_unlabelcons"] = args.unlabel_cons_loss_coef

    # Mask-related losses if segmentation is enabled
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    # If using auxiliary losses, replicate weights for each decoder layer (except last)
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in weight_dict.items()}
            )
        weight_dict.update(aux_weight_dict)

    # Define which loss components to use
    losses = ["boxes"]
    if args.cons_loss:
        losses += ["consistency"]
    if args.masks:
        losses += ["masks"]

    # Criterion that handles point-based supervision and DETR losses
    criterion = PointCriterion(
        num_classes,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        args=args,
    )
    criterion.to(device)

    # Postprocessors: only bounding boxes here
    postprocessors = {"bbox": PostProcess()}

    return model, criterion, postprocessors
