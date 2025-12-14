# Copyright (c) Facebook, Inc. and its affiliates.
# ------------------------------------------------------------------------
"""
Implements the Hungarian Matcher used in DETR

The matcher finds a 1-to-1 assignment between:
    - model predictions (queries)
    - ground-truth objects

It constructs a cost matrix and solves the Linear Sum Assignment Problem (LSAP)
using the Hungarian algorithm.
"""

# -----------------------------
# Imports
# -----------------------------
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


# ------------------------------------------------------------------------
# Hungarian Matcher
# ------------------------------------------------------------------------
class HungarianMatcher(nn.Module):
    """
    Computes the optimal matching between predictions and ground-truth targets.

    - Each ground-truth object is matched to exactly one prediction.
    - Extra predictions remain unmatched (treated as "no-object").
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1
    ):
        """
        Initialize matching cost weights.

        Args:
            cost_class (float): weight for classification error cost
            cost_bbox (float): weight for L1 bounding box distance cost
            cost_giou (float): weight for generalized IoU cost
        """
        super().__init__()

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        # At least one cost must be active
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "All matching costs cannot be 0"


    # ------------------------------------------------------------------
    # Forward: Perform Hungarian matching
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Performs bipartite matching between predictions and targets.

        Args:
            outputs (dict):
                "pred_logits": [batch_size, num_queries, num_classes]
                "pred_boxes" : [batch_size, num_queries, 4] (cx, cy, w, h)

            targets (list of dicts):
                Each dict contains:
                    "labels": [num_objects]
                    "boxes" : [num_objects, 4]

        Returns:
            list of tuples per batch element:
                (index_i, index_j)

            where:
                index_i : indices of matched predictions
                index_j : corresponding indices of matched targets
        """

        # -----------------------------
        # Get batch size & num queries
        # -----------------------------
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # -----------------------------
        # Flatten predictions
        # -----------------------------
        # Convert logits to probabilities
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        # Shape: [batch_size * num_queries, num_classes]

        # Flatten predicted boxes
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        # Shape: [batch_size * num_queries, 4]


        # -----------------------------
        # Concatenate targets
        # -----------------------------
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])


        # -----------------------------
        # Compute cost components
        # -----------------------------

        # Classification cost:
        # cost = 1 - probability(predicted_class == target_class)
        # Implemented as negative probability for simplicity
        cost_class = -out_prob[:, tgt_ids]

        # Bounding box L1 distance cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Generalized IoU cost
        # Need to convert boxes from cxcywh → xyxy format
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )

        # -----------------------------
        # Final weighted cost matrix
        # -----------------------------
        C = (
            self.cost_bbox * cost_bbox +
            self.cost_class * cost_class +
            self.cost_giou * cost_giou
        )

        # Reshape to per-batch matrices
        C = C.view(bs, num_queries, -1).cpu()


        # -----------------------------
        # Solve Hungarian assignment
        # -----------------------------
        # Number of targets per batch element
        sizes = [len(v["boxes"]) for v in targets]

        # Apply Hungarian algorithm separately to each batch element
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]

        # Return indices as tensors
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64)
            )
            for i, j in indices
        ]


# ------------------------------------------------------------------------
# Factory method for matcher
# ------------------------------------------------------------------------
def build_matcher(args):
    """
    Create HungarianMatcher using command-line arguments.
    """
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou
    )
