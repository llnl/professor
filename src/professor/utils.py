# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from typing import Optional, Union, Dict, Any, Sequence


def consolidated_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_types: Sequence[str] = ["l1", "l2", "linf"],
    weights: Optional[Sequence[float]] = None,
) -> Union[torch.Tensor, Dict[str, Any]]:
    """
    Calculates multiple loss functions in one go and returns them individually or weighted sum.
    Computes the difference between pred and target only once for efficiency.

    Args:
        pred (torch.Tensor): Prediction tensor
        target (torch.Tensor): Target tensor
        loss_types (list): List of loss types to compute. Options: "l1", "l2", "linf"
            "sum-per-channel"
        weights (list, optional): Weights for each loss type. If provided, returns
            weighted sum. If None, returns dictionary of losses.

    Returns:
        dict or torch.Tensor: Dictionary of losses or weighted sum of losses
    """
    losses = {}

    # Calculate difference only once
    diff = pred - target
    abs_diff = torch.abs(diff)
    sum_axes = [ii for ii in range(len(pred.shape)) if ii != 1]

    if "l1" in loss_types:
        # L1 loss is mean of absolute differences
        losses["l1"] = torch.mean(abs_diff)

    if "sum-per-channel" in loss_types:
        # L1 loss is mean of absolute differences
        losses["sum-per-channel"] = torch.sum(
            abs_diff,
            sum_axes,
        )

    if "l1-sum" in loss_types:
        if "sum-per-channel" not in loss_types:
            # L1 loss is mean of absolute differences
            losses["sum-per-channel"] = torch.sum(
                abs_diff,
                sum_axes,
            )
        losses["l1-sum"] = torch.sum(losses["sum-per-channel"])

    if "l2-sum" in loss_types:
        losses["l2-sum"] = torch.sum(torch.square(diff))

    if "l2" in loss_types:
        # L2 loss is mean of squared differences
        losses["l2"] = torch.mean(torch.square(diff))

    if "linf" in loss_types:
        # L-infinity norm is the maximum absolute difference
        losses["linf"] = torch.max(abs_diff)

    # If weights are provided, return weighted sum
    if weights is not None:
        assert len(weights) == len(loss_types), "Number of weights must match number of loss types"
        total_loss = torch.tensor(0.0)
        for i, loss_type in enumerate(loss_types):
            total_loss += weights[i] * losses[loss_type]
        return total_loss

    # Otherwise return dictionary of individual losses
    return losses
