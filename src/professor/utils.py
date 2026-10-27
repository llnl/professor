# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import os
import re
import torch
from typing import Optional, Union, Dict, Any, Sequence, Tuple


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


def last_checkpoint(run_directory: str) -> Tuple[str, int]:
    """
    Finds the latest checkpoint that prof-trainer wrote to a run directory.

    Checkpoints are named by their zero-padded epoch number with a ``.pt`` suffix (``0042.pt``),
    or bare (``0042``) for runs written by versions between the 3D generators and this fix,
    so a run started with either version can be continued.

    Args:
        run_directory (str): Run directory, with or without a trailing slash

    Returns:
        Tuple[str, int]: Path of the latest checkpoint and its epoch number, or ("", -1)
        if the directory does not exist or holds no checkpoint
    """
    if not os.path.isdir(run_directory):
        return "", -1
    found = []
    for name in os.listdir(run_directory):
        match = re.fullmatch(r"(\d+)(\.pt)?", name)
        if match and os.path.isfile(os.path.join(run_directory, name)):
            found.append((int(match.group(1)), name))
    if not found:
        return "", -1
    epoch, name = max(found)
    return os.path.join(run_directory, name), epoch


def lr_schedule_factor(
    step: int,
    total_steps: int,
    schedule: str = "constant",
    warmup_steps: int = 0,
    lr_min: float = 0.01,
) -> float:
    """
    Learning rate of optimizer step `step` (counted from 0) as a fraction of the base learning rate.

    A linear warm-up over the first warmup_steps steps, then either the full learning rate
    ("constant") or a cosine decay to lr_min times the full learning rate at total_steps ("cosine").
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    if schedule == "cosine":
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return lr_min + (1.0 - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unrecognized learning rate schedule: {schedule}")
