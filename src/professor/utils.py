# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
import numpy as np
import torch
from numpy.typing import NDArray
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


def _combine_moments(
    a: Tuple[NDArray[Any], NDArray[Any], NDArray[Any]], b: Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Combine (count, mean, sum of squared deviations) of two sets of values (Chan et al.)"""
    n_a, mean_a, m2_a = a
    n_b, mean_b, m2_b = b
    n = n_a + n_b
    delta = mean_b - mean_a
    return n, mean_a + delta * n_b / n, m2_a + m2_b + delta**2 * n_a * n_b / n


def scaling_statistics(
    dataset: Any,
    n_samples: int,
    log_inputs: Sequence[int] = (),
    comm: Any = None,
) -> dict[str, NDArray[np.float64]]:
    """
    Per-channel mean and standard deviation of the fields and per-input range of the inputs
    (after taking the log of the inputs in log_inputs) over n_samples evenly spaced samples of
    the dataset. With an MPI communicator, the ranks share the reading and get the same result.
    """
    n = len(dataset)
    indices = np.unique(np.linspace(0, n - 1, max(1, min(n_samples, n))).round().astype(int))
    rank, size = (0, 1) if comm is None else (comm.rank, comm.size)

    moments = None
    input_min = None
    input_max = None
    for i in indices[rank::size]:
        item = dataset[int(i)]
        inputs = item[0].double().flatten().numpy()
        if len(log_inputs):
            log_inputs = list(log_inputs)
            if np.any(inputs[log_inputs] <= 0.0):
                raise ValueError(f"Inputs {log_inputs} must be positive to take their log, sample {i} has {inputs}")
            inputs[log_inputs] = np.log(inputs[log_inputs])
        fields = item[1].double().flatten(1).numpy()
        sample = (
            np.full(fields.shape[0], float(fields.shape[1])),
            fields.mean(axis=1),
            ((fields - fields.mean(axis=1, keepdims=True)) ** 2).sum(axis=1),
        )
        moments = sample if moments is None else _combine_moments(moments, sample)
        input_min = inputs if input_min is None else np.minimum(input_min, inputs)
        input_max = inputs if input_max is None else np.maximum(input_max, inputs)

    parts = [(moments, input_min, input_max)]
    if comm is not None:
        parts = comm.allgather(parts[0])
    parts = [p for p in parts if p[0] is not None]
    moments = parts[0][0]
    for p in parts[1:]:
        moments = _combine_moments(moments, p[0])
    count, mean, m2 = moments
    return {
        "field_mean": mean,
        "field_std": np.sqrt(m2 / count),
        "input_min": np.min([p[1] for p in parts], axis=0),
        "input_max": np.max([p[2] for p in parts], axis=0),
    }
