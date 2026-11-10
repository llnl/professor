# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
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


def neighbour_l1(error: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute difference of an error field between adjacent pixels, summed over the spatial axes.

    Penalizes error that changes from one pixel to the next (speckle, misplaced edges) and leaves
    smooth error alone.

    Args:
        error (torch.Tensor): Error pred - target, shape (n, channels, y, x) or (n, channels, y, x, z)

    Returns:
        torch.Tensor: mean |d_y error| + mean |d_x error| (+ mean |d_z error|)
    """
    loss = torch.zeros((), device=error.device, dtype=error.dtype)
    for dim in range(2, error.dim()):
        if error.shape[dim] > 1:
            loss = loss + torch.mean(torch.abs(torch.diff(error, dim=dim)))
    return loss


def pyramid_l1(error: torch.Tensor, levels: int = 5) -> torch.Tensor:
    """
    L1 norm of an error field split into a Laplacian pyramid, finest band weighted most.

    With G_0 = error and G_{l+1} the 2x2 (2x2x2 in 3D) average of G_l, the bands are
    B_l = G_l - up(G_{l+1}), with up() nearest-neighbour upsampling, and the loss is
    sum_l 2^-l mean |B_l| + 2^-L mean |G_L|. Each octave of the error then enters the loss on its
    own, so the few fine-scale bands are not swamped by the coarse error. Fields with a side
    shorter than 2^levels use fewer levels.

    Args:
        error (torch.Tensor): Error pred - target, shape (n, channels, y, x) or (n, channels, y, x, z)
        levels (int): Number of band-pass levels L

    Returns:
        torch.Tensor: The pyramid loss
    """
    if error.dim() not in (4, 5):
        raise ValueError(f"pyramid_l1 needs 2D or 3D fields, got shape {tuple(error.shape)}")
    avg_pool = F.avg_pool2d if error.dim() == 4 else F.avg_pool3d
    levels = min(levels, min(error.shape[2:]).bit_length() - 1)
    loss = torch.zeros((), device=error.device, dtype=error.dtype)
    g = error
    for level in range(levels):
        down = avg_pool(g, 2)
        band = g - F.interpolate(down, size=g.shape[2:], mode="nearest")
        loss = loss + 2.0**-level * torch.mean(torch.abs(band))
        g = down
    return loss + 2.0**-levels * torch.mean(torch.abs(g))


def field_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_target: str = "l1",
    grad_weight: float = 0.0,
    pyr_weight: float = 0.0,
    pyr_levels: int = 5,
) -> torch.Tensor:
    """
    Training loss for full fields: a pointwise term plus optional pixel-difference and pyramid terms.

        loss = pointwise(e) + grad_weight * neighbour_l1(e) + pyr_weight * pyramid_l1(e, pyr_levels)

    with e = pred - target and pointwise the mean absolute ('l1') or squared (anything else) error.
    With the default weights of 0 this is the plain pointwise loss. With 'l1' every term is linear
    in |e|, so the relative weights do not depend on the scale of the fields.

    Args:
        pred (torch.Tensor): Prediction, shape (n, channels, y, x) or (n, channels, y, x, z)
        target (torch.Tensor): Target, same shape
        loss_target (str): 'l1' for mean absolute error, anything else for mean squared error
        grad_weight (float): Weight of the neighbouring-pixel difference term
        pyr_weight (float): Weight of the Laplacian-pyramid term
        pyr_levels (int): Band-pass levels of the pyramid

    Returns:
        torch.Tensor: The loss
    """
    error = pred - target
    loss = torch.mean(torch.abs(error)) if loss_target == "l1" else torch.mean(torch.square(error))
    if grad_weight:
        loss = loss + grad_weight * neighbour_l1(error)
    if pyr_weight:
        loss = loss + pyr_weight * pyramid_l1(error, pyr_levels)
    return loss


def forward_jacobian_vector_product(
    model: torch.nn.Module, x: torch.Tensor, tangent: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Model output and its Jacobian-vector product (JVP) with tangent, i.e. the directional
    derivative of the output along tangent, from one forward-mode automatic differentiation pass.

    Args:
        model (torch.nn.Module): The model
        x (torch.Tensor): Model input
        tangent (torch.Tensor): Direction in input space, same shape as x

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: model(x) and d model(x + s * tangent) / ds at s = 0
    """
    with fwAD.dual_level():
        dual_output = model(fwAD.make_dual(x, tangent))
        output, output_derivative = fwAD.unpack_dual(dual_output)
    if output_derivative is None:  # output independent of the input
        output_derivative = torch.zeros_like(output)
    return output, output_derivative


def sobolev_loss(output_derivative: torch.Tensor, sensitivities: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error of the model's directional derivative against the reference sensitivities.

    With the tangent u = direction along the first k inputs, the reference directional derivative
    is sum_i u_i sensitivities_i. Samples whose sensitivities are not all finite are left out. The
    error is divided by k, so with u_i ~ N(0, 1 / rms_i^2) the loss estimates the mean over inputs
    of the derivative mse / rms_i^2 at the cost of one Jacobian-vector product.

    Args:
        output_derivative (torch.Tensor): Directional derivative of the model, shape (n, channels, y, x[, z])
        sensitivities (torch.Tensor): d fields / d inputs, shape (n, k, channels, y, x[, z])
        direction (torch.Tensor): Tangent u along the first k inputs, shape (n, k)

    Returns:
        torch.Tensor: The loss
    """
    is_valid = torch.isfinite(sensitivities).flatten(1).all(1)
    sensitivities = torch.where(
        is_valid.view(-1, *([1] * (sensitivities.dim() - 1))), sensitivities, torch.zeros_like(sensitivities)
    )
    reference_derivative = torch.einsum("nk,nk...->n...", direction.to(sensitivities.dtype), sensitivities)
    derivative_error = output_derivative - reference_derivative
    mse_per_sample = torch.mean(torch.square(derivative_error.flatten(1)), 1)
    return torch.sum(is_valid * mse_per_sample) / is_valid.sum().clamp(min=1) / sensitivities.shape[1]
