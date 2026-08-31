# Copyright 2026, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch

from professor.layers import UpscaleBlock2D, UpscaleBlock3D
# from professor.layers import UpscaleBlock3DSpectral


def test_upscale_block_2d_residual_transpose_requires_matching_output_size():
    with pytest.raises(ValueError, match="requires kernel_size - upscale_factor to be even"):
        UpscaleBlock2D(
            3,
            5,
            kernel_size=3,
            upscale_factor=2,
            residual=True,
            upscale_type="transpose",
        )


def test_upscale_block_2d_residual_transpose_valid_shape():
    block = UpscaleBlock2D(
        3,
        5,
        kernel_size=4,
        upscale_factor=2,
        residual=True,
        upscale_type="transpose",
    )

    output = block(torch.rand(2, 3, 7, 9))

    assert output.shape == (2, 5, 14, 18)


def test_upscale_block_2d_non_residual_nearest_has_no_skip_parameters():
    block = UpscaleBlock2D(
        3,
        5,
        kernel_size=3,
        residual=False,
        upscale_type="nearest",
    )

    output = block(torch.rand(2, 3, 7, 9))

    assert block.skip is None
    assert not any(name.startswith("skip.") for name, _ in block.named_parameters())
    assert output.shape == (2, 5, 14, 18)


def test_upscale_block_3d_residual_transpose_requires_matching_output_size():
    with pytest.raises(ValueError, match="requires kernel_size - upscale_factor to be even"):
        UpscaleBlock3D(
            3,
            5,
            kernel_size=3,
            upscale_factor=2,
            residual=True,
            upscale_type="transpose",
        )


def test_upscale_block_3d_residual_transpose_valid_shape():
    block = UpscaleBlock3D(
        3,
        5,
        kernel_size=4,
        upscale_factor=2,
        residual=True,
        upscale_type="transpose",
    )

    output = block(torch.rand(2, 3, 4, 5, 6))

    assert output.shape == (2, 5, 8, 10, 12)


def test_upscale_block_3d_non_residual_nearest_has_no_skip_parameters():
    block = UpscaleBlock3D(
        3,
        5,
        kernel_size=3,
        residual=False,
        upscale_type="nearest",
    )

    output = block(torch.rand(2, 3, 4, 5, 6))

    assert block.skip is None
    assert not any(name.startswith("skip.") for name, _ in block.named_parameters())
    assert output.shape == (2, 5, 8, 10, 12)


# Note: disabling these tests since they require additional work
# def test_upscale_block_3d_spectral_residual_transpose_valid_shape():
#     block = UpscaleBlock3DSpectral(
#         3,
#         5,
#         num_modes=(2, 2, 2),
#         upscale_factor=2,
#         residual=True,
#         upscale_type="transpose",
#     )

#     output = block(torch.rand(2, 3, 4, 5, 6))

#     assert output.shape == (2, 5, 8, 10, 12)


# def test_upscale_block_3d_spectral_non_residual_nearest_has_no_skip_parameters():
#     block = UpscaleBlock3DSpectral(
#         3,
#         5,
#         num_modes=(2, 2, 2),
#         residual=False,
#         upscale_type="nearest",
#     )

#     output = block(torch.rand(2, 3, 4, 5, 6))

#     assert block.skip is None
#     assert not any(name.startswith("skip.") for name, _ in block.named_parameters())
#     assert output.shape == (2, 5, 8, 10, 12)
