# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from torch import nn
from typing import Optional, Any, cast


class AlphaLinear(nn.Module):
    def __init__(self, n_channels: int = 1, n_dims: int = 2) -> None:
        super(AlphaLinear, self).__init__()
        tensor_size = [1, n_channels] + [1 for _ in range(n_dims)]

        self.alpha: nn.Parameter = nn.Parameter(
            torch.ones(*tensor_size),
            requires_grad=True,
        )
        self.beta: nn.Parameter = nn.Parameter(
            torch.ones(*tensor_size),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + self.beta


class ModTanh(nn.Module):
    def __init__(
        self, n_channels: int, d_max: Optional[torch.Tensor] = None, d_min: Optional[torch.Tensor] = None
    ) -> None:
        super(ModTanh, self).__init__()
        self.n_channels = n_channels
        if d_max is None:
            d_max = torch.zeros(n_channels)
        if d_min is None:
            d_min = torch.zeros(n_channels)
        self.register_buffer("d_max", d_max)
        self.register_buffer("d_min", d_min)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_range = 2.0
        out = torch.zeros_like(x)
        d_max = cast(torch.Tensor, self.d_max)
        d_min = cast(torch.Tensor, self.d_min)

        for i in range(self.n_channels):
            scale_ = feature_range / (d_max[i] - d_min[i])  # noqa E:501
            min_ = -1.0 - scale_ * d_min[i]
            out[:, i] = (torch.tanh(x[:, i]) - min_) / scale_
        return out


class ModAlphaTanh(nn.Module):
    def __init__(
        self, n_channels: int, d_max: Optional[torch.Tensor] = None, d_min: Optional[torch.Tensor] = None
    ) -> None:
        super(ModAlphaTanh, self).__init__()
        self.n_channels: int = n_channels
        if d_max is None:
            d_max = torch.zeros(n_channels)
        if d_min is None:
            d_min = torch.zeros(n_channels)
        self.register_buffer("d_max", d_max)
        self.register_buffer("d_min", d_min)
        self.alpha: nn.Parameter = nn.Parameter(
            torch.ones(n_channels),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_range = 2.0
        out = torch.zeros_like(x)
        d_max = cast(torch.Tensor, self.d_max)
        d_min = cast(torch.Tensor, self.d_min)

        for i in range(self.n_channels):
            xa = self.alpha[i] * x[:, i]
            scale_ = feature_range / (d_max[i] - d_min[i])  # noqa E:501
            min_ = -1.0 - scale_ * d_min[i]
            out[:, i] = (torch.tanh(xa) - min_) / scale_
        return out


class ModAlphaBetaTanh(nn.Module):
    def __init__(
        self, n_channels: int, d_max: Optional[torch.Tensor] = None, d_min: Optional[torch.Tensor] = None
    ) -> None:
        super(ModAlphaBetaTanh, self).__init__()
        self.n_channels: int = n_channels
        if d_max is None:
            d_max = torch.zeros(n_channels)
        if d_min is None:
            d_min = torch.zeros(n_channels)
        self.register_buffer("d_max", d_max)
        self.register_buffer("d_min", d_min)
        self.alpha: nn.Parameter = nn.Parameter(
            torch.ones(n_channels),
            requires_grad=True,
        )
        self.beta: nn.Parameter = nn.Parameter(
            torch.zeros(n_channels),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_range = 2.0
        out = torch.zeros_like(x)
        d_max = cast(torch.Tensor, self.d_max)
        d_min = cast(torch.Tensor, self.d_min)

        for i in range(self.n_channels):
            xa = self.alpha[i] * x[:, i] + self.beta[i]
            scale_ = feature_range / (d_max[i] - d_min[i])  # noqa E:501
            min_ = -1.0 - scale_ * d_min[i]
            out[:, i] = (torch.tanh(xa) - min_) / scale_
        return out


class ModIdentity(nn.Module):
    def __init__(
        self, n_channels: int, d_max: Optional[torch.Tensor] = None, d_min: Optional[torch.Tensor] = None
    ) -> None:
        super(ModIdentity, self).__init__()
        self.n_channels = n_channels
        if d_max is None:
            d_max = torch.zeros(n_channels)
        if d_min is None:
            d_min = torch.zeros(n_channels)
        self.register_buffer("d_max", d_max)
        self.register_buffer("d_min", d_min)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        d_max = cast(torch.Tensor, self.d_max)
        d_min = cast(torch.Tensor, self.d_min)

        for i in range(self.n_channels):
            out[:, i] = (x[:, i] - d_min[i]) / (1.0 / (d_max[i] - d_min[i]))  # noqa E:501
        return out


class InputIdentity(nn.Module):
    def __init__(self, n_input: int, myXScaler: Any) -> None:
        super(InputIdentity, self).__init__()
        self.n_input = n_input
        in_min_ = torch.from_numpy(myXScaler.min_)
        in_scale_ = torch.from_numpy(myXScaler.scale_)
        in_min_ = in_min_.float()
        in_scale_ = in_scale_.float()
        self.register_buffer("in_min", in_min_)
        self.register_buffer("in_scale", in_scale_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_scale = cast(torch.Tensor, self.in_scale)
        in_min = cast(torch.Tensor, self.in_min)
        t1 = x[:, :, 0, 0] * in_scale
        t2 = t1 + in_min
        return t2.view(-1, self.n_input, 1, 1)


class ModReLU(nn.Module):
    def __init__(
        self, n_channels: int, d_max: Optional[torch.Tensor] = None, d_min: Optional[torch.Tensor] = None
    ) -> None:
        super(ModReLU, self).__init__()
        self.n_channels = n_channels
        if d_max is None:
            d_max = torch.zeros(n_channels)
        if d_min is None:
            d_min = torch.zeros(n_channels)
        self.register_buffer("d_max", d_max)
        self.register_buffer("d_min", d_min)
        self.ReLU: nn.ReLU = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        d_max = cast(torch.Tensor, self.d_max)
        d_min = cast(torch.Tensor, self.d_min)

        for i in range(self.n_channels):
            out[:, i] = (self.ReLU(x[:, i]) - d_min[i]) / (1.0 / (d_max[i] - d_min[i]))  # noqa E501
        return out


class UpscaleBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        upscale_factor: int = 2,
        bias: bool = False,
        batch_norm: bool = True,
        upscale_type: str = "transpose",
        activation_function: nn.Module | None = None,
    ):
        super(UpscaleBlock2D, self).__init__()
        layers = []

        if upscale_type == "transpose":
            padding = (kernel_size - upscale_factor) // 2
            layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, upscale_factor, padding, bias=bias)
            )
        elif upscale_type in ["linear", "nearest"]:
            interp_kwargs = {}
            if "linear" in upscale_type:
                upscale_type = "bilinear"
                interp_kwargs["align_corners"] = True

            layers.append(nn.Upsample(scale_factor=upscale_factor, mode=upscale_type, **interp_kwargs))
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding="same", bias=bias))
        else:
            raise Exception(f"Unrecognized layer upscale type: {upscale_type}")

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation_function is not None:
            layers.append(activation_function)

        self.upscaler: nn.Sequential = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.upscaler(input)


class UpscaleBlock3D(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 4,
        upscale_factor: int = 2,
        bias: bool = False,
        batch_norm: bool = True,
        upscale_type: str = "transpose",
        activation_function: nn.Module | None = None,
    ):
        super(UpscaleBlock3D, self).__init__()

        layers = []
        if upscale_type == "transpose":
            padding = (kernel_size - upscale_factor) // 2
            layers.append(
                nn.ConvTranspose3d(in_features, out_features, kernel_size, upscale_factor, padding, bias=bias)
            )

        elif upscale_type in ["linear", "nearest"]:
            interp_kwargs = {}
            if "linear" in upscale_type:
                upscale_type = "trilinear"
                interp_kwargs["align_corners"] = True

            layers.append(nn.Upsample(scale_factor=upscale_factor, mode=upscale_type, **interp_kwargs))
            layers.append(nn.Conv3d(in_features, out_features, kernel_size, stride=1, padding="same", bias=bias))
        else:
            raise Exception(f"Unrecognized layer upscale type: {upscale_type}")

        if batch_norm:
            layers.append(nn.BatchNorm3d(out_features))

        if activation_function is not None:
            layers.append(activation_function)

        self.upscaler: nn.Sequential = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.upscaler(input)


class UpscaleBlock3DSpectral(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_modes: tuple[int, int, int] = (8, 8, 8),
        upscale_factor: int = 2,
        bias: bool = False,
        batch_norm: bool = True,
        upscale_type: str = "transpose",
        activation_function: nn.Module | None = None,
    ):
        from neuralop.layers.spectral_convolution import SpectralConv  # type: ignore

        super(UpscaleBlock3DSpectral, self).__init__()

        layers = []
        if upscale_type == "transpose":
            layers.append(
                SpectralConv(in_features, out_features, num_modes, resolution_scaling_factor=upscale_factor, bias=bias)
            )
        elif upscale_type in ["linear", "nearest"]:
            interp_kwargs = {}
            if "linear" in upscale_type:
                upscale_type = "trilinear"
                interp_kwargs["align_corners"] = True

            layers.append(nn.Upsample(scale_factor=upscale_factor, mode=upscale_type, **interp_kwargs))
            layers.append(SpectralConv(in_features, out_features, num_modes, bias=bias))

        if batch_norm:
            layers.append(nn.BatchNorm3d(out_features))

        if activation_function is not None:
            layers.append(activation_function)

        self.upscaler: nn.Sequential = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.upscaler(input)
