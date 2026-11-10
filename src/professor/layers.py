# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy

import torch
from torch import nn
from typing import Optional, Any, Sequence, cast


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


class ScaledAlphaLinear(AlphaLinear):
    """
    AlphaLinear followed by a fixed per-channel affine map, (alpha * x + beta) * scale + offset.

    The network before this layer works on fields of order 1, while the model output keeps
    the physical units of the dataset. offset and scale are saved with the model as buffers.
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_dims: int = 2,
        offset: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
    ) -> None:
        super(ScaledAlphaLinear, self).__init__(n_channels=n_channels, n_dims=n_dims)
        tensor_size = [1, n_channels] + [1 for _ in range(n_dims)]
        offset = [0.0] * n_channels if offset is None else [float(v) for v in offset]
        scale = [1.0] * n_channels if scale is None else [float(v) for v in scale]
        if len(offset) != n_channels or len(scale) != n_channels:
            raise ValueError(f"offset and scale need {n_channels} values, got {len(offset)} and {len(scale)}")
        self.register_buffer("offset", torch.tensor(offset).view(*tensor_size))
        self.register_buffer("scale", torch.tensor(scale).view(*tensor_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.alpha * x + self.beta) * self.scale + self.offset


class InputScaling(nn.Module):
    """
    Fixed per-input map to order 1: x_i -> (f_i(x_i) - offset_i) / scale_i, where f_i is log
    for the inputs listed in log_inputs and the identity otherwise.

    It is the first layer of a model that takes the raw inputs of the dataset, of shape
    (batch, n_inputs, 1, ...). offset, scale and the log mask are saved with the model as buffers.
    """

    def __init__(
        self,
        offset: Sequence[float],
        scale: Sequence[float],
        log_inputs: Optional[Sequence[int]] = None,
    ) -> None:
        super(InputScaling, self).__init__()
        offset = [float(v) for v in offset]
        scale = [float(v) for v in scale]
        if len(offset) != len(scale):
            raise ValueError(f"offset and scale need the same length, got {len(offset)} and {len(scale)}")
        log_mask = [False] * len(offset)
        for i in [] if log_inputs is None else log_inputs:
            log_mask[int(i)] = True
        self.register_buffer("offset", torch.tensor(offset).view(1, -1, 1))
        self.register_buffer("scale", torch.tensor(scale).view(1, -1, 1))
        self.register_buffer("log_mask", torch.tensor(log_mask).view(1, -1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        log_mask = cast(torch.Tensor, self.log_mask)
        # log only where flagged; the where() keeps the gradient finite for the other inputs
        log_x = torch.log(torch.where(log_mask, x, torch.ones_like(x)))
        x = torch.where(log_mask, log_x, x)
        x = (x - cast(torch.Tensor, self.offset)) / cast(torch.Tensor, self.scale)
        return x.reshape(shape)


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
        residual: bool = False,
        bias: bool = False,
        batch_norm: bool = True,
        upscale_type: str = "transpose",
        activation_function: nn.Module | None = None,
    ):
        super().__init__()
        self._residual: bool = residual

        # Check upscaling options
        conv_upscale = upscale_type == "transpose"
        if self._residual and conv_upscale and (kernel_size - upscale_factor) % 2 != 0:
            raise ValueError(
                "UpscaleBlock2D with residual=True and upscale_type='transpose' "
                "requires kernel_size - upscale_factor to be even."
            )

        interp_kwargs = {}
        if "linear" in upscale_type:
            upscale_type = "bilinear"
            interp_kwargs["align_corners"] = True
        if self._residual and conv_upscale:
            upscale_type = "bilinear"

        # Build the upsampler
        self.skip: nn.Module | None = None
        if upscale_type not in ["bilinear", "nearest", "transpose"]:
            raise ValueError(f"Unrecognized layer upscale type: {upscale_type}")
        if self._residual and upscale_type in ["bilinear", "nearest"]:
            self.skip = nn.Sequential(
                nn.Upsample(scale_factor=upscale_factor, mode=upscale_type, **interp_kwargs),
                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding="same", bias=False),
            )

        self.main: nn.Module
        if conv_upscale:
            padding = (kernel_size - upscale_factor) // 2
            self.main = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, upscale_factor, padding, bias=bias)

        else:
            self.main = nn.Sequential(
                nn.Upsample(scale_factor=upscale_factor, mode=upscale_type, **interp_kwargs),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding="same", bias=bias),
            )

        self.norm: nn.Module
        if batch_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()

        self.activation: nn.Module
        if activation_function is not None:
            self.activation = activation_function
        else:
            self.activation = nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._residual:
            s = self.skip(input)  # type: ignore
            x = self.main(input)
            y = self.norm(x)
            z = s + y
            return self.activation(z)

        else:
            x = self.main(input)
            y = self.norm(x)
            return self.activation(y)


class AddCoords(nn.Module):
    """Append normalized (y, x) pixel coordinates in [-1, 1] as two extra channels."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        y = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        grid = torch.stack(torch.meshgrid(y, xx, indexing="ij"))
        return torch.cat([x, grid.expand(n, -1, -1, -1)], 1)


class ResBlock(nn.Module):
    """Pre-activation residual block: x + conv(act(conv(act(x)))), 3x3 convolutions.

    Each activation is a copy of activation_function (ReLU if None). The second convolution
    (or its batch norm scale) starts at zero, so the block is the identity at initialization
    and stacking blocks does not change the starting model.
    """

    def __init__(self, features: int, activation_function: nn.Module | None = None, batch_norm: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(2):
            act = copy.deepcopy(activation_function) if activation_function is not None else nn.ReLU()
            if hasattr(act, "inplace"):
                act.inplace = False  # never overwrite the block input that the skip adds back
            layers += [act, nn.Conv2d(features, features, kernel_size=3, padding="same", bias=not batch_norm)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(features))
        self.body: nn.Sequential = nn.Sequential(*layers)
        last = self.body[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class UpscaleBlock3D(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 4,
        upscale_factor: int = 2,
        residual: bool = False,
        bias: bool = False,
        batch_norm: bool = True,
        upscale_type: str = "transpose",
        separable_conv: bool = False,
        activation_function: nn.Module | None = None,
    ):
        super().__init__()
        self._residual: bool = residual

        # Check upscaling options
        conv_upscale = upscale_type == "transpose"
        if self._residual and conv_upscale and (kernel_size - upscale_factor) % 2 != 0:
            raise ValueError(
                "UpscaleBlock3D with residual=True and upscale_type='transpose' "
                "requires kernel_size - upscale_factor to be even."
            )

        interp_kwargs = {}
        if "linear" in upscale_type:
            upscale_type = "trilinear"
            interp_kwargs["align_corners"] = True
        if self._residual and conv_upscale:
            upscale_type = "trilinear"

        # Build the upsampler
        self.skip: nn.Module | None = None
        if upscale_type not in ["trilinear", "nearest", "transpose"]:
            raise ValueError(f"Unrecognized layer upscale type: {upscale_type}")
        if self._residual and upscale_type in ["trilinear", "nearest"]:
            self.skip = nn.Sequential(
                nn.Upsample(scale_factor=upscale_factor, mode=upscale_type, **interp_kwargs),
                nn.Conv3d(in_features, out_features, 1, stride=1, padding="same", bias=False),
            )

        # Build the main path
        layers = []
        if conv_upscale:
            padding = (kernel_size - upscale_factor) // 2
            if separable_conv:
                layers.append(
                    nn.ConvTranspose3d(
                        in_features,
                        in_features,
                        kernel_size=kernel_size,
                        stride=upscale_factor,
                        padding=padding,
                        bias=False,
                        groups=in_features,
                    )
                )
                layers.append(
                    nn.Conv3d(
                        in_features,
                        out_features,
                        kernel_size=1,
                        stride=1,
                        padding="same",
                        bias=bias,
                    )
                )
            else:
                layers.append(
                    nn.ConvTranspose3d(in_features, out_features, kernel_size, upscale_factor, padding, bias=bias)
                )

        else:
            layers.append(nn.Upsample(scale_factor=upscale_factor, mode=upscale_type, **interp_kwargs))
            if separable_conv:
                layers.append(
                    nn.Conv3d(
                        in_features,
                        in_features,
                        kernel_size=kernel_size,
                        stride=1,
                        padding="same",
                        bias=False,
                        groups=in_features,
                    )
                )
                layers.append(nn.Conv3d(in_features, out_features, kernel_size=1, stride=1, padding="same", bias=bias))
            else:
                layers.append(nn.Conv3d(in_features, out_features, kernel_size, stride=1, padding="same", bias=bias))

        self.main = nn.Sequential(*layers)

        # Build the norm and activation
        self.norm: nn.Module
        if batch_norm:
            self.norm = nn.BatchNorm3d(out_features)
        else:
            self.norm = nn.Identity()

        self.activation: nn.Module
        if activation_function is not None:
            self.activation = activation_function
        else:
            self.activation = nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._residual:
            s = self.skip(input)  # type: ignore
            x = self.main(input)
            y = self.norm(x)
            z = s + y
            return self.activation(z)

        else:
            x = self.main(input)
            y = self.norm(x)
            return self.activation(y)


class UpscaleBlock3DSpectral(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_modes: tuple[int, int, int] = (8, 8, 8),
        upscale_factor: int = 2,
        residual: bool = False,
        bias: bool = False,
        batch_norm: bool = True,
        upscale_type: str = "transpose",
        activation_function: nn.Module | None = None,
    ):
        # TODO: add this requirement to the pyroject.toml when this feature is ready
        from neuralop.layers.spectral_convolution import SpectralConv  # type: ignore

        super().__init__()
        self._residual: bool = residual

        # Check upscaling options
        conv_upscale = upscale_type == "transpose"

        interp_kwargs = {}
        if "linear" in upscale_type:
            upscale_type = "trilinear"
            interp_kwargs["align_corners"] = True
        if self._residual and conv_upscale:
            upscale_type = "trilinear"

        # Build the upsampler
        self.skip: nn.Module | None = None
        if upscale_type not in ["trilinear", "nearest", "transpose"]:
            raise ValueError(f"Unrecognized layer upscale type: {upscale_type}")
        if self._residual and upscale_type in ["trilinear", "nearest"]:
            self.skip = nn.Sequential(
                nn.Upsample(scale_factor=upscale_factor, mode=upscale_type, **interp_kwargs),
                nn.Conv3d(in_features, out_features, 1, stride=1, padding="same", bias=False),
            )

        self.main: nn.Module
        if conv_upscale:
            self.main = SpectralConv(
                in_features,
                out_features,
                num_modes,
                resolution_scaling_factor=upscale_factor,
                bias=bias,
            )
        else:
            layers = []
            layers.append(nn.Upsample(scale_factor=upscale_factor, mode=upscale_type, **interp_kwargs))
            layers.append(SpectralConv(in_features, out_features, num_modes, bias=bias))
            self.main = nn.Sequential(*layers)

        self.norm: nn.Module
        if batch_norm:
            self.norm = nn.BatchNorm3d(out_features)
        else:
            self.norm = nn.Identity()

        self.activation: nn.Module
        if activation_function is not None:
            self.activation = activation_function
        else:
            self.activation = nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._residual:
            s = self.skip(input)  # type: ignore
            x = self.main(input)
            y = self.norm(x)
            z = s + y
            return self.activation(z)

        else:
            x = self.main(input)
            y = self.norm(x)
            return self.activation(y)
