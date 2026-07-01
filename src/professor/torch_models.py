# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Initially written by Tom Stitt stitt4@llnl.gov
# Heavily modified by Charles Jekel jekel1@llnl.gov
import torch
import torch.nn as nn
import numpy as np
from torchlayers.upsample import ConvPixelShuffle  # type: ignore[import-untyped]
from typing import cast, Dict, Iterable


class RB(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.square(x))


# this stuff is from the pytroch dcgan tutorial
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# initialize all weightto mean=0, stdev=0.2
def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != "ConvAutoencoder":
        nn.init.normal_(cast(torch.Tensor, m.weight.data), 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(cast(torch.Tensor, m.weight.data), 1.0, 0.02)
        nn.init.constant_(cast(torch.Tensor, m.bias.data), 0)


class GeneratorTwo(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_pixels: int,
        n_channels: int,
        max_features: int = 512,
        first_layer: nn.Module = nn.Identity(),
        last_layer: nn.Module = nn.Tanh(),
    ) -> None:
        super(GeneratorTwo, self).__init__()
        n_pixels_8 = n_pixels // 8
        max_features_2 = max_features // 2
        max_features_4 = max_features // 4
        layers = (
            first_layer,
            nn.Linear(input_size, n_pixels_8 * n_pixels_8 * max_features, bias=False),
            nn.BatchNorm1d(n_pixels_8 * n_pixels_8 * max_features),
            nn.LeakyReLU(),
            nn.Unflatten(1, (max_features, n_pixels_8, n_pixels_8)),
            nn.ConvTranspose2d(max_features, max_features_2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(max_features_2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(max_features_2, max_features_4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(max_features_4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(max_features_4, n_channels, 4, stride=2, padding=1, bias=False),
            last_layer,
        )
        self.main: nn.Sequential = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)


class GeneratorOO(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_pixels: int,
        n_channels: int,
        max_features: int = 512,
        first_layer: nn.Module = nn.Identity(),
        last_layer: nn.Module = nn.Tanh(),
        activation: nn.Module = nn.LeakyReLU(),
    ) -> None:
        super(GeneratorOO, self).__init__()
        self.n_pixels_8: int = n_pixels // 8
        self.max_features_2: int = max_features // 2
        self.max_features_4: int = max_features // 4
        self.max_features: int = max_features
        self.n_pixels: int = n_pixels
        self.n_channels: int = n_channels
        self.first_layer: nn.Module = first_layer
        self.last_layer: nn.Module = last_layer
        self.activation: nn.Module = activation
        self.lin1: nn.Linear = nn.Linear(input_size, self.n_pixels_8 * self.n_pixels_8 * max_features, bias=False)
        self.bn1: nn.BatchNorm1d = nn.BatchNorm1d(self.n_pixels_8 * self.n_pixels_8 * max_features)
        self.conv1: nn.ConvTranspose2d = nn.ConvTranspose2d(
            max_features, self.max_features_2, 4, stride=2, padding=1, bias=False
        )
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(self.max_features_2)
        self.conv2: nn.ConvTranspose2d = nn.ConvTranspose2d(
            self.max_features_2, self.max_features_4, 4, stride=2, padding=1, bias=False
        )
        self.bn3: nn.BatchNorm2d = nn.BatchNorm2d(self.max_features_4)
        self.conv3: nn.ConvTranspose2d = nn.ConvTranspose2d(
            self.max_features_4, n_channels, 4, stride=2, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        x = self.activation(self.bn1(self.lin1(x)))
        x = x.view(-1, self.max_features, self.n_pixels_8, self.n_pixels_8)
        x = self.activation(self.bn2(self.conv1(x)))
        x = self.activation(self.bn3(self.conv2(x)))
        x = self.last_layer(self.conv3(x))
        return x


class Generator(nn.Module):
    def __init__(
        self,
        input_size: int,
        im_size: int,
        num_channels: int,
        min_features: int = 64,
        max_features: int = 512,
        first_layer: nn.Module = nn.Identity(),
        last_layer: nn.Module = nn.Tanh(),
        use_batch_norm: bool = True,
        y_kernel: int = 4,
        x_kernel: int = 4,
        act_fun: str = "ReLU",
        last_bias: bool = False,
    ) -> None:
        super(Generator, self).__init__()
        self.activation_functions: Dict[str, nn.Module] = {
            "ReLU": nn.ReLU(inplace=True),
            "Tanh": nn.Tanh(),
            "Softplus": nn.Softplus(),
            "SoftSign": nn.Softsign(),
            "Mish": nn.Mish(inplace=True),
            "SiLU": nn.SiLU(inplace=True),
            "GELU": nn.GELU(),
            "CELU": nn.SELU(inplace=True),
            "LeakyReLU": nn.LeakyReLU(inplace=True),
            "ELU": nn.ELU(inplace=True),
        }
        activation_function = self.activation_functions[act_fun]
        self.use_batch_norm: bool = use_batch_norm
        self.first_layer: nn.Module = first_layer
        self.last_layer: nn.Module = last_layer
        # some possible last layers include
        # nn.Tanh()
        # nn.Identity()
        # nn.ReLU(True)
        # RB()
        n_layers = np.log2(im_size) - 1
        if np.floor(n_layers) != n_layers:
            print("warning: output_size ({im_size}) is not a power of 2")
        n_layers = int(np.floor(n_layers))

        l1_out_features = np.minimum(min_features * 2**n_layers, max_features)
        first_kernel = (y_kernel, x_kernel)
        max_kernel = max(first_kernel)
        if self.use_batch_norm:
            layer_1 = (
                self.first_layer,
                nn.ConvTranspose2d(input_size, l1_out_features, first_kernel, 1, 0, bias=False),
                nn.BatchNorm2d(l1_out_features),
                activation_function,
            )
        else:
            layer_1 = (
                self.first_layer,
                nn.ConvTranspose2d(input_size, l1_out_features, first_kernel, 1, 0, bias=False),
                activation_function,
            )
        target_image = im_size // 2
        middle_layers = ()
        out_features = min_features
        while target_image > max_kernel:
            target_image = target_image // 2
            if out_features * 2 <= l1_out_features:
                in_features = out_features * 2
            else:
                in_features = out_features
            if self.use_batch_norm:
                middle_layers = (
                    nn.ConvTranspose2d(in_features, out_features, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_features),
                    activation_function,
                ) + middle_layers
            else:
                middle_layers = (
                    nn.ConvTranspose2d(in_features, out_features, 4, 2, 1, bias=False),
                    activation_function,
                ) + middle_layers
            out_features = in_features

        layers = (
            layer_1
            + middle_layers
            + (
                nn.ConvTranspose2d(min_features, num_channels, 4, 2, 1, bias=last_bias),
                # Hardtanh and (None) also seem to work well
                self.last_layer,
            )
        )

        self.main: nn.Sequential = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)


class Generator3D(nn.Module):
    def __init__(
        self,
        input_size: int,
        im_size: int,
        num_channels: int,
        min_features: int = 64,
        max_features: int = 512,
        first_layer: nn.Module = nn.Identity(),
        last_layer: nn.Module = nn.Tanh(),
        use_batch_norm: bool = True,
        z_kernel: int = 4,
        y_kernel: int = 4,
        x_kernel: int = 4,
        act_fun: str = "ReLU",
        last_bias: bool = False,
    ) -> None:
        """
        Generator that uses a dense 3D generator to estimate a 3D field.
        """
        super(Generator3D, self).__init__()
        self.activation_functions: Dict[str, nn.Module] = {
            "ReLU": nn.ReLU(inplace=True),
            "Tanh": nn.Tanh(),
            "Softplus": nn.Softplus(),
            "SoftSign": nn.Softsign(),
            "Mish": nn.Mish(inplace=True),
            "SiLU": nn.SiLU(inplace=True),
            "GELU": nn.GELU(),
            "CELU": nn.SELU(inplace=True),
            "LeakyReLU": nn.LeakyReLU(inplace=True),
            "ELU": nn.ELU(inplace=True),
        }
        activation_function = self.activation_functions[act_fun]
        self.use_batch_norm: bool = use_batch_norm
        self.first_layer: nn.Module = first_layer
        self.last_layer: nn.Module = last_layer
        # some possible last layers include
        # nn.Tanh()
        # nn.Identity()
        # nn.ReLU(True)
        # RB()
        n_layers = np.log2(im_size) - 1
        if np.floor(n_layers) != n_layers:
            print("warning: output_size ({im_size}) is not a power of 2")
        n_layers = int(np.floor(n_layers))

        l1_out_features = np.minimum(min_features * 2**n_layers, max_features)
        first_kernel = (z_kernel, y_kernel, x_kernel)
        max_kernel = max(first_kernel)
        if self.use_batch_norm:
            layer_1 = (
                self.first_layer,
                nn.ConvTranspose3d(input_size, l1_out_features, first_kernel, 1, 0, bias=False),
                nn.BatchNorm3d(l1_out_features),
                activation_function,
            )
        else:
            layer_1 = (
                self.first_layer,
                nn.ConvTranspose3d(input_size, l1_out_features, first_kernel, 1, 0, bias=False),
                activation_function,
            )
        target_image = im_size // 2
        middle_layers = ()
        out_features = min_features
        while target_image > max_kernel:
            target_image = target_image // 2
            if out_features * 2 <= l1_out_features:
                in_features = out_features * 2
            else:
                in_features = out_features
            if self.use_batch_norm:
                middle_layers = (
                    nn.ConvTranspose3d(in_features, out_features, 4, 2, 1, bias=False),
                    nn.BatchNorm3d(out_features),
                    activation_function,
                ) + middle_layers
            else:
                middle_layers = (
                    nn.ConvTranspose3d(in_features, out_features, 4, 2, 1, bias=False),
                    activation_function,
                ) + middle_layers
            out_features = in_features

        layers = (
            layer_1
            + middle_layers
            + (
                nn.ConvTranspose3d(min_features, num_channels, 4, 2, 1, bias=last_bias),
                # Hardtanh and (None) also seem to work well
                self.last_layer,
            )
        )

        self.main: nn.Sequential = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)


class Generator3DTriplane(nn.Module):
    def __init__(
        self,
        input_size: int,
        im_size: int,
        num_channels: int,
        intermediate_channels: int = 3,
        min_features: int = 64,
        max_features: int = 512,
        first_layer: nn.Module = nn.Identity(),
        last_layer: nn.Module = nn.Tanh(),
        use_batch_norm: bool = True,
        y_kernel: int = 4,
        x_kernel: int = 4,
        act_fun: str = "ReLU",
        last_bias: bool = False,
    ) -> None:
        """
        Generator that uses three 2D child generators and a
        fully-connected reconstruction network to estimate a 3D field.
        """
        super(Generator3DTriplane, self).__init__()

        if intermediate_channels == 0:
            intermediate_channels = num_channels

        # Note: we are forwarding all arguments to the child generators except for the last layer
        self.generator_x: Generator = Generator(
            input_size,
            im_size,
            intermediate_channels,
            min_features=min_features,
            max_features=max_features,
            first_layer=first_layer,
            use_batch_norm=use_batch_norm,
            y_kernel=y_kernel,
            x_kernel=x_kernel,
            act_fun=act_fun,
            last_bias=last_bias,
        )
        self.generator_y: Generator = Generator(
            input_size,
            im_size,
            intermediate_channels,
            min_features=min_features,
            max_features=max_features,
            first_layer=first_layer,
            use_batch_norm=use_batch_norm,
            y_kernel=y_kernel,
            x_kernel=x_kernel,
            act_fun=act_fun,
            last_bias=last_bias,
        )
        self.generator_z: Generator = Generator(
            input_size,
            im_size,
            intermediate_channels,
            min_features=min_features,
            max_features=max_features,
            first_layer=first_layer,
            use_batch_norm=use_batch_norm,
            y_kernel=y_kernel,
            x_kernel=x_kernel,
            act_fun=act_fun,
            last_bias=last_bias,
        )

        activation_function = self.generator_z.activation_functions[act_fun]
        self.reconstruction_layers = nn.Sequential(
            nn.Conv3d(in_channels=3 * intermediate_channels, out_channels=3 * intermediate_channels, kernel_size=1),
            nn.BatchNorm3d(3 * intermediate_channels),
            activation_function,
            nn.Conv3d(in_channels=3 * intermediate_channels, out_channels=num_channels, kernel_size=1),
            nn.BatchNorm3d(num_channels),
            activation_function,
        )
        self.last_layer = last_layer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Evaluate the three generators to get tensors of size
        # [B, C, Y, Z], [B, C, X, Z], and [B, C, X, Y]
        x = self.generator_x(input).unsqueeze(2)
        y = self.generator_y(input).unsqueeze(3)
        z = self.generator_z(input).unsqueeze(4)

        # Broadcast tensors to common shape of [B, C, X, Y, Z]
        Nb, Nc, _, Ny, Nz = x.shape
        _, _, Nx, _, _ = y.shape
        bx = torch.broadcast_to(x, (Nb, Nc, Nx, Ny, Nz))
        by = torch.broadcast_to(y, (Nb, Nc, Nx, Ny, Nz))
        bz = torch.broadcast_to(z, (Nb, Nc, Nx, Ny, Nz))

        a = torch.cat([bx, by, bz], dim=1)  # shape: (Nb, 3*Nc, Nx, Ny, Nz)
        b = self.reconstruction_layers(a)
        return self.last_layer(b)


def _build_middle_layer_schedule(
    im_size: int, input_size: Iterable[int], input_features: int, min_features: int
) -> tuple[list[int], list[int], list[int]]:
    layer_input_features: list[int] = []
    layer_output_features: list[int] = []
    layer_min_axis: list[int] = []
    target_image = im_size // 2
    layer_size = list(input_size)

    while min(layer_size) < target_image:
        layer_input_features.append(input_features)
        input_features = max(min_features, input_features // 2)
        layer_output_features.append(input_features)
        layer_size = [2 * x for x in layer_size]
        layer_min_axis.append(min(layer_size))

    if len(layer_output_features):
        layer_output_features[-1] = min_features
    else:
        print("Model configuration requires no middle layers")

    return layer_input_features, layer_output_features, layer_min_axis


class Generator3DSpectral(nn.Module):
    def __init__(
        self,
        input_size: int,
        im_size: int,
        num_channels: int,
        min_features: int = 64,
        max_features: int = 512,
        first_layer: nn.Module = nn.Identity(),
        last_layer: nn.Module = nn.Tanh(),
        use_batch_norm: bool = True,
        z_kernel: int = 4,
        y_kernel: int = 4,
        x_kernel: int = 4,
        n_modes: int = 8,
        act_fun: str = "ReLU",
        last_bias: bool = False,
    ) -> None:
        """
        Generator that uses a dense 3D generator to estimate a 3D field.
        """
        from neuralop.layers.spectral_convolution import SpectralConv # type: ignore

        super(Generator3DSpectral, self).__init__()
        self.activation_functions: Dict[str, nn.Module] = {
            "ReLU": nn.ReLU(inplace=True),
            "Tanh": nn.Tanh(),
            "Softplus": nn.Softplus(),
            "SoftSign": nn.Softsign(),
            "Mish": nn.Mish(inplace=True),
            "SiLU": nn.SiLU(inplace=True),
            "GELU": nn.GELU(),
            "CELU": nn.SELU(inplace=True),
            "LeakyReLU": nn.LeakyReLU(inplace=True),
            "ELU": nn.ELU(inplace=True),
        }

        activation_function = self.activation_functions[act_fun]
        self.use_batch_norm: bool = use_batch_norm
        self.first_layer: nn.Module = first_layer
        self.last_layer: nn.Module = last_layer

        n_layers = np.log2(im_size) - 1
        if np.floor(n_layers) != n_layers:
            print("warning: output_size ({im_size}) is not a power of 2")
        n_layers = int(np.floor(n_layers))

        out_features = np.minimum(min_features * 2**n_layers, max_features)
        layer_size = (x_kernel, y_kernel, z_kernel)

        # Build the first layers
        layers: list[nn.Module] = [
            self.first_layer,
            nn.ConvTranspose3d(input_size, out_features, layer_size, 1, 0, bias=False),
        ]
        if self.use_batch_norm:
            layers.append(nn.BatchNorm3d(out_features))

        layers.append(activation_function)

        # Build the middle layers
        layer_input_features, layer_output_features, layer_min_axis = _build_middle_layer_schedule(
            im_size, layer_size, out_features, min_features
        )
        for in_features, out_features, min_axis in zip(layer_input_features, layer_output_features, layer_min_axis):
            if min_axis >= 32:
                layers.append(
                    SpectralConv(
                        in_features, out_features, (n_modes, n_modes, n_modes), resolution_scaling_factor=2, bias=False
                    )
                )
            else:
                layers.append(nn.ConvTranspose3d(in_features, out_features, 4, 2, 1, bias=False))

            if self.use_batch_norm:
                layers.append(nn.BatchNorm3d(out_features))

            layers.append(activation_function)

        # Build the final layers
        layers.append(
            SpectralConv(
                min_features, num_channels, (n_modes, n_modes, n_modes), resolution_scaling_factor=2, bias=last_bias
            )
        )
        layers.append(self.last_layer)

        self.main: nn.Sequential = nn.Sequential(*layers)
        # self.main = layers

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = torch.reshape(input, (input.shape[0], input.shape[1], 1, 1, 1))
        # for ii, layer in enumerate(self.main):
        #     print(ii, layer, x.shape)
        #     x = layer(x)
        # return x
        return self.main(x)


class GenSubPixelConv(nn.Module):
    def __init__(
        self,
        input_size: int,
        im_size: int,
        num_channels: int,
        min_features: int = 64,
        max_features: int = 512,
        first_layer: nn.Module = nn.Identity(),
        last_layer: nn.Module = nn.Tanh(),
        use_batch_norm: bool = True,
        y_kernel: int = 4,
        x_kernel: int = 4,
        act_fun: str = "ReLU",
        last_bias: bool = False,
    ) -> None:
        super(GenSubPixelConv, self).__init__()
        self.activation_functions: Dict[str, nn.Module] = {
            "ReLU": nn.ReLU(inplace=True),
            "Tanh": nn.Tanh(),
            "Softplus": nn.Softplus(),
            "SoftSign": nn.Softsign(),
            "Mish": nn.Mish(inplace=True),
            "SiLU": nn.SiLU(inplace=True),
            "GELU": nn.GELU(),
            "CELU": nn.SELU(inplace=True),
            "LeakyReLU": nn.LeakyReLU(inplace=True),
            "ELU": nn.ELU(inplace=True),
        }
        activation_function = self.activation_functions[act_fun]
        self.use_batch_norm: bool = use_batch_norm
        self.first_layer: nn.Module = first_layer
        self.last_layer: nn.Module = last_layer
        # some possible last layers include
        # nn.Tanh()
        # nn.Identity()
        # nn.ReLU(True)
        # RB()
        n_layers = np.log2(im_size) - 1
        if np.floor(n_layers) != n_layers:
            print("warning: output_size ({im_size}) is not a power of 2")
        n_layers = int(np.floor(n_layers))

        l1_out_features = np.minimum(min_features * 2**n_layers, max_features)
        first_kernel = (y_kernel, x_kernel)
        max_kernel = max(first_kernel)
        if self.use_batch_norm:
            layer_1 = (
                self.first_layer,
                nn.ConvTranspose2d(
                    in_channels=input_size,
                    out_channels=l1_out_features,
                    kernel_size=first_kernel,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                # ConvPixelShuffle(
                #     input_size,
                #     l1_out_features,
                #     upscale_factor=2,
                #     kernel_size=first_kernel,
                #     stride=1,
                #     padding=0,
                #     bias=False,
                # ),
                nn.BatchNorm2d(l1_out_features),
                activation_function,
            )
        else:
            layer_1 = (
                self.first_layer,
                nn.ConvTranspose2d(
                    in_channels=input_size,
                    out_channels=l1_out_features,
                    kernel_size=first_kernel,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                # ConvPixelShuffle(
                #     input_size,
                #     l1_out_features,
                #     upscale_factor=2,
                #     # kernel_size=first_kernel,
                #     # stride=1,
                #     # padding=0,
                #     bias=False,
                # ),
                activation_function,
            )
        target_image = im_size // 2
        middle_layers = ()
        out_features = min_features
        while target_image > max_kernel:
            target_image = target_image // 2
            if out_features * 2 <= l1_out_features:
                in_features = out_features * 2
            else:
                in_features = out_features
            # print(in_features, out_features)
            if self.use_batch_norm:
                middle_layers = (
                    # nn.ConvTranspose2d(
                    #     in_channels=in_features,
                    #     out_channels=out_features,
                    #     kernel_size=4,
                    #     stide=2,
                    #     padding=1,
                    #     bias=False
                    # ),
                    ConvPixelShuffle(
                        in_features,
                        out_features,
                        upscale_factor=2,
                        # kernel_size=4,
                        # stride=2,
                        # padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_features),
                    activation_function,
                ) + middle_layers
            else:
                middle_layers = (
                    # nn.ConvTranspose2d(
                    #     in_channels=in_features,
                    #     out_channels=out_features,
                    #     kernel_size=4,
                    #     stide=2,
                    #     padding=1,
                    #     bias=False
                    # ),
                    ConvPixelShuffle(
                        in_features,
                        out_features,
                        upscale_factor=2,
                        # kernel_size=4,
                        # stride=2,
                        # padding=1,
                        bias=False,
                    ),
                    activation_function,
                ) + middle_layers
            out_features = in_features

        layers = (
            layer_1
            + middle_layers
            + (
                # nn.ConvTranspose2d(
                #     in_channels=min_features,
                #     out_channels=num_channels,
                #     kernel_size=4,
                #     stide=2,
                #     padding=1,
                #     bias=False
                # ),
                ConvPixelShuffle(
                    min_features,
                    num_channels,
                    upscale_factor=2,
                    # kernel_size=4,
                    # stride=2,
                    # padding=1,
                    bias=last_bias,
                ),
                # Hardtanh and (None) also seem to work well
                self.last_layer,
            )
        )

        self.main: nn.Sequential = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)


class GenResizeConv(nn.Module):
    def __init__(
        self,
        input_size: int,
        im_size: int,
        num_channels: int,
        min_features: int = 64,
        max_features: int = 512,
        first_layer: nn.Module = nn.Identity(),
        last_layer: nn.Module = nn.Tanh(),
        use_batch_norm: bool = True,
        y_kernel: int = 4,
        x_kernel: int = 4,
        act_fun: str = "ReLU",
    ) -> None:
        super(GenResizeConv, self).__init__()
        self.activation_functions: Dict[str, nn.Module] = {
            "ReLU": nn.ReLU(inplace=True),
            "Tanh": nn.Tanh(),
            "Softplus": nn.Softplus(),
            "SoftSign": nn.Softsign(),
            "Mish": nn.Mish(inplace=True),
            "SiLU": nn.SiLU(inplace=True),
            "GELU": nn.GELU(),
            "CELU": nn.SELU(inplace=True),
            "LeakyReLU": nn.LeakyReLU(inplace=True),
            "ELU": nn.ELU(inplace=True),
        }
        activation_function = self.activation_functions[act_fun]
        self.use_batch_norm: bool = use_batch_norm
        self.first_layer: nn.Module = first_layer
        self.last_layer: nn.Module = last_layer
        # some possible last layers include
        # nn.Tanh()
        # nn.Identity()
        # nn.ReLU(True)
        # RB()
        n_layers = np.log2(im_size) - 1
        if np.floor(n_layers) != n_layers:
            print("warning: output_size ({im_size}) is not a power of 2")
        n_layers = int(np.floor(n_layers))

        l1_out_features = np.minimum(min_features * 2**n_layers, max_features)
        first_kernel = (y_kernel, x_kernel)
        max_kernel = max(first_kernel)
        if self.use_batch_norm:
            layer_1 = (
                self.first_layer,
                nn.ConvTranspose2d(
                    in_channels=input_size,
                    out_channels=l1_out_features,
                    kernel_size=first_kernel,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(l1_out_features),
                activation_function,
            )
        else:
            layer_1 = (
                self.first_layer,
                nn.ConvTranspose2d(
                    in_channels=input_size,
                    out_channels=l1_out_features,
                    kernel_size=first_kernel,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                # ConvPixelShuffle(
                #     input_size,
                #     l1_out_features,
                #     upscale_factor=2,
                #     # kernel_size=first_kernel,
                #     # stride=1,
                #     # padding=0,
                #     bias=False,
                # ),
                activation_function,
            )
        target_image = im_size // 2
        middle_layers = ()
        out_features = min_features
        for _ in range(n_layers - 1):
            first_kernel = (first_kernel[0] * 2, first_kernel[1] * 2)
        final_image = first_kernel
        while target_image > max_kernel:
            first_kernel = (first_kernel[0] // 2, first_kernel[1] // 2)
            target_image = target_image // 2
            if out_features * 2 <= l1_out_features:
                in_features = out_features * 2
            else:
                in_features = out_features
            # print(in_features, out_features)
            if self.use_batch_norm:
                middle_layers = (
                    nn.Upsample(first_kernel),
                    nn.Conv2d(
                        in_channels=in_features,
                        out_channels=out_features,
                        kernel_size=(3, 3),
                        padding="same",
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_features),
                    activation_function,
                ) + middle_layers
            else:
                middle_layers = (
                    nn.Upsample(first_kernel),
                    nn.Conv2d(
                        in_channels=in_features,
                        out_channels=out_features,
                        kernel_size=(3, 3),
                        padding="same",
                        bias=False,
                    ),
                    activation_function,
                ) + middle_layers
            out_features = in_features

        layers = (
            layer_1
            + middle_layers
            + (
                nn.Upsample(final_image),
                nn.Conv2d(
                    in_channels=min_features,
                    out_channels=num_channels,
                    kernel_size=(3, 3),
                    padding="same",
                    bias=False,
                ),
                # Hardtanh and (None) also seem to work well
                self.last_layer,
            )
        )

        self.main: nn.Sequential = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)


class FlatView(nn.Module):
    def __init__(self, output_shape: int) -> None:
        super(FlatView, self).__init__()
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, self.output_shape)


class Encoder(nn.Module):
    def __init__(
        self,
        im_size: int,
        num_channels: int,
        latent_size: int,
        n_output: int,
        min_features: int = 64,
        max_features: int = 512,
        y_kernel: int = 4,
        x_kernel: int = 4,
        latent_actfun: nn.Module = nn.ReLU(True),
        last_layer: nn.Module = nn.Identity(),
    ) -> None:
        super(Encoder, self).__init__()
        n_layers = np.log2(im_size) - 1
        if np.floor(n_layers) != n_layers:
            print("warning: im_size ({im_size}) is not a power of 2")
        n_layers = int(np.floor(n_layers))
        first_kernel = (y_kernel, x_kernel)
        # min_kernel = min(first_kernel)
        max_kernel = max(first_kernel)
        if max_kernel < 4:
            first_kernel = (y_kernel, x_kernel)
            n_layers += 1
        layers = ()
        in_features = num_channels
        out_features = min_features
        for i in range(n_layers - 1):
            if i > 0:
                out_features = min(max_features, in_features * 2)
            layers += (
                nn.Conv2d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
                nn.BatchNorm2d(out_features),
                nn.ReLU(True),
            )
            in_features = out_features

        layers += (
            nn.Conv2d(
                in_channels=in_features,
                out_channels=latent_size,
                kernel_size=first_kernel,
                stride=1,
                padding=0,
                bias=True,
            ),
            # Sigmoid
            # nn.Tanh(),
            latent_actfun,
            nn.Identity(),
            FlatView(latent_size),
            nn.Linear(latent_size, n_output),
            last_layer,
        )
        # print(layers)
        self.main: nn.Sequential = nn.Sequential(*layers)
        self.layer_count: int = 0
        for _ in self.main.modules():
            self.layer_count += 1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)

    def latent_space(self, x: torch.Tensor) -> torch.Tensor:
        count = 0
        for m in self.main.modules():
            if count > 0:
                x = m(x)
            count += 1
            if count + 2 == self.layer_count:
                break
        return x

    def params_from_latent(self, x: torch.Tensor) -> torch.Tensor:
        count = 0
        for m in self.main.modules():
            count += 1
            if count + 2 >= self.layer_count:
                x = m(x)
        return x


class ConvAutoencoder(nn.Module):
    def __init__(
        self,
        im_size: int,
        num_channels: int,
        latent_size: int,
        min_features: int = 64,
        max_features: int = 512,
        ngpus: int = 0,
    ) -> None:
        super().__init__()
        self.ngpus = ngpus

        raise Exception("This method is under construction")
        # TODO: These calls don't match their function signature
        """
        self.encoder = Encoder(
            im_size=im_size,
            num_channels=num_channels,
            latent_size=latent_size,
            min_features=min_features,
            max_features=max_features,
            ngpus=ngpus,
        )
        self.decoder = Generator(
            im_size=im_size,
            num_channels=num_channels,
            input_size=latent_size,
            min_features=min_features,
            max_features=max_features,
            ngpus=ngpus,
        )
        """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise Exception("This method is under construction")
        x = self.encoder(input)
        return self.decoder(x)
