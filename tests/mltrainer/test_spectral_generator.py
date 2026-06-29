import torch
import torch.nn as nn
from torchinfo import summary
from professor.torch_models import Generator3DSpectral
from professor.layers import AlphaLinear



def test_spectral_generator():
    batch_size = 1
    num_input = 4
    num_channels = 2

    x = torch.randn(batch_size, num_input)
    final_layer = AlphaLinear(n_channels=num_channels, n_dims=3)

    gen = Generator3DSpectral(num_input,
            128,
            num_channels=num_channels,
            min_features=16,
            max_features=64,
            z_kernel=8,
            y_kernel=8,
            x_kernel=8,
            first_layer=nn.Identity(),
            last_layer=final_layer)

    y = gen(x)
    print(y.shape)

    summary(
        gen,
        input_size=(batch_size, num_input, 1, 1),
        depth=7,
        device="cpu"
    )
