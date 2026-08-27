"""Regenerate the small model checkpoint used by unit tests."""

from pathlib import Path

import torch

from professor.layers import AlphaLinear
from professor.torch_models import Generator


torch.manual_seed(1231)

model = Generator(
    input_size=4,
    im_size=32,
    num_channels=1,
    min_features=2,
    max_features=4,
    first_layer=torch.nn.Identity(),
    last_layer=AlphaLinear(n_channels=1),
    y_kernel=2,
    x_kernel=2,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# One synthetic training epoch is enough to produce a valid trainer checkpoint.
inputs = torch.rand(2, 4, 1, 1)
target = torch.zeros(2, 1, 32, 32)
loss = torch.nn.functional.l1_loss(model(inputs), target)
loss.backward()
optimizer.step()

output_path = Path(__file__).with_name("tiny-generator.pt")
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, output_path)
print(f"Wrote {output_path} ({output_path.stat().st_size} bytes)")
