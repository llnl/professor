# Copyright 2026, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from professor.layers import AlphaLinear, InputScaling, ScaledAlphaLinear
from professor.torch_models import Generator
from professor.utils import scaling_statistics
from professor.vela.config import FullSchema, Keys, build_template_config
from professor.vela.utils import get_object_reference, initialize_function_calls_in_parameters


def test_scaled_alpha_linear_is_alpha_linear_in_dataset_units():
    torch.manual_seed(0)
    layer = ScaledAlphaLinear(n_channels=3, n_dims=2, offset=[1.0, 0.0, -2.0], scale=[0.1, 2.0, 5.0])
    plain = AlphaLinear(n_channels=3, n_dims=2)
    x = torch.randn(2, 3, 4, 5)
    offset = torch.tensor([1.0, 0.0, -2.0]).view(1, 3, 1, 1)
    scale = torch.tensor([0.1, 2.0, 5.0]).view(1, 3, 1, 1)

    assert torch.allclose(layer(x), plain(x) * scale + offset)
    assert set(layer.state_dict()) == {"alpha", "beta", "offset", "scale"}
    assert [n for n, _ in layer.named_parameters()] == ["alpha", "beta"]


def test_scaled_alpha_linear_3d_and_defaults():
    layer = ScaledAlphaLinear(n_channels=2, n_dims=3)
    x = torch.randn(1, 2, 3, 4, 5)
    assert torch.allclose(layer(x), AlphaLinear(n_channels=2, n_dims=3)(x))
    with pytest.raises(ValueError, match="need 2 values"):
        ScaledAlphaLinear(n_channels=2, offset=[0.0])


def test_input_scaling_matches_reference():
    """log of the flagged inputs, then (x - offset) / scale, as in the Rivus Rayleigh-Taylor study"""
    offset = [-8.7, -3.3, 4.5, -1.2, 5000.0]
    scale = [2.0, 3.6, 1.0, 3.4, 5000.0]
    layer = InputScaling(offset=offset, scale=scale, log_inputs=[0, 1, 2, 3])
    raw = torch.tensor([[1e-4, 0.1, 90.0, 0.3, 2500.0], [1e-3, 1.0, 50.0, 0.01, 0.0]], dtype=torch.float64)

    expected = raw.clone()
    expected[:, :4] = torch.log(expected[:, :4])
    expected = (expected - torch.tensor(offset, dtype=torch.float64)) / torch.tensor(scale, dtype=torch.float64)
    output = layer.double()(raw.view(2, 5, 1, 1))

    assert output.shape == (2, 5, 1, 1)
    assert torch.allclose(output.view(2, 5), expected)


def test_input_scaling_gradient_is_finite_for_non_log_inputs():
    """zero and negative values of inputs that are not logged give finite gradients"""
    layer = InputScaling(offset=[0.0, 0.0], scale=[1.0, 2.0], log_inputs=[0])
    x = torch.tensor([[2.0, 0.0], [3.0, -1.0]]).view(2, 2, 1, 1).requires_grad_()
    layer(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.allclose(x.grad.view(2, 2), torch.tensor([[0.5, 0.5], [1.0 / 3.0, 0.5]]))


def test_scaling_layers_in_a_generator_round_trip_through_state_dict():
    torch.manual_seed(0)

    def build(offset, scale, in_offset, in_scale):
        return Generator(
            3,
            32,
            2,
            min_features=4,
            max_features=8,
            first_layer=InputScaling(in_offset, in_scale, log_inputs=[1]),
            last_layer=ScaledAlphaLinear(2, offset=offset, scale=scale),
        )

    model = build([1.0, 2.0], [0.5, 3.0], [0.1, -1.0, 2.0], [1.0, 2.0, 3.0])
    clone = build([0.0, 0.0], [1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    clone.load_state_dict(model.state_dict(), strict=True)
    x = torch.rand(2, 3, 1, 1) + 0.1
    assert torch.allclose(clone.eval()(x), model.eval()(x))


class _Dataset:
    def __init__(self, inputs, fields):
        self.inputs = inputs
        self.fields = fields

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return torch.from_numpy(self.inputs[i]).view(-1, 1, 1), torch.from_numpy(self.fields[i])


class _Comm:
    """the ranks of a communicator, run one after the other"""

    def __init__(self, rank, size, gathered):
        self.rank = rank
        self.size = size
        self.gathered = gathered

    def allgather(self, value):
        self.gathered[self.rank] = value
        return self.gathered


def _dataset(n=40):
    rng = np.random.default_rng(0)
    inputs = rng.uniform(0.5, 4.0, size=(n, 3)).astype(np.float32)
    fields = (rng.normal(size=(n, 2, 6, 5)) * [[[3.0]], [[0.01]]] + [[[1e3]], [[-2.0]]]).astype(np.float32)
    return inputs, fields


def test_scaling_statistics_of_all_samples():
    inputs, fields = _dataset()
    stats = scaling_statistics(_Dataset(inputs, fields), n_samples=1000, log_inputs=[2])

    per_channel = fields.astype(np.float64).transpose(1, 0, 2, 3).reshape(2, -1)
    log_inputs = inputs.astype(np.float64)
    log_inputs[:, 2] = np.log(log_inputs[:, 2])
    assert np.allclose(stats["field_mean"], per_channel.mean(axis=1))
    assert np.allclose(stats["field_std"], per_channel.std(axis=1))
    assert np.allclose(stats["input_min"], log_inputs.min(axis=0))
    assert np.allclose(stats["input_max"], log_inputs.max(axis=0))


def test_scaling_statistics_is_the_same_on_every_rank():
    """the ranks read different samples and gather the same statistics as one rank reading them all"""
    inputs, fields = _dataset()
    dataset = _Dataset(inputs, fields)
    serial = scaling_statistics(dataset, n_samples=9)

    size = 3
    # the last rank gathers the other ranks' parts
    gathered = [(None, None, None)] * size
    for rank in range(size):
        parallel = scaling_statistics(dataset, n_samples=9, comm=_Comm(rank, size, gathered))
    for key in serial:
        assert np.allclose(parallel[key], serial[key])


def test_scaling_statistics_rejects_log_of_non_positive_input():
    inputs, fields = _dataset()
    inputs[0, 1] = 0.0
    with pytest.raises(ValueError, match="must be positive"):
        scaling_statistics(_Dataset(inputs, fields), n_samples=1000, log_inputs=[1])


def test_build_template_config_with_scaling_layers(tmp_path):
    config_path = tmp_path / "template_config.yaml"
    build_template_config(
        config_fname=str(config_path),
        checkpoint_path="0000.pt",
        n_inputs=3,
        n_channels=2,
        x_pixels=16,
        y_pixels=16,
        min_features=4,
        max_features=8,
        x_kernel=2,
        y_kernel=2,
        input_parameters=[(f"parameter_{ii}", 0.5, 2.0) for ii in range(3)],
        field_offset=[1.0, 2.0],
        field_scale=[0.5, 3.0],
        input_offset=[0.1, -1.0, 2.0],
        input_scale=[1.0, 2.0, 3.0],
        log_inputs=[1],
    )

    conf = FullSchema(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)).val
    pytorch = conf[Keys.MODEL.value][Keys.PYTORCH.value]
    assert pytorch[Keys.EXECUTION.value][Keys.HALF_PRECISION.value] is False
    invocation = pytorch[Keys.INVOCATION.value]
    model = get_object_reference(invocation[Keys.TARGET.value])(
        **initialize_function_calls_in_parameters(invocation[Keys.PARAMS.value])
    )
    assert isinstance(model.first_layer, InputScaling)
    assert torch.equal(model.first_layer.log_mask.flatten(), torch.tensor([False, True, False]))
    assert torch.allclose(model.first_layer.scale.flatten(), torch.tensor([1.0, 2.0, 3.0]))
    assert isinstance(model.last_layer, ScaledAlphaLinear)
    assert torch.allclose(model.last_layer.offset.flatten(), torch.tensor([1.0, 2.0]))
    assert model(torch.rand(1, 3, 1, 1) + 0.5).shape == (1, 2, 16, 16)


def test_build_template_config_without_scaling_is_unchanged(tmp_path):
    config_path = tmp_path / "template_config.yaml"
    build_template_config(config_fname=str(config_path), checkpoint_path="0000.pt")
    conf = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    params = conf["model"]["pytorch"]["invocation"]["params"]
    assert params["first_layer"]["invocation"]["target"] == "torch.nn.Identity"
    assert params["last_layer"]["invocation"]["target"] == "professor.layers.AlphaLinear"
    assert conf["model"]["pytorch"]["execution"]["half_precision"] is True
