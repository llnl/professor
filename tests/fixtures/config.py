# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from omegaconf import OmegaConf
from professor.vela.config import (
    Config,
    FullSchema,
    Keys,
)

EXAMPLE_YAML = """
model:
  pytorch:
    name: Taylor Green Vortex
    checkpoints:
      - /usr/workspace/prof/models/tg-vortex-analytical/taylor-green-0150.pt
      - /usr/workspace/prof/models/tg-vortex-analytical/taylor-green-0150.pt
    devices:
      - 0
      - 1
    seed: 1231
    dimensions:
      n_inputs: 4
      batch_size: 1
      y_pixels: 512
      x_pixels: 512
    invocation:
      target: professor.torch_models.Generator
      params:
        input_size: ${model.pytorch.dimensions.n_inputs}
        im_size: ${model.pytorch.dimensions.y_pixels}
        num_channels: 1
        max_features: 1024
        min_features: 128
        first_layer:
          invocation:
            target: torch.nn.Identity
        last_layer:
          invocation:
            target: prof.layers.AlphaLinear
            params:
              n_channels: ${model.pytorch.invocation.params.num_channels}
        act_fun: "ReLU"
        y_kernel: 2
        x_kernel: 2
    execution:
      half_precision: True
      jit: True

gui:
  napari:
    fields:
      - materials
    sliders:
      a:
        lower_bound: 0.1
        upper_bound: 1.0
        initial_value: mean
      b:
        lower_bound: 0.1
        upper_bound: 1.0
        initial_value: mean
      n:
        lower_bound: 1.0
        upper_bound: 5.0
        initial_value: mean
      Time:
        lower_bound: 0.0
        upper_bound: 3.0
        initial_value: 0.0
    appearance:
      colormap: magma
      theme: light
    plugins:
      analytical:
        auto_load: True
        target: analytical.example_tg_vortex_implementation.TaylorGreenVortexAnalytical
        settings:
          nearest_points: /usr/workspace/prof/models/tg-vortex-analytical/all_inputs.npy
"""


@pytest.fixture
def raw_conf():
    return OmegaConf.create(EXAMPLE_YAML)


@pytest.fixture
def conf(raw_conf):
    return OmegaConf.to_container(raw_conf, resolve=True)


@pytest.fixture
def validated_conf(conf):
    return FullSchema(conf).val


@pytest.fixture
def model_conf(conf):
    return conf.get(Keys.MODEL.value)


@pytest.fixture
def pytorch_conf(model_conf):
    return model_conf.get(Keys.PYTORCH.value)


@pytest.fixture
def checkpoints_conf(pytorch_conf):
    return pytorch_conf.get(Keys.CHECKPOINTS.value)


@pytest.fixture
def dimensions_conf(pytorch_conf):
    return pytorch_conf.get(Keys.DIMENSIONS.value)


@pytest.fixture
def invocation_conf(pytorch_conf):
    return pytorch_conf.get(Keys.INVOCATION.value)


@pytest.fixture
def execution_conf(pytorch_conf):
    return pytorch_conf.get(Keys.EXECUTION.value)


@pytest.fixture
def gui_conf(conf):
    return conf.get(Keys.GUI.value)


@pytest.fixture
def napari_conf(gui_conf):
    return gui_conf.get(Keys.NAPARI.value)


@pytest.fixture
def fields_conf(napari_conf):
    return napari_conf.get(Keys.FIELDS.value)


@pytest.fixture
def sliders_conf(napari_conf):
    return napari_conf.get(Keys.SLIDERS.value)


@pytest.fixture
def appearance_conf(napari_conf):
    return napari_conf.get(Keys.APPEARANCE.value)


@pytest.fixture
def plugins_conf(napari_conf):
    return napari_conf.get(Keys.PLUGINS.value)


@pytest.fixture
def config_mock(mocker, raw_conf):
    # mocker.patch(
    #     "vela.config.OmegaConf.load",
    #     return_value=OmegaConf.create(raw_conf),
    # )
    mocker.patch("professor.vela.config.Config._load_cfg", return_value=OmegaConf.create(raw_conf))
    return Config("")
