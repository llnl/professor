# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from professor.vela.model import PyTorchModel

# from omegaconf import OmegaConf
from omegaconf import OmegaConf
from professor.vela.config import Config
import pytest

"""
Mocker argument comes from pytest-mock
"""

YAML_EXAMPLE = """

"""
script_dir = os.path.dirname(__file__)
MOCK_CONFIG = OmegaConf.load(os.path.join(script_dir, "test-config.yml"))
print(MOCK_CONFIG)
MOCK_RADIO_RULES = {"materials": 0}


@pytest.fixture()
def model_mock(mocker):
    # mocker.patch("professor.vela.config.Config._load_cfg", return_value=MOCK_CONFIG)
    cfg = Config(os.path.join(script_dir, "test-config.yml"))
    model = PyTorchModel(cfg)
    return model


def test_init_patch(model_mock):
    assert model_mock is not None
    assert model_mock.models[-1].image.shape == (1, 1, 32, 32)


def test_load_cfg(model_mock):
    assert model_mock._config.conf == MOCK_CONFIG
    assert model_mock._config.conf["model"] == MOCK_CONFIG["model"]
    assert model_mock._config.seed == MOCK_CONFIG["model"]["pytorch"]["seed"]
    assert model_mock._config.dimensions == MOCK_CONFIG["model"]["pytorch"]["dimensions"]
    assert model_mock._config.invocation == MOCK_CONFIG["model"]["pytorch"]["invocation"]
    assert model_mock._config.execution == MOCK_CONFIG["model"]["pytorch"]["execution"]
    assert model_mock._config.gui == MOCK_CONFIG["gui"]["napari"]


# def test_define_radio_rules(model_mock):
#     model_mock._define_radio_rules()
#     assert model_mock.radio_rules == MOCK_RADIO_RULES


# def test_set_torch_device_gpu(model_mock, mocker):
#     i = mocker.patch("professor.vela.model.torch.cuda.device_count", return_value=4)
#     j = mocker.patch("professor.vela.model.torch.cuda.is_available", return_value=True)
#     k = mocker.patch("professor.vela.model.torch.cuda.set_device")
#     o = mocker.patch("professor.vela.model.torch.device", return_value="cuda")
#     m = mocker.patch("professor.vela.model.torch.cuda.current_device")
#     model_mock._set_torch_device()
#     assert model_mock.device == "cuda"
#     assert i.call_count == 1
#     assert j.call_count == 1
#     assert k.call_count == 1
#     assert o.call_count == 1
#     assert m.call_count == 1


# def test_set_torch_device_cpu(model_mock, mocker):
#     i = mocker.patch("professor.vela.model.torch.cuda.device_count")
#     j = mocker.patch("professor.vela.model.torch.cuda.is_available", return_value=False)
#     k = mocker.patch("professor.vela.model.torch.device", return_value="cpu")
#     model_mock._set_torch_device()
#     assert model_mock.device == "cpu"
#     assert i.call_count == 1
#     assert j.call_count == 1
#     assert k.call_count == 1


def test_set_seed(model_mock, mocker):
    i = mocker.patch("professor.vela.model.np.random.seed")
    j = mocker.patch("professor.vela.model.torch.manual_seed")
    k = mocker.patch("professor.vela.model.torch.cuda.is_available", return_value=True)
    o = mocker.patch("professor.vela.model.torch.cuda.manual_seed")
    model_mock._set_seed()
    assert i.call_count == 1
    assert j.call_count == 1
    assert k.call_count == 1
    assert o.call_count == 1
