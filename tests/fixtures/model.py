# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from professor.vela.model import PyTorchModel


@pytest.fixture
def pytorch_model_mock(config_mock):
    return PyTorchModel(config_mock)
