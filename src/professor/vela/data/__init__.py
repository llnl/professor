# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

_data_path: str = os.path.dirname(os.path.abspath(__file__))
template_config_path: str = os.path.join(_data_path, "template_config.yaml")
