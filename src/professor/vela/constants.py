# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Constants used by other modules
"""

import logging

LOG_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

VELA_INIT_NAME = "vela"
CONFIG_MODULE_NAME = "vela.config"
MODELS_MODULE_NAME = "vela.model"
GUI_MODULE_NAME = "vela.gui"

GUI_INSTANCE_NAME = "Vela GUI"
