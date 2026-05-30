# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Union
from professor.vela import logger
from professor.vela.config import Config, Keys
from professor.vela.model import Model, PyTorchModel
from professor.vela.gui import GUI, Napari


class Vela:
    def __init__(
        self,
        cfgpath: str,
        colormap: Union[str, None] = None,
        theme: Union[str, None] = None,
    ):
        self.config: Config = Config(cfgpath)
        self.colormap: str
        self.theme: str
        self._set_colormap_and_theme_args(colormap, theme)
        self.model: Model = self._get_model()
        self.gui: GUI = self._get_gui()

    def _get_model(self) -> Model:
        name = self.config.get_top_level_key_name(model=True)  # noqa F841
        return PyTorchModel(self.config)

    def _get_gui(self) -> GUI:
        name = self.config.get_top_level_key_name(gui=True)  # noqa F841
        return Napari(self.config, self.model, self.colormap, self.theme)

    def _set_colormap_and_theme_args(self, colormap: Union[str, None], theme: Union[str, None]):
        # Prioritze the passed in colormap and theme over the config arguments
        if colormap is None:
            self.colormap = self.config.appearance[Keys.COLORMAP.value]
        else:
            self.colormap = colormap

        logger.info(f"Setting colormap to: {self.colormap}")

        if theme is None:
            self.theme = self.config.appearance[Keys.THEME.value]
        else:
            self.theme = theme

        logger.info(f"Setting theme to: {self.theme}")
