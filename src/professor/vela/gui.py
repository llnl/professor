# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import timeit
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
import napari
import napari.experimental
from napari.layers import Image
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from napari.settings import get_settings, NapariSettings

try:
    from qtpy import QtWidgets
    from qtpy.QtCore import Qt
except ImportError:
    # TODO: Make sure that this works with pyside as well
    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt

from superqt import QLabeledDoubleSlider
from typing import List, Optional, Tuple, Callable

from professor.vela.config import Config, Keys
from professor.vela.constants import GUI_INSTANCE_NAME
from professor.vela.model import Model
from professor.vela.utils import get_object_reference

logger: logging.Logger = logging.getLogger(__name__)


class GUI(ABC):
    @abstractmethod
    def __init__(self, config: Config, model: Model, colormap: str, theme: str):
        self._config: Config = config
        self._model: Model = model
        self.colormap: str = colormap
        self.theme: str = theme

    @abstractmethod
    def _initialize_viewer(self) -> None:
        """
        This should create an empty GUI and return a reference to it. The class
        member for the viewer should also be set.

        :return: The GUI object
        """
        pass

    @abstractmethod
    def _load_checkpoints(self) -> None:
        """
        Load the model checkpoints into the viewer.

        :return: None
        """
        pass


class Napari(GUI):
    MEAN_LAYER_TITLE: str = "μ"
    SD_LAYER_TITLE: str = "σ"

    def __init__(self, config: Config, model: Model, colormap: str, theme: str) -> None:
        super().__init__(config, model, colormap, theme)

        self.viewer: napari.Viewer
        self.sliders: Sliders
        self.colorbar: MPLColorbar
        self.mu_sd_toggle: bool
        self.mean_layer: Optional[Image] = None
        self.std_layer: Optional[Image] = None

        self._initialize_viewer()
        self._add_widgets()
        self._load_checkpoints()
        self._add_new_viewer_class_field()
        self._initialize_plugins()

        # Tracks new changes to the field selection -> initially first one
        self._previous_field = self._config.fields[0]

        # Center the images whenever the layers change
        self.viewer.reset_view()
        self.viewer.layers.events.inserted.connect(self._reset_camera_callback)
        self.viewer.layers.events.removed.connect(self._reset_camera_callback)

        # Keep GUI alive
        napari.run()

    def _reset_camera_callback(self):
        logger.debug("Resetting the camera")
        self.viewer.reset_view()

    def _initialize_viewer(self) -> None:
        # Open an empty viewer
        self.viewer: napari.Viewer = napari.Viewer()
        self.napari_settings: NapariSettings = get_settings()

        # Change the theme and title of the viewer window
        self.viewer.theme = self.theme
        self.viewer.title = self._config.model_name

        # Put all the checkpoint images in a grid formation
        self.napari_settings.application.grid_stride = -1
        self.napari_settings.application.grid_height = 1
        self.viewer.grid.enabled = True

        # NOTE: CJ napari 5.0 broke the FPS counter
        # FPS measurement in the viewer
        # self.viewer.text_overlay.visible = True
        # self.viewer.window.qt_viewer.canvas.measure_fps(callback=self._update_fps)

    def _load_checkpoints(self) -> None:
        self.checkpoint_names: List[str] = []

        for i in range(0, self._model.num_checkpoints):
            # c = self._model.CHECKPOINT_STR + str(i)
            current_checkpoint = self._model.models[i]
            new_layer = self.viewer.add_image(
                current_checkpoint.image[0, 0].cpu().numpy(),
                name=current_checkpoint.name,
                colormap=self.colormap,
            )
            self.checkpoint_names.append(current_checkpoint.name)
            # Save a reference to the gui_layer in the model collection
            assert isinstance(new_layer, Image)
            current_checkpoint.gui_layer = new_layer

            current_checkpoint.gui_layer.events.colormap.connect(self.colorbar.update_colorbar)

        self.viewer.layers.selection.active = self.viewer.layers[self.checkpoint_names[0]]

    def _add_widgets(self) -> None:
        self.sliders = Sliders(self._config, self._model, self)
        self.viewer.window.add_dock_widget(self.sliders, area="bottom")

        self.colorbar = MPLColorbar(self._config, self._model, self, self.sliders)
        self.viewer.window.add_dock_widget(self.colorbar.widget, area="top")

    def _initialize_plugins(self) -> None:
        """
        Initialize the plugins specified in the configuration file.

        :return: None
        """
        self.using_plugins: bool = False
        if self._config.plugins is not None:
            self.using_plugins = True
            for p in self._config.plugins:
                if self._config.plugins[p][Keys.AUTO_LOAD.value]:
                    if p == "custom-widget":
                        widget_name = self._config.plugins[p]["widget"]
                        logger.debug("Attempting to load custom-widget")
                        # get object reference does not initialize the object!
                        myWidget = get_object_reference(widget_name)(self.viewer)
                        self.viewer.window.add_dock_widget(myWidget)
                    else:
                        self.viewer.window.add_plugin_dock_widget(p)
                    logger.info(f"Auto loading plugin: {p}")
        # NOTE: This is a hacky way to do this - it would be nice if plugins
        # could choose if they wanted this update plot stats feature too
        else:
            # Tracks whether the stats button is on
            self.mu_sd_toggle = False
            self.update_plot_stats()

    def _add_new_viewer_class_field(self) -> None:
        """
        In order to share instances of objects between external things like plugins,
        we store the references as metadata in the main image layer. Since this class
        is the main entry point and contains references to all the other subwidgets,
        we can just share this class reference.
        """
        self.viewer.__dict__[GUI_INSTANCE_NAME] = self

    # def _update_fps(self, fps):
    #     """Updates FPS Text Overlay"""
    #     self.viewer.text_overlay.text = f"{fps:1.1f} FPS"
    #     logger.info(f"FPS: {fps}")

    def update_image(self) -> None:
        key = self.sliders.field_selector.currentText()
        idx = self._config.field_map[key]
        new_key = False

        for i in range(0, self._model.num_checkpoints):
            # c = self._model.CHECKPOINT_STR + str(i)
            current_checkpoint = self._model.models[i]

            j = 0
            for slider_name in self._config.sliders.keys():
                current_slider = self.sliders.__dict__[slider_name]["slider"]
                current_checkpoint.tensor[0, j, 0, 0] = current_slider.value()
                j += 1

            start = timeit.default_timer()
            current_checkpoint.inference()
            end = timeit.default_timer()

            logger.info(f"Inference Time ({current_checkpoint.device}): {end - start}")

            # TODO: Have to make a call to CPU to get it to work. Expected
            # that should not have to do that.
            new_img = current_checkpoint.image[0, idx].cpu().numpy()
            current_checkpoint.gui_layer.data = new_img  # type: ignore

            if self._previous_field != key:
                new_key = True
                assert current_checkpoint.gui_layer is not None
                current_checkpoint.gui_layer.reset_contrast_limits()

        if new_key:
            self._previous_field = key

    def update_colorbar(self) -> None:
        self.colorbar.update_colorbar()

    def reset_parameters(self) -> None:
        for slider_name in self._config.sliders.keys():
            current_slider = self.sliders.__dict__[slider_name]["slider"]
            initial_value = self._config.sliders[slider_name]["initial_value"]
            current_slider.setValue(initial_value)
        self.update_colorbar()

    def update_plot_stats(self) -> None:
        if self.using_plugins:
            return
        key = self.sliders.field_selector.currentText()
        channel = self._config.field_map[key]
        if self.sliders.stats_toggle and not self.mu_sd_toggle:
            self.mu_sd_toggle = True
            mean_img, std_img = self._model.get_pixelwise_mean_and_sd(channel)

            logger.debug(f"Adding {self.MEAN_LAYER_TITLE}")
            img = self.viewer.add_image(
                mean_img,
                name=self.MEAN_LAYER_TITLE,
                colormap=self.colormap,
            )
            assert isinstance(img, Image)
            self.mean_layer = img

            logger.debug(f"Adding {self.SD_LAYER_TITLE}")
            img = self.viewer.add_image(
                std_img,
                name=self.SD_LAYER_TITLE,
                colormap=self.colormap,
            )
            assert isinstance(img, Image)
            self.std_layer = img

        elif self.sliders.stats_toggle and self.mu_sd_toggle:
            if (self.mean_layer is not None) and (self.std_layer is not None):
                # We are just updating the layers now
                mean_img, std_img = self._model.get_pixelwise_mean_and_sd(channel)
                self.mean_layer.data = mean_img
                self.std_layer.data = std_img
                self.mean_layer.reset_contrast_limits()
                self.std_layer.reset_contrast_limits()

    def remove_plot_stats(self) -> None:
        self.mu_sd_toggle = False
        self.viewer.layers.pop(self.MEAN_LAYER_TITLE)  # TODO: This call is expecting int not str
        logger.debug(f"Removing layer: {self.MEAN_LAYER_TITLE}")
        self.viewer.layers.pop(self.SD_LAYER_TITLE)  # TODO: This call is expecting int not str
        logger.debug(f"Removing layer: {self.SD_LAYER_TITLE}")
        self.viewer.layers.selection.active = self.viewer.layers[self.checkpoint_names[0]]
        logger.debug(f"Selecting layer: {self.checkpoint_names[0]}")


class Sliders(QtWidgets.QWidget):
    SLIDERS_DECIMAL_COUNT: int = 4

    def __init__(self, config: Config, model: Model, gui: Napari, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self._layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        self._config: Config = config
        self._model: Model = model
        self._gui: Napari = gui

        self.reset_button: QtWidgets.QPushButton
        self.reset_button: QtWidgets.QPushButton
        self.stats_button: QtWidgets.QPushButton

        self._create_field_selector_widget()
        self._create_slider_and_label_widgets()
        self._create_reset_button_widget()

        self.stats_toggle: bool = False
        if self._model.num_checkpoints > 1:
            # Only add the stats button when there are multiple checkpoints
            self.stats_toggle = True
            self._create_stats_button_widget()

        self.setLayout(self._layout)

    def _create_field_selector_widget(self) -> None:
        self.field_selector: QtWidgets.QComboBox = QtWidgets.QComboBox()
        self.field_selector.addItems(self._config.fields)
        self.field_selector.currentTextChanged.connect(self._gui.update_image)
        self.field_selector.currentTextChanged.connect(self._gui.update_colorbar)
        self.field_selector.currentTextChanged.connect(self._gui.update_plot_stats)
        self._layout.addWidget(self.field_selector)

    def _create_slider_and_label_widgets(self) -> None:
        # {param_name: {"label": QLabel, "slider": QSlider}}
        for slider_name in self._config.sliders.keys():
            self.__dict__[slider_name] = {
                "label": QtWidgets.QLabel(),
                "slider": QLabeledDoubleSlider(Qt.Orientation.Horizontal),
            }

            lower_bound = self._config.sliders[slider_name]["lower_bound"]
            upper_bound = self._config.sliders[slider_name]["upper_bound"]

            # Set the label values
            current_label = self.__dict__[slider_name]["label"]
            current_label.setText(f"{slider_name}: [{lower_bound}, {upper_bound}]")

            # Set the slider values
            current_slider = self.__dict__[slider_name]["slider"]
            current_slider.setDecimals(self.SLIDERS_DECIMAL_COUNT)
            current_slider.setMinimum(lower_bound)
            current_slider.setMaximum(upper_bound)

            initial_val = self._config.sliders[slider_name]["initial_value"]
            current_slider.setValue(initial_val)

            current_slider.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)

            current_slider.valueChanged.connect(self._gui.update_image)
            current_slider.sliderReleased.connect(self._gui.update_colorbar)
            current_slider.sliderReleased.connect(self._gui.update_plot_stats)

            self._layout.addWidget(current_label)
            self._layout.addWidget(current_slider)

    def _create_reset_button_widget(self) -> None:
        """Create The Reset Button Widget"""
        self.reset_button = QtWidgets.QPushButton()
        self.reset_button.setText("Reset All Parameters")
        self.reset_button.clicked.connect(self._gui.reset_parameters)
        self.reset_button.clicked.connect(self._gui.update_plot_stats)
        self._layout.addWidget(self.reset_button)

    def _create_stats_button_widget(self) -> None:
        self.stats_button = QtWidgets.QPushButton()
        self.stats_button.setText("Remove Stats Images")
        self.stats_button.clicked.connect(self._toggle_stats_button)
        self._layout.addWidget(self.stats_button)

    def _toggle_stats_button(self) -> None:
        if self.stats_toggle:
            self.stats_button.setText("Add Stats Images")
            self.stats_toggle = False
            self._gui.remove_plot_stats()
        else:
            self.stats_button.setText("Remove Stats Images")
            self.stats_toggle = True
            self._gui.update_plot_stats()

    def connect_sliders_to_callback(self, func: Callable) -> None:
        """
        Connect the sliders to a function when any slider value changes.
        This is useful for if a plugin wants to update something based on if any
        slider value changes.
        """
        for slider_name in self._config.sliders:
            current_slider = self.__dict__[slider_name]["slider"]
            current_slider.valueChanged.connect(func)


class MPLColorbar:
    def __init__(self, config: Config, model: Model, gui: Napari, sliders: Sliders) -> None:
        self._config: Config = config
        self._model: Model = model
        self._gui: Napari = gui
        self._sliders: Sliders = sliders

        if self._gui.theme == "dark":
            plt.style.use("dark_background")

        self.widget: FigureCanvasQTAgg = FigureCanvasQTAgg(Figure(figsize=(5, 1)))
        self.fig: Figure = self.widget.figure
        # (left, bottom, width, height)
        self.ax: Axes = self.fig.add_axes((0.19, 0.4, 0.75, 0.3))
        cmin, cmax = self.calculate_clim()
        logger.info(f"Color Limits: {cmin} - {cmax}")
        self.cb: ColorbarBase = ColorbarBase(
            self.ax,
            orientation="horizontal",
            cmap=self._gui.colormap,
            norm=Normalize(cmin, cmax),
        )

    def calculate_clim(self) -> Tuple[float, float]:
        """
        Get the minimum and maximum color values from the array and return it.

        arr: Numpy image array
        """
        f = self._config.field_map[self._sliders.field_selector.currentText()]
        min = self._model.models[0].image[0, f].min().item()
        max = self._model.models[0].image[0, f].max().item()
        return min, max

    def update_colorbar(self) -> None:
        """
        Updates the color limits. Also updates the colormap if necessary.
        """
        cmin, cmax = self.calculate_clim()
        logger.debug(f"Updating color limits: ({cmin}, {cmax})")
        self.cb.mappable.set_clim(cmin, cmax)

        # NOTE: This will fail if there is no direct mapping from the napari
        # colormaps to the matplotlib colormaps. An extra mapping will have to
        # be done to avoid this, however this is probably a moot point since
        # any matplotlib colormap can be passed to CreateGUI.
        layer_cmap = self._gui.viewer.layers[self._gui.checkpoint_names[0]].colormap  # type: ignore
        logger.debug(f"Setting colorbar colormap to: {layer_cmap.name}")
        self.cb.mappable.set_cmap(layer_cmap.name)

        self.fig.canvas.draw_idle()
