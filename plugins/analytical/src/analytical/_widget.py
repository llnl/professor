# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from napari import Viewer

try:
    from qtpy.QtWidgets import QGridLayout, QPushButton, QTabWidget, QWidget
except ImportError:
    # TODO: Make sure that this works with pyside as well
    from PySide6.QtWidgets import QGridLayout, QPushButton, QTabWidget, QWidget

from scipy.spatial import KDTree
from professor.vela.config import Keys
from professor.vela.constants import GUI_INSTANCE_NAME
from professor.vela.utils import get_object_reference

from analytical.base import AnalyticalBase


class AnalyticalWidget(QWidget):
    YAML_KEY = "analytical"
    GRAPH_WIDTH = 4.5
    GRAPH_HEIGHT = 2
    TABLE_WIDTH = 4.5
    TABLE_HEIGHT = 2
    NEAREST_POINTS = "nearest_points"

    def __init__(self, napari_viewer: Viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.gui = self.viewer.__dict__[GUI_INSTANCE_NAME]
        self.model = self.gui._model
        self.config = self.gui._config
        self.sliders = self.gui.sliders
        self.plugin = self.config.plugins[self.YAML_KEY]

        self._create_initial_exact_image()
        self._create_initial_error_image()
        self._modify_gui_settings()

        self.setLayout(QGridLayout())

        self._instantiate_analytical_subclass()
        self._get_graphs()
        self._load_simulation_points()
        self._create_tables_figure_canvas()
        self._create_evaluate_button()
        self._match_sliders_to_simulation_button()
        self._create_revert_to_model_button()

        self.sliders.connect_sliders_to_callback(self.subclass.update)

        self._add_widgets()

    def _instantiate_analytical_subclass(self) -> None:
        """
        Instantiates the AnalyticalBase subclass target specified from the YAML config.

        ex: `target: analytical.tg_vortex_implementation.TaylorGreenVortexAnalytical`

        :return: None
        """
        class_ref = get_object_reference(self.plugin[Keys.TARGET.value])
        self.subclass = class_ref(self.gui, self.config, self.model, self.sliders)

    def _add_widgets(self) -> None:
        """
        Build the widget.

        :return: None
        """
        row = 0
        col = 0
        row_span = 1
        col_span = 2

        if self.tabs is not None:
            self.layout().addWidget(self.tabs, row, col, row_span, col_span) # type: ignore
            row += 1
        self.layout().addWidget(self.tables, row, col, row_span, col_span) # type: ignore
        self.layout().addWidget(self.evaluate_button, row + 1, col, row_span, col_span) # type: ignore
        self.layout().addWidget(self.revert_to_ml_button, row + 2, col) # type: ignore
        self.layout().addWidget(self.match_simulation_button, row + 2, col + 1) # type: ignore

    def _get_graphs(self) -> None:
        """
        Get all the methods decorated with @AnalyticalBase.figure in the implementation
        class and add the returned graphs as a tab widget.

        :return: None
        """
        # Get all the callable method names as strings
        figures = [
            m
            for m in dir(self.subclass)
            if not m.startswith("__") and callable(getattr(self.subclass, m))
        ]

        self.tabs = None
        # If a callable method is decorated with the figure decorator, call it
        for func in figures:
            ref = getattr(self.subclass, func)
            if AnalyticalBase.is_figure_decorated(ref):
                if self.tabs is None:
                    self.tabs = QTabWidget()
                kwargs = {"width": self.GRAPH_WIDTH, "height": self.GRAPH_HEIGHT}
                graph, label = ref(**kwargs)
                self.tabs.addTab(graph, label)

    def _create_initial_exact_image(self) -> None:
        """
        Add an empty image to the viewer representing the exact solution.

        :return: None
        """
        self.viewer.add_image(
            np.ones((self.config.x_pixels, self.config.y_pixels)),
            colormap=self.gui.colormap,
            name="Exact Solution",
        )

    def _create_initial_error_image(self) -> None:
        """
        Add an empty image to the viewer representing the error image.

        :return: None
        """
        self.viewer.add_image(
            np.ones((self.config.x_pixels, self.config.y_pixels)),
            colormap=self.gui.colormap,
            name="L1 Error",
        )

    def _modify_gui_settings(self) -> None:
        """
        Modify the GUI settings to support the Grid View.

        Model Image | Exact Solution Image | Error Image

        :return: None
        """
        self.gui.napari_settings.application.grid_stride = -1
        self.gui.napari_settings.application.grid_height = 1
        self.viewer.grid.enabled = True

        self.viewer.layers.selection.active = self.viewer.layers[
            self.gui.checkpoint_names[0]
        ]

    def _create_tables_figure_canvas(self) -> None:
        """
        Creates a new Matplotlib canvas to hold tables.

        :return: None
        """
        self.tables = FigureCanvasQTAgg(
            Figure(figsize=(self.TABLE_WIDTH, self.TABLE_HEIGHT))
        )
        self.tables_fig = self.tables.figure

        self._create_error_table()
        self._create_nearest_simulation_points_table()

    def _create_error_table(self) -> None:
        """
        Create an empty Matplotlib table to hold the error values.

        :return: None
        """
        self.error_data = np.zeros((1, 4))
        # self.error_row_label = ("L1 Error", "L2 Error", "LInf Error", "Distance")
        self.error_col_label = ("L1 Error", "L2 Error", "LInf Error", "Distance")
        self.error_ax = self.tables_fig.add_subplot(2, 1, 2)
        # self.error_ax = self.tables_fig.add_axes([0.78, 0.15, 0.2, 0.67])
        # self.error_table = self.error_ax.table(
        #     cellText=self.error_data, rowLabels=self.error_row_label, loc="center"
        # )
        self.error_table = self.error_ax.table(
            cellText=self.error_data, colLabels=self.error_col_label, loc="center" # type: ignore
        )
        self.error_ax.set_title("Error Metrics", y=1.1, pad=-14)
        self.error_ax.set_axis_off()
        self.error_table.auto_set_column_width(
            col=list(range(len(self.error_col_label)))
        )

    def _create_nearest_simulation_points_table(self) -> None:
        """
        Create an empty Matplotlib table to hold the closest points values.

        :return: None
        """
        self.closest_points = np.zeros((1, 4))
        self.col_label = ("x length", "y length", "Exp", "Time")
        # self.closest_ax = self.tables_fig.add_axes([0.22, 0.15, 0.2, 0.67])
        self.closest_ax = self.tables_fig.add_subplot(2, 1, 1)
        self.closest_table = self.closest_ax.table(
            cellText=self.closest_points, colLabels=self.col_label, loc="center" # type: ignore
        )
        self.closest_ax.set_title("Closest Point", y=1.1, pad=-14)
        self.closest_ax.set_axis_off()
        self.closest_table.auto_set_column_width(col=list(range(len(self.col_label))))

    def _calculate_errors(self) -> None:
        """
        Calculate the L1 and L2 errors based on the exact solution calculated.

        :return: None
        """
        image = self.gui._model.models[0].image[0, 0].cpu().numpy()
        self.L1_error = np.abs(self.exact_solution - image)
        self.viewer.layers["L1 Error"].data = self.L1_error

        self._get_nearest_simulation_points()
        self.L2_error = np.square(self.exact_solution - image)
        self._update_error_table()

    def _load_simulation_points(self) -> None:
        """
        Load the simulation points from the file.

        :return: None
        """
        path = self.plugin[Keys.SETTINGS.value][self.NEAREST_POINTS]
        self.simulation_points = np.load(path)

    def _get_nearest_simulation_points(self) -> None:
        """
        Based on the current slider values, get the closest simulation point.

        :return: None
        """
        slider_values = []
        for param_name in self.config.sliders:
            current_slider = self.sliders.__dict__[param_name]["slider"]
            slider_values.append(current_slider.value())

        slider_count = len(self.config.sliders)

        sim_points = [0] * slider_count

        for i in range(slider_count):
            sim_points[i] = self.simulation_points[:, i]

        # NOTE: np.stack is emulating a call to np.c_ here. We can pass the sim_points
        # directly with np.stack since its a func call while we can't with np.c_
        tree = KDTree(np.stack(sim_points, axis=-1))
        self.dd, self.ii = tree.query([slider_values])

    def _update_error_table(self) -> None:
        """
        Update the error table with the calculated error values.

        :return: None
        """
        # NOTE: The self.dd[0] might break in the future depending on the nearest
        # points file

        self.error_data = [
            [
                np.round(np.sum(self.L1_error), 2),
                np.round(np.sum(self.L2_error), 2),
                np.round(np.max(self.L1_error), 2),
                np.round(self.dd[0], 2),
            ]
        ]
        self.error_ax.clear()
        self.error_ax.set_axis_off()
        self.error_ax.set_title("Error Metrics", y=1.1, pad=-14)
        self.error_table = self.error_ax.table(
            cellText=self.error_data, colLabels=self.error_col_label, loc="center"
        )
        self.error_table.auto_set_column_width(
            col=list(range(len(self.error_col_label)))
        )

    def _update_nearest_simulation_points_table(self) -> None:
        """
        Update the nearest simulation points table with the simulation data.

        :return: None
        """
        self.closest_points = np.round(self.simulation_points[self.ii], 3)
        self.closest_ax.clear()
        self.closest_ax.set_axis_off()
        self.closest_ax.set_title("Closest Point", y=1.1, pad=-14)
        self.closest_table = self.closest_ax.table(
            cellText=self.closest_points, colLabels=self.col_label, loc="center"
        )
        self.closest_table.auto_set_column_width(col=list(range(len(self.col_label))))

    def _subclass_evaluate(self) -> None:
        """
        Call the evaluate method of the implementation class and add the result to
        the viewer.

        :return: None
        """
        self.exact_solution = self.subclass.evaluate(
            self.config.x_pixels, self.config.y_pixels
        )
        self.viewer.layers["Exact Solution"].data = self.exact_solution
        self._save_slider_state()
        self._reset_match_simulations_button()

    def _update_tables(self) -> None:
        """
        Single call to update the Matplotlib tables.

        :return: None
        """
        self._calculate_errors()
        self._update_nearest_simulation_points_table()
        self.tables_fig.canvas.draw_idle()

    def _create_evaluate_button(self) -> None:
        """
        Add an `evaluate` button to the widget for triggering the evaluate sequence.

        :return: None
        """
        self.evaluate_button = QPushButton()
        self.evaluate_button.setText("Evaluate")
        self.evaluate_button.clicked.connect(self._subclass_evaluate)
        self.evaluate_button.clicked.connect(self._update_tables)
        self.evaluate_button.clicked.connect(self._set_slider_tip_to_nearest_simulation)
        self.evaluate_button.clicked.connect(self._reset_match_simulations_button)

        self.previous_slider_values = [0] * len(self.config.sliders.keys())

    def _match_sliders_to_simulation_button(self) -> None:
        """
        Add a `Match Sliders to Simulation` button.

        :return: None
        """
        self.match_simulation_button = QPushButton()
        self.match_simulation_button.setText("Match Sliders to Simulation")
        self.match_simulation_button.setDisabled(True)
        self.match_simulation_button.clicked.connect(
            self._match_sliders_to_nearest_simulation
        )
        self.match_simulation_button.clicked.connect(
            self._set_slider_tip_to_nearest_simulation
        )

    def _create_revert_to_model_button(self) -> None:
        """
        Add a `Revert Sliders to ML` button.

        :return: None
        """
        self.revert_to_ml_button = QPushButton()
        self.revert_to_ml_button.setText("Revert Sliders to ML")
        self.revert_to_ml_button.setDisabled(True)
        self.revert_to_ml_button.clicked.connect(
            self._revert_sliders_to_previous_position
        )
        self.revert_to_ml_button.clicked.connect(
            self._set_slider_tip_to_nearest_simulation
        )

    def _save_slider_state(self) -> None:
        """
        Saves the current values of the sliders to a class member.

        :return: None
        """
        for idx, slider_name in enumerate(self.config.sliders.keys()):
            current_slider = self.sliders.__dict__[slider_name]["slider"]
            self.previous_slider_values[idx] = current_slider.value()

    def _set_slider_tip_to_nearest_simulation(self) -> None:
        """
        Adds hover information for the sliders displaying the difference
        between the ML prediction and simulation prediction for the slider.

        :return: None
        """
        for idx, slider_name in enumerate(self.config.sliders.keys()):
            current_slider = self.sliders.__dict__[slider_name]["slider"]
            new_value = self.closest_points[0][idx]
            s = f"ML: {current_slider.value():.5f} | "
            s += f"Simulation Point: {new_value:.5f} | "
            s += f"Diff: {(current_slider.value() - new_value):.5f}"
            current_slider.setToolTip(s)

    def _reset_match_simulations_button(self) -> None:
        """
        Resets the initial enabled/disabled state for the match and revert buttons.

        :return: None
        """
        self.match_simulation_button.setDisabled(False)
        self.revert_to_ml_button.setDisabled(True)

    def _revert_sliders_to_previous_position(self) -> None:
        """
        Reverts the sliders to the saved ML prediction position before they were
        moved to match the simulation position.

        :return: None
        """
        for idx, slider_name in enumerate(self.config.sliders.keys()):
            current_slider = self.sliders.__dict__[slider_name]["slider"]
            previous_value = self.previous_slider_values[idx]
            current_slider.setValue(previous_value)

        self.revert_to_ml_button.setDisabled(True)
        self.match_simulation_button.setDisabled(False)

    def _match_sliders_to_nearest_simulation(self) -> None:
        """
        Moves the sliders to match what the nearest simulation point is.

        :return: None
        """
        for idx, slider_name in enumerate(self.config.sliders.keys()):
            current_slider = self.sliders.__dict__[slider_name]["slider"]
            new_value = self.closest_points[0][idx]
            current_slider.setValue(new_value)

        self.revert_to_ml_button.setDisabled(False)
        self.match_simulation_button.setDisabled(True)
