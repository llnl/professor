# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: MIT

from typing import Tuple
import numpy as np
from professor.vela.config import Config
from professor.vela.gui import GUI, Sliders
from professor.vela.model import Model
from professor.vela.plugins import AnalyticalBase
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PenetrationCalculation(AnalyticalBase):
    def __init__(
        self, gui: GUI, config: Config, model: Model, sliders: Sliders
    ) -> None:
        """
        Call the parent super method to initialize the gui, model, and sliders
        class members.

        :param gui: Reference to the CreateGUI object for the viewer
        :param model: Reference to the InitModel object for the model params
        :param sliders: Reference to the Sliders object for slider access
        """
        super().__init__(gui, config, model, sliders)

    def update(self) -> None:
        """
        On every slider update, the tg-vortex implementation updates the energy
        spectrum graph. Anything that is desired to be done on the movement of
        sliders should be placed here.

        :return: None
        """
        phi2 = self.estimate_penn_depth()
        self.scalar_bars[0].set_height(phi2)
        self.scalar_fig.canvas.draw_idle()

    def estimate_penn_depth(self) -> float:
        """
        This implements Phi 2, equation 16 from:

        Dane M. Sterbentz, Charles F. Jekel, Daniel A. White, Robert N. Rieben,
        Jonathan L. Belof; Linear shaped-charge jet optimization using machine
        learning methods. J. Appl. Phys. 28 July 2023; 134 (4): 045102.
        https://doi.org/10.1063/5.0156373
        """
        image = self.gui._model.models[0].image[0, :].cpu().numpy()
        image = image[:, 230:280]
        liner = image[5]
        d = image[0]
        vx = image[1]
        idx = liner > 0.5  # these are the indices of the jet!
        phi_unintegrated = np.sqrt(d[idx])*vx[idx]
        phi2 = phi_unintegrated.sum() / idx.size
        # print(phi2, d[idx].shape)
        return phi2

    @AnalyticalBase.figure
    def scalar_bar_chart(
        self, width: float, height: float
    ) -> Tuple[FigureCanvasQTAgg, str]:
        """
        Creates the energy spectrum graph used for the tg-vortex simulation.
        It is decorated with AnalyticalBase.figure to be discovered as a graph
        that should be added to the viewer.

        :param width: The width of the FigureCanvas
        :param height: The height of the FigureCanvas
        :return: A tuple containing the FigureCanvasQTAgg object, and the
            display name for the graph in the viewer.
        """
        self.scalar_bar_chart = FigureCanvasQTAgg(
            Figure(figsize=(width, height)),
        )
        self.scalar_fig = self.scalar_bar_chart.figure
        self.scalar_ax = self.scalar_fig.add_axes([0.1, 0.15, 0.85, 0.75])

        phi2 = self.estimate_penn_depth()
        fruits = ['Phi-2']
        counts = [phi2]
        bar_labels = ['blue']
        bar_colors = ['tab:blue']

        self.scalar_bars = self.scalar_ax.bar(
            fruits, counts, label=bar_labels, color=bar_colors,
        )
        self.scalar_ax.set_title('Estimated penetration depth')
        self.scalar_ax.set_ylim((0.0, 0.4))
        return (self.scalar_bar_chart, "Penetration depth")
