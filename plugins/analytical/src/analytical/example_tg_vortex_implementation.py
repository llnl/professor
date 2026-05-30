# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Tuple

import numpy as np
import scipy.stats as stats
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy.interpolate import griddata
from tqdm import tqdm
from professor.vela.config import Config
from professor.vela.gui import GUI, Sliders
from professor.vela.model import Model

from analytical.base import AnalyticalBase


class TaylorGreenVortexAnalytical(AnalyticalBase):
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

        image = self.gui._model.models[0].image[0, 0].cpu().numpy()
        npix = np.shape(image)[0]
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2

        kfreq = np.fft.fftfreq(npix) * npix
        kfreq2D = np.meshgrid(kfreq, kfreq)
        knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()

        kbins = np.arange(0.5, npix // 2 + 1, 1.0)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        Abins, _, _ = stats.binned_statistic(
            knrm, fourier_amplitudes, statistic="mean", bins=kbins
        )

        Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        # image 1

        self.energy_ml_line.set_data(kvals, Abins)
        self.energy_fig.canvas.draw_idle()

    def evaluate(self, x_pixels: int, y_pixels: int) -> np.ndarray:
        # Solve Exact Solution
        """
        Returns the exact solution image represented as a numpy array. Can use
        the x_pixels and y_pixels arguments to access the expected size of the
        returned image.

        Also placed here is logic related to updating the energy spectrum graph when
        the exact solution is calculated. When the calculations are finished, the
        calculated energy spectrum values are added to the energy spectrum graph.

        :param x_pixels: Amount of pixels the image should be in the x dimension.
        :param y_pixels: Amount of pixels the image should be in the y dimension.
        :return: An np array representing the image.
        """
        N = 2048
        S = 2.5
        dt = 0.1
        t = 0
        my_dpi = 512
        # Initialize grid
        X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))  # stretched
        X = 0.5 * (1 + np.tanh(S * X) / np.tanh(S))
        Y = 0.5 * (1 + np.tanh(S * Y) / np.tanh(S))

        X = X.flatten()
        Y = Y.flatten()
        X0, Y0 = np.meshgrid(np.linspace(0, 1, my_dpi), np.linspace(0, 1, my_dpi))

        res = []
        for param_name in self.config.sliders:
            current_slider = self.sliders.__dict__[param_name]["slider"]
            res.append(current_slider.value())

        a = res[0]
        b = res[1]
        n = res[2]
        ts = res[3]
        # IC for tracing
        r = pow(np.abs((X - 0.5) / a), n) + pow(np.abs((Y - 0.5) / b), n)
        alpha = 1 / 50
        # mask = r < pow(0.5,n)
        mask = 1 / 2 + 1 / 2 * (np.tanh((pow(0.5, n) - r) / alpha))
        mask = mask.flatten().astype(dtype=float)

        # ode solve
        N_timesteps = int(np.ceil(ts / dt))
        for i in tqdm(range(N_timesteps)):
            if t > ts + 1e-5:
                break
            # SSP-RK3 Time Integration
            # Step 1
            X1 = X + np.cos(np.pi * Y) * np.sin(np.pi * X) * dt
            Y1 = Y - np.cos(np.pi * X) * np.sin(np.pi * Y) * dt
            # Step 2
            X1 = (
                0.75 * X
                + X1 * 0.25
                + np.cos(np.pi * Y1) * np.sin(np.pi * X1) * dt * 0.25
            )
            Y1 = (
                0.75 * Y
                + Y1 * 0.25
                - np.cos(np.pi * X1) * np.sin(np.pi * Y1) * dt * 0.25
            )
            # Step 3
            X = (
                (1 / 3) * X
                + X1 * (2 / 3)
                + np.cos(np.pi * Y1) * np.sin(np.pi * X1) * dt * (2 / 3)
            )
            Y = (
                (1 / 3) * Y
                + Y1 * (2 / 3)
                - np.cos(np.pi * X1) * np.sin(np.pi * Y1) * dt * (2 / 3)
            )
            t = t + dt
            if t > ts + 1e-5 and t < ts + dt - 1e-5:  # make final t match nn
                dt = dt - (t - ts)
                t = ts

        # Exact solution plot
        grid = griddata(
            np.transpose(np.vstack([X, Y])), mask, (X0, Y0), method="nearest"
        )

        # Update Energy spectrum with ML and exact
        image = self.gui._model.models[0].image[0, 0].cpu().numpy()
        npix = np.shape(image)[0]

        kfreq = np.fft.fftfreq(npix) * npix
        kfreq2D = np.meshgrid(kfreq, kfreq)
        knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
        knrm = knrm.flatten()
        kbins = np.arange(0.5, npix // 2 + 1, 1.0)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])

        # ML
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2
        fourier_amplitudes = fourier_amplitudes.flatten()

        Abins, _, _ = stats.binned_statistic(
            knrm, fourier_amplitudes, statistic="mean", bins=kbins
        )

        Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

        self.energy_ax.clear()
        self.energy_ax.loglog(kvals, Abins, label="ML")

        # Exact
        image = grid
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2
        fourier_amplitudes = fourier_amplitudes.flatten()

        Abins, _, _ = stats.binned_statistic(
            knrm, fourier_amplitudes, statistic="mean", bins=kbins
        )

        Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

        self.energy_ax.loglog(kvals, Abins, label="Exact")
        self.energy_ax.set_title("Energy Spectrum")
        self.energy_ax.set_xlabel("$k$")
        self.energy_ax.legend()
        self.energy_ax.set_ylim((1e1, 1e10))

        self.energy_fig.canvas.draw_idle()

        return grid

    @AnalyticalBase.figure
    def energy_spectrum_graph(
        self, width: float, height: float
    ) -> Tuple[FigureCanvasQTAgg, str]:
        """
        Creates the energy spectrum graph used for the tg-vortex simulation.
        It is decorated with AnalyticalBase.figure to be discovered as a graph
        that should be added to the viewer.

        :param width: The width of the FigureCanvas
        :param height: The height of the FigureCanvas
        :return: A tuple containing the FigureCanvasQTAgg object, and the display name
        for the graph in the viewer.
        """
        self.energy_graph = FigureCanvasQTAgg(Figure(figsize=(width, height)))
        self.energy_fig = self.energy_graph.figure
        # left, bottom, width, height
        self.energy_ax = self.energy_fig.add_axes([0.1, 0.15, 0.85, 0.75])

        image = self.gui._model.models[0].image[0, 0].cpu().numpy()
        npix = np.shape(image)[0]
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2

        kfreq = np.fft.fftfreq(npix) * npix
        kfreq2D = np.meshgrid(kfreq, kfreq)
        knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()

        kbins = np.arange(0.5, npix // 2 + 1, 1.0)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        Abins, _, _ = stats.binned_statistic(
            knrm, fourier_amplitudes, statistic="mean", bins=kbins
        )

        Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        # image 1
        self.energy_ml_line = self.energy_ax.loglog(kvals, Abins)
        self.energy_ml_line = self.energy_ml_line[0]
        self.energy_ax.set_title("Energy Spectrum")
        self.energy_ax.set_xlabel("$k$")
        self.energy_ax.set_ylim((1e1, 1e10))
        return (self.energy_graph, "Energy Spectrum")
