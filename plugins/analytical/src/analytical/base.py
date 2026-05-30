# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from professor.vela.gui import GUI, Sliders
from professor.vela.model import Model
from professor.vela.config import Config


class AnalyticalBase(ABC):
    """
    AnalyticalBase is the abstract base class that should be implemented when
    when it is possible and desired to compare an ML model prediction to a
    an approximate/exact solution that is calculable.

    Methods marked as "@abstractmethod" need to be implemented in the child class.
    """

    @abstractmethod
    def __init__(self, gui: GUI, config: Config, model: Model, sliders: Sliders):
        """
        Instantiates three class objects for referencing the gui, model, and slider
        objects. The implementation class has access to these objects.

        This method needs to be reimplemented with a call to the super().__init__ method
        like so:

        ```
        class ChildClass(AnalyticalBase):
            def __init__(self, gui, model, sliders) -> None:
                super().__init__(gui, model, sliders)
        ```

        :param gui: Reference to the CreateGUI object for the viewer
        :param config: Reference to the Config object for the YAML entries
        :param model: Reference to the InitModel object for the model params
        :param sliders: Reference to the Sliders object for slider access
        """
        self.gui = gui
        self.config = config
        self.model = model
        self.sliders = sliders

    @abstractmethod
    def update(self) -> None:
        """
        Update is what should occur each time one of the sliders move.

        For example, if it is desired that a graph is changed everytime the sliders
        update, then that logic should be implemented in this method.

        :param self AnalyticalBase: Reference to the current AnalyticalBase class
        :return: None
        """

    @abstractmethod
    def evaluate(self, x_pixels: int, y_pixels: int) -> np.ndarray:
        """
        Evaluate should return an exact/approximate image solution for the parameters. This
        method would contain the logic necessary for calculating an exact solution for a problem.

        :return: np.ndarray image that should match the x_pixles/y_pixels of ML image.
        """

    @classmethod
    def figure(cls, func: Callable) -> Callable:
        """
        Figure is a decorator used for denoting that a method in the implementation class
        returns a Matplotlib Graph.

        ex:

        ```
        @AnalyticalBase.figure
        def matplotlib_graph(
            width: float, height: float
        ) -> Tuple[FigureCanvasQTAgg, str]:
            self.graph = FigureCanvasQTAgg(Figure(figsize=(width, height)))
            self.graph_fig = self.graph.figure
            # left, bottom, width, height
            self.fig_ax = self.graph_fig.add_axes([0.1, 0.15, 0.85, 0.75])
            ...
            # implement the figure here
            return (self.graph, "My Super Cool Graph Name")
        ```

        As arguments, the decorated function should take in a `width` and `height`
        value corresponding to how big the figure should be.

        The decorated function should return arguments in the form of:

        Tuple[matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg, str]

        Where the first item is the actual FigureCanvas of the Matplotlib graph and
        the second item is the desired display name for the graph.

        :param cls AbstractBase: Reference to the static AbstractBase class
        :param func: The function that is being wrapped
        :return: The result of the wrapped function
        """

        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        wrapper._figure_wrapped = True
        return wrapper

    @classmethod
    def is_figure_decorated(cls, func: Callable) -> bool:
        """
        Return True is a method has been decorated with @figure. Return False
        if not.

        :param cls AbstractBase: Reference to the static AbstractBase class
        :param func: The function of interest.
        :return: True if the method has been decorated with @figure, False if not.
        """
        return bool(getattr(func, "_figure_wrapped", False))
