# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from typing import Callable, Self, Any, Iterable, Dict, Tuple
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from napari import Viewer

try:
    from qtpy.QtWidgets import QGridLayout, QTabWidget, QWidget
except ImportError:
    # TODO: Make sure that this works with pyside as well
    from PySide6.QtWidgets import QGridLayout, QTabWidget, QWidget

from professor.vela.gui import GUI, Sliders
from professor.vela.model import Model
from professor.vela.config import Config
from professor.vela.config import Keys
from professor.vela.constants import GUI_INSTANCE_NAME
from professor.vela.utils import get_object_reference


class AnalyticalBase(ABC):
    """
    AnalyticalBase is the abstract base class that should be implemented when
    when it is possible and desired to compare an ML model prediction to a
    an approximate/exact solution that is calculable.

    Methods marked as "@abstractmethod" need to be implemented in the child class.
    """

    @abstractmethod
    def __init__(self, gui: GUI, config: Config, model: Model, sliders: Sliders) -> None:
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
        self.gui: GUI = gui
        self.config: Config = config
        self.model: Model = model
        self.sliders: Sliders = sliders

    @abstractmethod
    def update(self) -> None:
        """
        Update is what should occur each time one of the sliders move.

        For example, if it is desired that a graph is changed everytime the sliders
        update, then that logic should be implemented in this method.

        :param self AnalyticalBase: Reference to the current AnalyticalBase class
        :return: None
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

        def wrapper(self: Self, *args: Iterable[Any], **kwargs: Dict[str, Any]) -> Tuple[FigureCanvasQTAgg, str]:
            return func(self, *args, **kwargs)

        wrapper._figure_wrapped = True  # type: ignore
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


class SimpleWidget(QWidget):
    YAML_KEY = "custom-widget"
    GRAPH_WIDTH = 4.5
    GRAPH_HEIGHT = 2

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer: Viewer = napari_viewer
        self.gui = self.viewer.__dict__[GUI_INSTANCE_NAME]  # type: ignore
        self.model: Model = self.gui._model
        self.config: Config = self.gui._config
        self.sliders: Sliders = self.gui.sliders
        self.plugin: dict[str, Any] = self.config.plugins[self.YAML_KEY]

        self.tabs: QTabWidget | None

        self.setLayout(QGridLayout())
        self._instantiate_analytical_subclass()
        self._get_graphs()

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
            layout = self.layout()
            if layout is None:
                raise Exception("Layout was not properly constructed")
            # Note: there appears to be an issue with the addWidget overloads
            layout.addWidget(self.tabs, row, col, row_span, col_span)  # type: ignore
            row += 1

    def _get_graphs(self) -> None:
        """
        Get all the methods decorated with @AnalyticalBase.figure in the implementation
        class and add the returned graphs as a tab widget.

        :return: None
        """
        # Get all the callable method names as strings
        figures = [m for m in dir(self.subclass) if not m.startswith("__") and callable(getattr(self.subclass, m))]

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
