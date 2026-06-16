import argparse
import os
import h5py
import torch
import numpy as np
from dataclasses import dataclass
import dash  # type: ignore
import dash_bootstrap_components as dbc  # type: ignore
from dash import Input, Output, State, dcc, html, clientside_callback  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import plotly.graph_objs as go  # type: ignore
from typing import Any, Iterable


class ProfessorHelper:
    """
    Helper class for launching and evaluating professor models
    """
    def __init__(self) -> None:
        self._fields: list[str] = []
        self._sliders: dict[str, Any] = {}
        self._device: torch.Device = ""
        self._model: torch.nn.modules.module.Module | None = None

    def load(self, config_fname: str) -> tuple[list[str], dict[str, Any]]:
        """
        Load a professor model from a yaml-format configuration file

        Args:
            config_name: Configuration filename

        Returns:
            List of model prediction fields, dictionary of slider inputs 
        """
        from professor.vela import config, model

        print(f"Loading model: {config_fname}")
        c = config.Config(os.path.expanduser(config_fname))
        self._fields = c.fields
        self._sliders = c.sliders
        m = model.PyTorchModel(c)
        self._device = m._get_checkpoint_device("0")
        self._model = m.models[0].ref
        print("Done!")
        return self._fields, self._sliders

    def run(self, model_args: Iterable[float]) -> np.ndarray:
        """
        Evaluate the loaded professor model

        Args:
            model_args: list of model inputs

        Returns:
            np.ndarray of model results
        """
        if self._model is None:
            return np.zeros(0)

        with torch.no_grad():
            tmp = np.array(model_args)
            x = torch.Tensor(np.reshape(tmp, (1, -1, 1, 1))).half().to(self._device)
            y = self._model(x).detach().to("cpu").numpy()
            return y


prof_model = ProfessorHelper()


@dataclass(frozen=True)
class SliderConfig:
    name: str
    default: float = 0.0
    min: float = 0.0
    max: float = 1.0
    step: float = 0.01


@dataclass(frozen=True)
class DropdownConfig:
    name: str
    options: list[str]


@dataclass(frozen=True)
class EntryConfig:
    name: str
    value: str


def build_slider(config: SliderConfig) -> dbc.Row:
    """
    Build a row that contains a label and slider
    """
    tooltip = {
        "placement": "bottom",
        # "always_visible": True
    }

    w = dcc.Slider(
        config.min,
        config.max,
        value=config.default,
        step=config.step,
        vertical=False,
        tooltip=tooltip,
        id=f"slider_{config.name}",
    )

    columns = [dbc.Col(html.P(config.name)), dbc.Col(w)]
    return dbc.Row(columns, className="border-bottom pb-3 mb-3")


def build_dropdown(config: DropdownConfig) -> dbc.Row:
    """
    Build a row that contains a label and dropdown
    """
    w = dbc.Select(id=f"dropdown_{config.name}", options=config.options, value=config.options[0])

    columns = [dbc.Col(html.P(config.name)), dbc.Col(w)]
    return dbc.Row(columns, className="border-bottom pb-3 mb-3")


def build_entry(config: EntryConfig) -> dbc.Row:
    """
    Build a row that contains a label and entry box
    """
    w = dbc.Input(id=f"entry_{config.name}", type="text", placeholder="(empty)", value=config.value)

    columns = [dbc.Col(html.P(config.name)), dbc.Col(w)]
    return dbc.Row(columns, className="border-bottom pb-3 mb-3")


def build_dummy_figure() -> go.Figure:
    """
    Build an empty placeholder figure
    """
    dummy_figure = go.Figure()
    axis_def = {"showline": False, "zeroline": False, "showgrid": False, "range": (0, 1)}

    dummy_figure.update_layout(
        clickmode="none",
        margin=dict(l=20, r=20, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=True,
        font_color="rgba(0,0,0,0)",
        xaxis=axis_def,
        yaxis=axis_def,
    )
    return dummy_figure


def get_color_scale(img: np.ndarray, saturation: float) -> tuple[float, float]:
    """
    Get the desired color range for an input array and saturation
    """
    cmin = np.amin(img)
    cmax = np.amax(img)
    if saturation > 1.01:
        cmid = 0.5 * (cmin + cmax)
        cr = 0.5 * (cmax - cmin) / saturation
        cmin = cmid - cr
        cmax = cmid + cr

    return cmin, cmax


def get_style_kwargs(light_mode: bool = False):
    """
    Define common style arguments for figures

    Returns:
        dict: Dictionary of style kwargs
    """
    axis_color = "white"
    marker_color = "white"
    font_color = "white"
    font_family = "Open Sans"
    axis_font_size = 14
    title_font_size = 14
    if light_mode:
        axis_color = "black"
        marker_color = "black"
        font_color = "black"

    style_kwargs = {
        "clickmode": "none",
        "margin": dict(l=20, r=20, t=50, b=50),
        "autosize": True,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font_color": font_color,
        "title_font_color": font_color,
        "xaxis_title_font_color": font_color,
        "yaxis_title_font_color": font_color,
        "font_family": font_family,
        "title_font_family": font_family,
        "xaxis_title_font_family": font_family,
        "yaxis_title_font_family": font_family,
        "font_size": axis_font_size,
        "title_font_size": title_font_size,
        "xaxis_title_font_size": axis_font_size,
        "yaxis_title_font_size": axis_font_size,
    }

    return axis_color, marker_color, font_color, style_kwargs


def dash_plot_2D(
    img: np.ndarray, label: str = "Title", colorscale: str = "Turbo", saturation: float = 1, light_mode: bool = True
) -> go.Figure:
    """
    Render a 2D plotly figure for a given input array
    """
    cmin, cmax = get_color_scale(img, saturation)
    axis_color, marker_color, font_color, style_kwargs = get_style_kwargs(light_mode)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=np.transpose(img),
            colorscale=colorscale,
            name="image-layer",
            zsmooth="fast",
            zmin=cmin,
            zmax=cmax,
            colorbar={"title": label},
        )
    )

    xaxis_def = {
        "mirror": True,
        "ticks": "outside",
        "showline": True,
        "zeroline": False,
        "showgrid": False,
        "linecolor": axis_color,
        "tickcolor": axis_color,
        "title": "X",
    }
    yaxis_def = {
        "mirror": True,
        "ticks": "outside",
        "showline": True,
        "zeroline": False,
        "showgrid": False,
        "linecolor": axis_color,
        "tickcolor": axis_color,
        "title": "Y",
    }

    fig_kwargs = {"showlegend": True}

    # Update the figure
    fig.update_layout(xaxis=xaxis_def, yaxis=yaxis_def, **fig_kwargs, **style_kwargs)

    return fig


def dash_plot_2D_planes(
    img: np.ndarray, label: str = "Title", colorscale: str = "Turbo", saturation: float = 1, light_mode: bool = True
) -> go.Figure:
    """
    Render a series of 2D plotly figures that show slices through an array
    """
    img = np.array(img)
    cmin, cmax = get_color_scale(img, saturation)
    axis_color, marker_color, font_color, style_kwargs = get_style_kwargs(light_mode)

    N = np.shape(img)
    ii = N[0] // 2
    jj = N[1] // 2
    kk = N[2] // 2

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Xmid", "Ymid", "Zmid", ""))
    fig.add_trace(
        go.Heatmap(
            z=np.transpose(img[ii, :, :]),
            colorscale=colorscale,
            name="image-layer",
            zsmooth="fast",
            zmin=cmin,
            zmax=cmax,
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=np.transpose(img[:, jj, :]),
            colorscale=colorscale,
            name="image-layer",
            zsmooth="fast",
            zmin=cmin,
            zmax=cmax,
            showscale=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=np.transpose(img[:, :, kk]),
            colorscale=colorscale,
            name="image-layer",
            zsmooth="fast",
            zmin=cmin,
            zmax=cmax,
            colorbar={"title": label},
        ),
        row=2,
        col=1,
    )

    # Setup axes
    xaxis_def = {
        "mirror": True,
        "ticks": "outside",
        "showline": True,
        "zeroline": False,
        "showgrid": False,
        "linecolor": axis_color,
        "tickcolor": axis_color,
    }
    yaxis_def = {
        "mirror": True,
        "ticks": "outside",
        "showline": True,
        "zeroline": False,
        "showgrid": False,
        "linecolor": axis_color,
        "tickcolor": axis_color,
    }

    fig_kwargs = {"showlegend": True}

    # Update the figure
    fig.update_layout(xaxis=xaxis_def, yaxis=yaxis_def, **fig_kwargs, **style_kwargs)

    return fig


def dash_plot_3D(
    img: np.ndarray,
    label: str = "Title",
    colorscale: str = "Turbo",
    option_3d: str = "Volume",
    saturation: float = 1,
    light_mode: bool = True,
) -> go.Figure:
    """
    Render a 3D plotly figure for a given input array
    """
    img = np.array(img)
    cmin, cmax = get_color_scale(img, saturation)
    axis_color, marker_color, font_color, style_kwargs = get_style_kwargs(light_mode)

    N = np.shape(img)
    x = np.linspace(0, 1, N[0])
    y = np.linspace(0, 1, N[1])
    z = np.linspace(0, 1, N[2])

    fig_data = []

    if option_3d == "Volume":
        G = np.meshgrid(x, y, z, indexing="ij")
        fig_data.append(
            go.Volume(
                x=G[0].flatten(),
                y=G[1].flatten(),
                z=G[2].flatten(),
                value=img.flatten(),
                opacity=0.1,
                surface_count=5,
                colorscale=colorscale,
                name="image-layer",
                colorbar={"title": label},
            )
        )
    else:
        ii = N[0] // 2
        jj = N[1] // 2
        kk = N[2] // 2
        xy, xz = np.meshgrid(y, z, indexing="ij")
        xx = np.zeros(np.shape(xy)) + 0.5
        fig_data.append(
            go.Surface(x=xx, y=xy, z=xz, surfacecolor=img[ii, :, :], colorscale=colorscale, cmin=cmin, cmax=cmax)
        )

        yx, yz = np.meshgrid(x, z, indexing="ij")
        yy = np.zeros(np.shape(xy)) + 0.5
        fig_data.append(
            go.Surface(
                x=yx,
                y=yy,
                z=yz,
                surfacecolor=img[:, jj, :],
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=False,
            )
        )

        zx, zy = np.meshgrid(x, y, indexing="ij")
        zz = np.zeros(np.shape(xy)) + 0.5
        fig_data.append(
            go.Surface(
                x=zx,
                y=zy,
                z=zz,
                surfacecolor=img[:, :, kk],
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=False,
            )
        )

    # Setup axes
    scene = {
        "xaxis_title": "X",
        "yaxis_title": "Y",
        "zaxis_title": "Z",
        "xaxis": {"range": [0, 1], "gridcolor": axis_color},
        "yaxis": {"range": [0, 1], "gridcolor": axis_color},
        "zaxis": {"range": [0, 1], "gridcolor": axis_color},
    }

    fig_kwargs = {"showlegend": True, "scene": scene}

    # Update the figure
    fig = go.Figure(data=fig_data)
    fig.update_layout(**fig_kwargs, **style_kwargs)

    return fig


def build_application(fields: list[str] = [], sliders: list[SliderConfig] = []) -> dash.Dash:
    """
    Build the ploty dash application
    """
    # Configure global app behavior
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])

    # Build the navbar
    titlebar = html.A(
        dbc.Row(
            [
                dbc.Col(dbc.NavbarBrand("Professor Visualization Tool")),
            ],
            align="center",
        ),
        target="_blank",
        style={"textDecoration": "none"},
    )

    color_mode_switch = html.Span(
        [
            dbc.Label(className="fa fa-moon", html_for="light-mode-switch", style={"marginRight": "5px"}),
            dbc.Switch(
                id="light-mode-switch",
                value=False,
                className="d-inline-block ms-1",
                persistence=True,
                style={"marginBottom": "0px"},
            ),
            dbc.Label(className="fa fa-sun", html_for="light-mode-switch", style={"marginLeft": "5px"}),
        ]
    )
    color_switch = dbc.NavbarBrand([color_mode_switch])

    navbar = dbc.Navbar(
        dbc.Container([titlebar, color_switch], fluid=True),
        color="dark",
        dark=True,
    )

    # Build the sidebar
    config_rows = [
        build_dropdown(DropdownConfig(name="Field", options=fields)),
        build_dropdown(DropdownConfig(name="Colorscale", options=["Turbo", "Viridis", "Inferno", "gray", "RdBu"])),
        build_dropdown(DropdownConfig(name="Rendering", options=["2D", "ThreeSlice", "Volume"])),
        build_slider(SliderConfig(name="Saturation", default=1, min=1, max=10)),
    ]
    config_inputs = [
        Input("dropdown_Field", "value"),
        Input("dropdown_Colorscale", "value"),
        Input("dropdown_Rendering", "value"),
        Input("slider_Saturation", "value"),
    ]
    etc_rows = [dbc.Row(html.Button("Download Data", id="download-button", n_clicks=0))]

    slider_rows = [build_slider(s) for s in sliders]
    slider_inputs = [Input(f"slider_{s.name}", "value") for s in sliders]

    sidebar = html.Div(
        [
            html.H4("Config"),
            html.Hr(),
            html.Div(dbc.Container(config_rows, fluid=True)),
            html.H4("Parameters"),
            html.Hr(),
            html.Div(dbc.Container(slider_rows + etc_rows, fluid=True)),
        ],
        style={
            "height": "90vh",
            "overflowY": "scroll",
            "position": "fixed",
            "border": "2px solid #808080",
            "padding": "15px",
        },
    )

    # Build the figure graph
    graph_config = {
        "toImageButtonOptions": {
            "format": "png",
            "height": 480,
            "width": 640,
            "scale": 4,
        }
    }
    graph_style = {"width": "90vh", "height": "90vh"}

    figure_graph = html.Div(
        dcc.Graph(
            id="figure-graph",
            animate=False,
            responsive=True,
            config=graph_config,
            style=graph_style,
            figure=build_dummy_figure(),
        )
    )

    # Build the app layout
    data_store = dcc.Store(id="model-results", storage_type="local", data={})
    download_file = dcc.Download(id="download-file")
    win_content = dbc.Row([dbc.Col(sidebar, width=4), dbc.Col(figure_graph, width=8)], justify="center")
    win = dbc.Container(win_content, fluid=True)
    app.layout = html.Div(id="interface-container", children=[data_store, download_file, navbar, win])

    # Build app callbacks
    @app.callback(Output("model-results", "data"), slider_inputs)
    def update_model(*inputs: float):
        tmp = prof_model.run(inputs)
        res = {}
        for ii, k in enumerate(fields):
            res[k] = np.squeeze(tmp[0, ii, ...])
        return res

    @app.callback(
        Output("figure-graph", "figure"),
        Input("model-results", "data"),
        Input("light-mode-switch", "value"),
        config_inputs,
    )
    def render_figure(
        data: dict[str, np.ndarray],
        light_mode: bool,
        field: str,
        colorscale: str,
        rendering_style: str,
        saturation: float,
    ) -> go.Figure:
        img = data.get(field)
        if img is None:
            return dash.no_update

        if len(np.shape(img)) == 2:
            return dash_plot_2D(img, label=field, colorscale=colorscale, saturation=saturation, light_mode=light_mode)
        elif rendering_style == "2D":
            return dash_plot_2D_planes(
                img, label=field, colorscale=colorscale, saturation=saturation, light_mode=light_mode
            )
        else:
            return dash_plot_3D(
                img,
                label=field,
                colorscale=colorscale,
                saturation=saturation,
                option_3d=rendering_style,
                light_mode=light_mode,
            )

    @app.callback(
        Output("download-file", "data"),
        Input("download-button", "n_clicks"),
        State("model-results", "data"),
        prevent_initial_call=True,
    )
    def download_data(nclicks: int, data: dict[str, np.ndarray]):
        with h5py.File("model_data.h5", "w") as f:
            for k, v in data.items():
                f.create_dataset(
                    k,
                    data=np.array(v).astype(np.float32),
                    dtype=np.float32,
                )
        return dcc.send_file("model_data.h5")

    clientside_callback(
        """
        (switchOn) => {
            document.documentElement.setAttribute('data-bs-theme', switchOn ? 'light' : 'dark');
            return window.dash_clientside.no_update
        }
        """,
        Output("light-mode-switch", "id"),
        Input("light-mode-switch", "value"),
    )

    return app


def main():
    """
    Parse user arguments, then build and launch the application
    """
    parser = argparse.ArgumentParser(
        prog="prof-dash-gui",
        description="Professor dash visualization",
    )
    parser.add_argument("config", type=str, help="Model configuration file")
    parser.add_argument(
        "-p",
        "--port",
        default=8888,
        type=int,
        help="Target port",
    )
    args = parser.parse_args()

    fields, slider_config = prof_model.load(args.config)
    sliders = []
    for k, s in slider_config.items():
        sliders.append(SliderConfig(name=k, default=s["initial_value"], min=s["lower_bound"], max=s["upper_bound"]))

    app = build_application(fields, sliders)
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
