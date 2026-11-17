from __future__ import annotations

import argparse
import multiprocessing
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any

import dash  # type: ignore
import dash_bootstrap_components as dbc  # type: ignore
import numpy as np
import plotly.graph_objs as go  # type: ignore
from dash import ALL, MATCH, Input, Output, State, clientside_callback, dcc, html  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

from professor.file_io import (
    RESULT_SIZE_LIMIT,
    build_config_payload_from_path,
    build_config_payload_from_text,  # noqa: F401
    build_config_payload_from_upload,
    write_hdf5,
)
from professor.gui.model_cache import ResultCache, decode_array, encode_array, model_cache, result_cache  # noqa: F401

SLIDER_MARK_COUNT = 5
SLIDER_MARK_SIG_DIGITS = 6


@dataclass(frozen=True)
class SliderConfig:
    name: str
    default: float = 0.0
    min: float = 0.0
    max: float = 1.0
    step: float = 0.001


@dataclass(frozen=True)
class DropdownConfig:
    name: str
    options: list[str]


@dataclass(frozen=True)
class EntryConfig:
    name: str
    value: str


def _format_slider_mark(value: float) -> str:
    label = f"{value:.{SLIDER_MARK_SIG_DIGITS}g}"
    if label == "-0":
        return "0"
    return label


def _build_slider_marks(slider: SliderConfig) -> dict[float, str]:
    if slider.min == slider.max:
        return {slider.min: _format_slider_mark(slider.min)}

    raw_values = np.linspace(slider.min, slider.max, SLIDER_MARK_COUNT)
    marks = {}
    for value in raw_values:
        mark_value = float(f"{value:.{SLIDER_MARK_SIG_DIGITS}g}")
        marks[mark_value] = _format_slider_mark(float(value))
    return marks


def build_slider(slider: SliderConfig, instance_id: str, component_type: str = "model-slider") -> dbc.Row:
    """
    Build a row that contains a label and slider.
    """
    tooltip = {"placement": "bottom"}
    widget = dcc.Slider(
        slider.min,
        slider.max,
        value=slider.default,
        step=slider.step,
        marks=_build_slider_marks(slider),
        vertical=False,
        tooltip=tooltip,
        id={"type": component_type, "instance_id": instance_id, "name": slider.name},
    )
    columns = [dbc.Col(html.P(slider.name)), dbc.Col(widget)]
    return dbc.Row(columns, className="border-bottom pb-3 mb-3")


def build_dropdown(config: DropdownConfig, component_type: str, instance_id: str) -> dbc.Row:
    """
    Build a row that contains a label and dropdown.
    """
    widget = dbc.Select(
        id={"type": component_type, "instance_id": instance_id},
        options=config.options,
        value=config.options[0],
    )
    columns = [dbc.Col(html.P(config.name)), dbc.Col(widget)]
    return dbc.Row(columns, className="border-bottom pb-3 mb-3")


def build_entry(config: EntryConfig) -> dbc.Row:
    """
    Build a row that contains a label and entry box.
    """
    widget = dbc.Input(id=f"entry_{config.name}", type="text", placeholder="(empty)", value=config.value)
    columns = [dbc.Col(html.P(config.name)), dbc.Col(widget)]
    return dbc.Row(columns, className="border-bottom pb-3 mb-3")


def build_dummy_figure() -> go.Figure:
    """
    Build an empty placeholder figure.
    """
    dummy_figure = go.Figure()
    axis_def = {"showline": False, "zeroline": False, "showgrid": False, "range": (0, 1)}

    dummy_figure.add_annotation(
        text="Loading model (this may take up to a few minutes)...",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 24, "color": "red"},
        align="center",
    )

    dummy_figure.update_layout(
        clickmode="none",
        margin={"l": 20, "r": 20, "t": 50, "b": 50},
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
    Get the desired color range for an input array and saturation.
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
    Define common style arguments for figures.
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
        "margin": {"l": 20, "r": 20, "t": 50, "b": 50},
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
    Render a 2D plotly figure for a given input array.
    """
    cmin, cmax = get_color_scale(img, saturation)
    axis_color, _marker_color, _font_color, style_kwargs = get_style_kwargs(light_mode)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=img,
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
    fig.update_layout(xaxis=xaxis_def, yaxis=yaxis_def, **fig_kwargs, **style_kwargs)

    return fig


def dash_plot_2D_planes(
    img: np.ndarray, label: str = "Title", colorscale: str = "Turbo", saturation: float = 1, light_mode: bool = True
) -> go.Figure:
    """
    Render a series of 2D plotly figures that show slices through an array.
    """
    img = np.array(img)
    cmin, cmax = get_color_scale(img, saturation)
    axis_color, _marker_color, _font_color, style_kwargs = get_style_kwargs(light_mode)

    shape = np.shape(img)
    ii = shape[0] // 2
    jj = shape[1] // 2
    kk = shape[2] // 2

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Xmid", "Ymid", "Zmid", ""))
    fig.add_trace(
        go.Heatmap(
            z=img[ii, :, :],
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
            z=img[:, jj, :],
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
            z=img[:, :, kk],
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
    Render a 3D plotly figure for a given input array.
    """
    img = np.array(img)
    cmin, cmax = get_color_scale(img, saturation)
    axis_color, _marker_color, _font_color, style_kwargs = get_style_kwargs(light_mode)

    shape = np.shape(img)
    x = np.linspace(0, 1, shape[0])
    y = np.linspace(0, 1, shape[1])
    z = np.linspace(0, 1, shape[2])

    fig_data = []

    if option_3d == "Volume":
        grid = np.meshgrid(x, y, z, indexing="ij")
        fig_data.append(
            go.Volume(
                x=grid[0].flatten(),
                y=grid[1].flatten(),
                z=grid[2].flatten(),
                value=img.flatten(),
                opacity=0.1,
                surface_count=5,
                colorscale=colorscale,
                name="image-layer",
                colorbar={"title": label},
            )
        )
    else:
        ii = shape[0] // 2
        jj = shape[1] // 2
        kk = shape[2] // 2
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

    scene = {
        "xaxis_title": "X",
        "yaxis_title": "Y",
        "zaxis_title": "Z",
        "xaxis": {"range": [0, 1], "gridcolor": axis_color},
        "yaxis": {"range": [0, 1], "gridcolor": axis_color},
        "zaxis": {"range": [0, 1], "gridcolor": axis_color},
    }

    fig_kwargs = {"showlegend": True, "scene": scene}
    fig = go.Figure(data=fig_data)
    fig.update_layout(**fig_kwargs, **style_kwargs)

    return fig


def build_navbar() -> dbc.Navbar:
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
    app_menu = dbc.DropdownMenu(
        [
            dbc.DropdownMenuItem("Load model", id="load-model-button", n_clicks=0),
            dbc.DropdownMenuItem("Download data", id="download-data-menu-button", n_clicks=0),
        ],
        label="Menu",
        align_end=True,
        in_navbar=True,
    )
    nav_controls = html.Div([color_switch, app_menu], className="d-flex align-items-center gap-2")

    return dbc.Navbar(
        dbc.Container([titlebar, nav_controls], fluid=True),
        color="dark",
        dark=True,
    )


def build_load_model_modal(initial_status: Any = None, is_open: bool = False, app_type: str = "local") -> dbc.Modal:

    modal_rows = []
    if app_type == "server":
        modal_rows.append(
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Upload(
                            id="config-upload",
                            children=html.Div(
                                [
                                    "Drag and drop or ",
                                    html.Span(
                                        "select",
                                        style={
                                            "color": "#0d6efd",
                                            "textDecoration": "underline",
                                            "cursor": "pointer",
                                        },
                                    ),
                                    " a YAML config or bundle",
                                ]
                            ),
                            multiple=False,
                            className="border rounded p-4 text-center",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Input(
                                id="server-config-path",
                                type="text",
                                placeholder="Server-side config path",
                            ),
                            dbc.Button("Load Path", id="load-path-button", n_clicks=0, className="mt-2"),
                        ],
                        md=6,
                    ),
                ],
                className="g-3",
            )
        )

    elif app_type == "local":
        modal_rows.append(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Input(
                                id="server-config-path",
                                type="text",
                                placeholder="Server-side config path",
                            )
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Button("Select Model", id="select-path-button", n_clicks=0, className="mt-2"),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Button("Load Model", id="load-path-button", n_clicks=0, className="mt-2"),
                        ],
                        md=6,
                    ),
                ],
                className="g-3",
            )
        )

    else:
        raise ValueError(f"Unrecognized app type: {app_type}")

    modal_rows.append(dbc.Row(html.Div(id="landing-status", className="mt-3", children=initial_status)))

    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Load Model Config")),
            dbc.ModalBody(dbc.Container(modal_rows, fluid=True)),
            dbc.ModalFooter(dbc.Button("Close", id="close-load-model-modal", n_clicks=0)),
        ],
        id="load-model-modal",
        is_open=is_open,
        size="lg",
    )


def build_model_panel(payload: dict[str, Any]) -> html.Div:
    instance_id = payload["instance_id"]
    fields = payload["fields"]
    sliders = [SliderConfig(**slider) for slider in payload["sliders"]]

    config_rows = [
        build_dropdown(DropdownConfig(name="Field", options=fields), "field-dropdown", instance_id),
        build_dropdown(
            DropdownConfig(name="Colorscale", options=["Turbo", "Viridis", "Inferno", "gray", "RdBu"]),
            "colorscale-dropdown",
            instance_id,
        ),
        build_dropdown(
            DropdownConfig(name="Rendering", options=["2D", "ThreeSlice", "Volume"]),
            "rendering-dropdown",
            instance_id,
        ),
        build_slider(SliderConfig(name="Saturation", default=1, min=1, max=10), instance_id, "plot-slider"),
    ]
    slider_rows = [build_slider(slider, instance_id) for slider in sliders]

    sidebar = html.Div(
        [
            html.H4(payload["model_name"]),
            html.Div(payload["source"], className="small text-muted text-break"),
            html.Hr(),
            html.H5("Config"),
            html.Div(dbc.Container(config_rows, fluid=True)),
            html.H5("Parameters"),
            html.Hr(),
            html.Div(dbc.Container(slider_rows, fluid=True)),
        ],
        style={
            "height": "90vh",
            "overflowY": "scroll",
            "position": "fixed",
            "border": "2px solid #808080",
            "padding": "15px",
        },
    )

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
            id={"type": "figure-graph", "instance_id": instance_id},
            animate=False,
            responsive=True,
            config=graph_config,
            style=graph_style,
            figure=build_dummy_figure(),
        )
    )

    return html.Div(
        [
            dcc.Store(id={"type": "model-config", "instance_id": instance_id}, storage_type="memory", data=payload),
            dcc.Store(id={"type": "model-results", "instance_id": instance_id}, storage_type="memory", data={}),
            dcc.Store(id={"type": "plot-data", "instance_id": instance_id}, storage_type="memory", data={}),
            dbc.Container(
                dbc.Row([dbc.Col(sidebar, width=4), dbc.Col(figure_graph, width=8)], justify="center"),
                fluid=True,
            ),
        ]
    )


def build_status(message: str, color: str = "danger") -> dbc.Alert:
    return dbc.Alert(message, color=color, dismissable=True)


def _select_file_path() -> str:
    """
    Request a configuration filename from the user via a pop up search window
    """
    import tkinter as tk
    from tkinter import filedialog

    # Open the tk window and push it to the top layer
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.focus_force()

    # Ask the user for the filename
    filetypes = [
        ("Config (*.yml, *.yaml)", "*.yml *.yaml"),
        ("Archive (*.zip, *.tar, *.tar.gz)", "*.zip *.tar *.tar.gz"),
        ("All", "*"),
    ]
    file_path = filedialog.askopenfilename(parent=root, title="Select Target Model", filetypes=filetypes)
    root.destroy()
    return file_path


def _get_file_path_worker(connection: Any) -> None:
    try:
        connection.send((True, _select_file_path()))
    except Exception as exc:  # noqa: BLE001
        connection.send((False, str(exc)))
    finally:
        connection.close()


def get_file_path() -> str:
    """Open the Tk file picker in a separate process and return the chosen path.

    macOS requires Tkinter to run on the main thread, but Dash callbacks may run
    on a worker thread. The separate process gives the picker its own main thread.
    """
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(target=_get_file_path_worker, args=(child_connection,))

    try:
        process.start()
    except BaseException:
        parent_connection.close()
        child_connection.close()
        raise

    child_connection.close()
    try:
        try:
            result = parent_connection.recv()
        except EOFError:
            result = None
    finally:
        parent_connection.close()
        process.join()
        process.close()

    if result is None:
        raise RuntimeError("The file picker process exited without returning a path.")

    success, value = result
    if not success:
        raise RuntimeError(f"Could not open the file picker: {value}")
    return value


def build_application(initial_config: str | None = None, app_type: str = "local") -> dash.Dash:
    """
    Build the plotly Dash application.
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True,
    )

    def serve_layout() -> html.Div:
        session_id = str(uuid.uuid4())
        active_payload = None
        initial_panel = html.Div()
        initial_status = html.Div()
        if initial_config is not None:
            try:
                active_payload = build_config_payload_from_path(initial_config, session_id)
                initial_panel = build_model_panel(active_payload)
                initial_status = build_status(f"Loaded {active_payload['source']}", color="success")
            except Exception as exc:  # noqa: BLE001
                initial_status = build_status(str(exc))

        return html.Div(
            id="interface-container",
            children=[
                dcc.Store(id="session-id", storage_type="memory", data=session_id),
                dcc.Store(id="active-model", storage_type="memory", data=active_payload),
                dcc.Download(id="download-file"),
                build_navbar(),
                build_load_model_modal(initial_status, is_open=initial_config is None, app_type=app_type),
                html.Div(id="visualization-container", children=initial_panel),
            ],
        )

    app.layout = serve_layout

    @app.callback(
        Output("load-model-modal", "is_open"),
        Input("load-model-button", "n_clicks"),
        Input("close-load-model-modal", "n_clicks"),
        State("load-model-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_load_model_modal(load_clicks: int, close_clicks: int, is_open: bool) -> bool:
        triggered_id = dash.ctx.triggered_id
        if triggered_id == "load-model-button":
            return True
        if triggered_id == "close-load-model-modal":
            return False
        return is_open

    if app_type == "server":

        @app.callback(
            Output("active-model", "data"),
            Output("visualization-container", "children"),
            Output("landing-status", "children"),
            Input("config-upload", "contents"),
            Input("load-path-button", "n_clicks"),
            State("config-upload", "filename"),
            State("server-config-path", "value"),
            State("session-id", "data"),
            State("active-model", "data"),
            prevent_initial_call=True,
        )
        def load_config_server(
            upload_contents: str | None,
            n_clicks: int,
            upload_filename: str | None,
            server_config_path: str | None,
            session_id: str,
            active_payload: dict[str, Any] | None,
        ):
            triggered_id = dash.ctx.triggered_id
            try:
                if triggered_id == "config-upload":
                    if upload_contents is None or upload_filename is None:
                        raise ValueError("No uploaded config was provided.")
                    payload = build_config_payload_from_upload(upload_contents, upload_filename, session_id)
                else:
                    if not server_config_path:
                        raise ValueError("Enter a server-side config path before loading.")
                    payload = build_config_payload_from_path(server_config_path, session_id)
            except Exception as exc:  # noqa: BLE001
                return dash.no_update, dash.no_update, build_status(str(exc))

            if active_payload is not None:
                result_cache.release_instance(active_payload["instance_id"])
                model_cache.release_session(session_id)
            return payload, build_model_panel(payload), build_status(f"Loaded {payload['source']}", color="success")

    elif app_type == "local":

        @app.callback(
            Output("server-config-path", "value"),
            Input("select-path-button", "n_clicks"),
            prevent_initial_call=True,
        )
        def select_config_path(
            n_clicks: int,
        ):
            return get_file_path()

        @app.callback(
            Output("active-model", "data"),
            Output("visualization-container", "children"),
            Output("landing-status", "children"),
            Input("load-path-button", "n_clicks"),
            State("server-config-path", "value"),
            State("session-id", "data"),
            State("active-model", "data"),
            prevent_initial_call=True,
        )
        def load_config_local(
            n_clicks: int,
            server_config_path: str | None,
            session_id: str,
            active_payload: dict[str, Any] | None,
        ):
            try:
                if not server_config_path:
                    raise ValueError("Enter a server-side config path before loading.")
                payload = build_config_payload_from_path(server_config_path, session_id)
            except Exception as exc:  # noqa: BLE001
                return dash.no_update, dash.no_update, build_status(str(exc))

            if active_payload is not None:
                result_cache.release_instance(active_payload["instance_id"])
                model_cache.release_session(session_id)
            return payload, build_model_panel(payload), build_status(f"Loaded {payload['source']}", color="success")

    else:
        raise ValueError(f"Unrecognized app type: {app_type}")

    @app.callback(
        Output({"type": "model-results", "instance_id": MATCH}, "data"),
        Input({"type": "model-slider", "instance_id": MATCH, "name": ALL}, "value"),
        State({"type": "model-slider", "instance_id": MATCH, "name": ALL}, "id"),
        State({"type": "model-config", "instance_id": MATCH}, "data"),
        State("active-model", "data"),
    )
    def update_model(
        slider_values: list[float],
        slider_ids: list[dict[str, str]],
        config_payload: dict[str, Any],
        active_payload: dict[str, Any] | None,
    ):
        if active_payload is None:
            return dash.no_update
        if active_payload["instance_id"] != config_payload["instance_id"]:
            return dash.no_update

        values_by_name = {}
        for slider_id, value in zip(slider_ids, slider_values):
            values_by_name[slider_id["name"]] = value

        model_args = []
        for name in config_payload["slider_order"]:
            model_args.append(values_by_name[name])

        try:
            helper = model_cache.get_or_load(config_payload, config_payload["session_id"])
            tmp = helper.run(model_args)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

        if tmp.size == 1:
            return {}

        result_metadata = result_cache.put(config_payload["instance_id"], config_payload["fields"], tmp)
        return {
            "token": result_metadata["token"],
            "shape": result_metadata["shape"],
            "fields": result_metadata["fields"],
        }

    @app.callback(
        Output({"type": "plot-data", "instance_id": MATCH}, "data"),
        Input({"type": "model-results", "instance_id": MATCH}, "data"),
        Input({"type": "field-dropdown", "instance_id": MATCH}, "value"),
    )
    def update_plot_data(model_results: dict[str, Any], field: str) -> dict[str, Any]:
        if not model_results:
            return {}
        if "error" in model_results:
            return model_results
        token = model_results.get("token")
        if token is not None:
            img = result_cache.get_array(token, field)
            if img is None:
                return {"error": "Model result expired. Adjust a parameter to rerun inference."}
            encoded_img = encode_array(img)
            if len(encoded_img) > RESULT_SIZE_LIMIT:
                return {"error": f"Selected field '{field}' is too large for browser storage."}
            return {field: encoded_img, "shape": model_results["shape"], "field": field}
        if field in model_results:
            return {field: model_results[field], "shape": model_results["shape"], "field": field}
        return {}

    @app.callback(
        Output({"type": "figure-graph", "instance_id": MATCH}, "figure"),
        Input({"type": "plot-data", "instance_id": MATCH}, "data"),
        Input("light-mode-switch", "value"),
        Input({"type": "colorscale-dropdown", "instance_id": MATCH}, "value"),
        Input({"type": "rendering-dropdown", "instance_id": MATCH}, "value"),
        Input({"type": "plot-slider", "instance_id": MATCH, "name": "Saturation"}, "value"),
    )
    def render_figure(
        data: dict[str, Any],
        light_mode: bool,
        colorscale: str,
        rendering_style: str,
        saturation: float,
    ) -> go.Figure:
        if not data:
            return dash.no_update
        if "error" in data:
            fig = go.Figure()
            fig.add_annotation(text=data["error"], x=0.5, y=0.5, showarrow=False)
            return fig

        field = data["field"]
        img = decode_array(data[field], data["shape"])  # type: ignore
        if len(np.shape(img)) == 2:
            return dash_plot_2D(img, label=field, colorscale=colorscale, saturation=saturation, light_mode=light_mode)
        if rendering_style == "2D":
            return dash_plot_2D_planes(
                img, label=field, colorscale=colorscale, saturation=saturation, light_mode=light_mode
            )
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
        Input("download-data-menu-button", "n_clicks"),
        State("active-model", "data"),
        prevent_initial_call=True,
    )
    def download_data(nclicks: int, active_payload: dict[str, Any] | None):
        if active_payload is None:
            return dash.no_update
        cached_result = result_cache.get_result_for_instance(active_payload["instance_id"])
        if cached_result is None:
            return dash.no_update

        filename = f"{active_payload['model_name']}-{active_payload['config_hash'][:12]}.h5"
        write_result = partial(write_hdf5, fields=cached_result.fields, arrays=cached_result.arrays)
        return dcc.send_bytes(write_result, filename)

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


def main() -> None:
    """
    Parse user arguments, then build and launch the application.
    """
    parser = argparse.ArgumentParser(
        prog="prof-dash-gui",
        description="Professor dash visualization",
    )
    parser.add_argument("config", nargs="?", type=str, help="Model configuration file")
    parser.add_argument(
        "-p",
        "--port",
        default=8888,
        type=int,
        help="Target port",
    )
    parser.add_argument(
        "--app-type",
        default="local",
        type=str,
        choices=("local", "server"),
        help="Application type",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run the Dash dashboard in debug mode",
    )
    args = parser.parse_args()

    app = build_application(args.config, args.app_type)
    app.run(host="127.0.0.1", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
