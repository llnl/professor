from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import io
import json
import os
import tarfile
import tempfile
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog

import dash  # type: ignore
import dash_bootstrap_components as dbc  # type: ignore
import h5py
import numpy as np
import plotly.graph_objs as go  # type: ignore
import torch
from dash import ALL, MATCH, Input, Output, State, clientside_callback, dcc, html  # type: ignore
from omegaconf import OmegaConf
from plotly.subplots import make_subplots  # type: ignore

from professor.vela import config, model

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
CONFIG_SUFFIXES = (".yaml", ".yml")
DEFAULT_CACHE_ENTRIES = 2
RESULT_SIZE_LIMIT = 24 * 1024 * 1024
DEFAULT_RESULT_CACHE_ENTRIES = 8
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


@dataclass
class CachedModel:
    key: str
    helper: ProfessorHelper
    ref_count: int = 0
    last_access: float = 0.0


@dataclass
class CachedResult:
    token: str
    instance_id: str
    fields: list[str]
    shape: list[int]
    arrays: dict[str, np.ndarray]
    last_access: float = 0.0


class ProfessorHelper:
    """
    Helper class for launching and evaluating one professor model.
    """

    def __init__(self, prof_config: config.Config) -> None:
        self._config = prof_config
        self._device: torch.device | str = ""
        self._model: torch.nn.Module | None = None

    def close(self) -> None:
        self._model = None
        self._config = None  # type: ignore[assignment]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, model_args: Iterable[float]) -> np.ndarray:
        """
        Evaluate the loaded professor model.
        """
        if self._config is None:
            return np.zeros(0)

        if self._model is None:
            loaded_model = model.PyTorchModel(self._config)
            self._device = loaded_model._get_checkpoint_device("0")
            self._model = loaded_model.models[0].ref

        with torch.no_grad():
            tmp = np.array(model_args)
            x = torch.Tensor(np.reshape(tmp, (1, -1, 1, 1)))
            if self._config.half_precision:
                x = x.half()
            y = self._model(x.to(self._device)).detach().to("cpu").numpy().astype(np.float32)
            return y


class ModelCache:
    def __init__(self, max_entries: int = DEFAULT_CACHE_ENTRIES) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[str, CachedModel] = OrderedDict()
        self._session_keys: dict[str, str] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def release_session(self, session_id: str) -> None:
        with self._lock:
            old_key = self._session_keys.pop(session_id, None)
            if old_key is None:
                return
            entry = self._entries.get(old_key)
            if entry is not None:
                entry.ref_count = max(0, entry.ref_count - 1)
        self._evict_if_needed()

    def get_or_load(self, payload: dict[str, Any], session_id: str) -> ProfessorHelper:
        cache_key = payload["cache_key"]
        key_lock = self._get_key_lock(cache_key)
        with key_lock:
            entry = self._get_entry(cache_key)
            if entry is None:
                entry = self._load_entry(payload)
                with self._lock:
                    self._entries[cache_key] = entry

            with self._lock:
                previous_key = self._session_keys.get(session_id)
                if previous_key != cache_key:
                    self._release_previous_locked(previous_key)
                    self._session_keys[session_id] = cache_key
                    entry.ref_count += 1
                entry.last_access = time.monotonic()
                self._entries.move_to_end(cache_key)

        self._evict_if_needed()
        return entry.helper

    def _get_entry(self, cache_key: str) -> CachedModel | None:
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is not None:
                entry.last_access = time.monotonic()
                self._entries.move_to_end(cache_key)
            return entry

    def _get_key_lock(self, cache_key: str) -> threading.Lock:
        with self._lock:
            lock = self._locks.get(cache_key)
            if lock is None:
                lock = threading.Lock()
                self._locks[cache_key] = lock
            return lock

    def _load_entry(self, payload: dict[str, Any]) -> CachedModel:
        prof_config = config.Config.from_text(payload["yaml_text"], source=payload["source"])
        helper = ProfessorHelper(prof_config)
        return CachedModel(key=payload["cache_key"], helper=helper, last_access=time.monotonic())

    def _release_previous_locked(self, previous_key: str | None) -> None:
        if previous_key is None:
            return
        previous_entry = self._entries.get(previous_key)
        if previous_entry is None:
            return
        previous_entry.ref_count = max(0, previous_entry.ref_count - 1)

    def _evict_if_needed(self) -> None:
        evicted: list[CachedModel] = []
        with self._lock:
            while len(self._entries) > self._max_entries:
                evict_key = None
                for key, entry in self._entries.items():
                    if entry.ref_count == 0:
                        evict_key = key
                        break
                if evict_key is None:
                    return
                evicted.append(self._entries.pop(evict_key))
                self._locks.pop(evict_key, None)

        for entry in evicted:
            entry.helper.close()


model_cache = ModelCache()


class ResultCache:
    def __init__(self, max_entries: int = DEFAULT_RESULT_CACHE_ENTRIES) -> None:
        self._max_entries = max_entries
        self._results: OrderedDict[str, CachedResult] = OrderedDict()
        self._instance_tokens: dict[str, str] = {}
        self._lock = threading.Lock()

    def put(self, instance_id: str, fields: list[str], output: np.ndarray) -> dict[str, Any]:
        token = str(uuid.uuid4())
        shape = list(np.shape(output)[2:])
        arrays = {}
        for index, field in enumerate(fields):
            arrays[field] = np.array(output[0, index, ...], dtype=np.float32, copy=True)

        result = CachedResult(
            token=token,
            instance_id=instance_id,
            fields=fields,
            shape=shape,
            arrays=arrays,
            last_access=time.monotonic(),
        )
        evicted = []
        with self._lock:
            previous_token = self._instance_tokens.get(instance_id)
            if previous_token is not None:
                self._results.pop(previous_token, None)
            self._instance_tokens[instance_id] = token
            self._results[token] = result
            while len(self._results) > self._max_entries:
                old_token, old_result = self._results.popitem(last=False)
                self._instance_tokens.pop(old_result.instance_id, None)
                evicted.append(old_token)

        if evicted:
            gc.collect()
        return {"token": token, "shape": shape, "fields": fields}

    def get_array(self, token: str, field: str) -> np.ndarray | None:
        with self._lock:
            result = self._results.get(token)
            if result is None:
                return None
            result.last_access = time.monotonic()
            self._results.move_to_end(token)
            array = result.arrays.get(field)
            if array is None:
                return None
            return array

    def get_result(self, token: str) -> CachedResult | None:
        with self._lock:
            result = self._results.get(token)
            if result is not None:
                result.last_access = time.monotonic()
                self._results.move_to_end(token)
            return result

    def get_result_for_instance(self, instance_id: str) -> CachedResult | None:
        with self._lock:
            token = self._instance_tokens.get(instance_id)
        if token is None:
            return None
        return self.get_result(token)

    def release_instance(self, instance_id: str) -> None:
        with self._lock:
            token = self._instance_tokens.pop(instance_id, None)
            if token is not None:
                self._results.pop(token, None)
        gc.collect()


result_cache = ResultCache()


def encode_array(x: np.ndarray) -> str:
    """
    Encode an NDarray as a string.
    """
    return base64.b64encode(x.astype(np.float32, copy=False).tobytes()).decode("utf-8")


def decode_array(x_str: str, shape: list[int]) -> np.ndarray:
    """
    Decode an NDarray from a string.
    """
    img_bytes = base64.b64decode(x_str.encode("utf-8"))  # type: ignore
    flat_arr = np.frombuffer(img_bytes, dtype=np.float32)
    return flat_arr.reshape(shape)


def _decode_upload(contents: str) -> bytes:
    if "," not in contents:
        raise ValueError("Uploaded file content was not a Dash data URL.")
    _, encoded = contents.split(",", 1)
    return base64.b64decode(encoded)


def _is_archive(filename: str) -> bool:
    lower_name = filename.lower()
    return lower_name.endswith(ARCHIVE_SUFFIXES)


def _is_config_file(filename: str) -> bool:
    lower_name = filename.lower()
    return lower_name.endswith(CONFIG_SUFFIXES)


def _safe_extract_zip(raw_data: bytes, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(io.BytesIO(raw_data)) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Archive entry escapes extraction directory: {info.filename}")
        archive.extractall(destination)


def _safe_extract_tar(raw_data: bytes, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(raw_data), mode="r:*") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Archive entry escapes extraction directory: {member.name}")
        archive.extractall(destination, members=members)


def _find_bundle_config(bundle_root: Path) -> Path:
    config_paths = [path for path in bundle_root.rglob("*") if path.is_file() and _is_config_file(path.name)]
    if not config_paths:
        raise ValueError("Uploaded bundle does not contain a YAML config file.")
    if len(config_paths) > 1:
        raise ValueError("Uploaded bundle must contain exactly one YAML config file.")
    return config_paths[0]


def _resolve_checkpoint_paths(
    conf: Any,
    base_dir: Path | None,
    allow_relative: bool,
) -> list[str]:
    checkpoints = conf[config.Keys.MODEL.value][config.Keys.PYTORCH.value][config.Keys.CHECKPOINTS.value]
    resolved: list[str] = []
    for raw_path in checkpoints:
        path = Path(os.path.expanduser(str(raw_path)))
        if not path.is_absolute():
            if not allow_relative or base_dir is None:
                raise ValueError(
                    "Uploaded YAML configs must use absolute checkpoint paths unless they are uploaded in a bundle."
                )
            path = base_dir / path
        path = path.resolve()
        if not path.exists():
            raise ValueError(f"Checkpoint path does not exist: {path}")
        resolved.append(str(path))

    conf[config.Keys.MODEL.value][config.Keys.PYTORCH.value][config.Keys.CHECKPOINTS.value] = resolved
    return resolved


def _checkpoint_mtimes(checkpoints: list[str]) -> list[float]:
    return [Path(path).stat().st_mtime for path in checkpoints]


def _target_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _hash_data(data: Any) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slider_configs(prof_config: config.Config) -> list[SliderConfig]:
    sliders = []
    for name, slider in prof_config.sliders.items():
        step = 1e-3 * (slider["upper_bound"] - slider["lower_bound"])
        sliders.append(
            SliderConfig(
                name=name,
                default=slider["initial_value"],
                min=slider["lower_bound"],
                max=slider["upper_bound"],
                step=step,
            )
        )
    return sliders


def build_config_payload_from_text(
    yaml_text: str,
    source: str,
    session_id: str,
    base_dir: Path | None = None,
    allow_relative: bool = False,
) -> dict[str, Any]:
    conf = OmegaConf.create(yaml_text)
    _resolve_checkpoint_paths(conf, base_dir=base_dir, allow_relative=allow_relative)
    normalized_yaml = OmegaConf.to_yaml(conf, resolve=True)
    prof_config = config.Config.from_text(normalized_yaml, source=source)
    dict_conf = OmegaConf.to_container(conf, resolve=True)
    config_hash = _hash_data(dict_conf)
    checkpoint_mtimes = _checkpoint_mtimes(prof_config.checkpoints)
    cache_key = _hash_data(
        {
            "config_hash": config_hash,
            "checkpoints": prof_config.checkpoints,
            "checkpoint_mtimes": checkpoint_mtimes,
            "target_device": _target_device_name(),
            "half_precision": prof_config.half_precision,
            "jit": prof_config.jit,
        }
    )

    return {
        "instance_id": str(uuid.uuid4()),
        "session_id": session_id,
        "source": source,
        "yaml_text": normalized_yaml,
        "model_name": prof_config.model_name,
        "fields": prof_config.fields,
        "sliders": [slider.__dict__ for slider in _slider_configs(prof_config)],
        "slider_order": list(prof_config.sliders.keys()),
        "appearance": prof_config.appearance,
        "config_hash": config_hash,
        "cache_key": cache_key,
        "checkpoints": prof_config.checkpoints,
        "checkpoint_mtimes": checkpoint_mtimes,
        "result_size_limit": RESULT_SIZE_LIMIT,
    }


def build_config_payload_from_path(config_path: str, session_id: str) -> dict[str, Any]:
    path = Path(os.path.expanduser(config_path)).resolve()
    if not path.exists():
        raise ValueError(f"Config path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")
    yaml_text = path.read_text(encoding="utf-8")
    return build_config_payload_from_text(
        yaml_text,
        source=str(path),
        session_id=session_id,
        base_dir=path.parent,
        allow_relative=True,
    )


def build_config_payload_from_upload(contents: str, filename: str, session_id: str) -> dict[str, Any]:
    raw_data = _decode_upload(contents)
    if _is_archive(filename):
        extraction_root = Path(tempfile.mkdtemp(prefix=f"professor-dash-{session_id}-"))
        if filename.lower().endswith(".zip"):
            _safe_extract_zip(raw_data, extraction_root)
        else:
            _safe_extract_tar(raw_data, extraction_root)
        config_path = _find_bundle_config(extraction_root)
        return build_config_payload_from_text(
            config_path.read_text(encoding="utf-8"),
            source=str(config_path),
            session_id=session_id,
            base_dir=extraction_root,
            allow_relative=True,
        )

    if not _is_config_file(filename):
        raise ValueError("Upload a YAML config or an archive containing one YAML config.")

    return build_config_payload_from_text(
        raw_data.decode("utf-8"),
        source=filename,
        session_id=session_id,
        allow_relative=False,
    )


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


def build_load_model_modal(initial_status: Any = None, is_open: bool = False, app_type: str = 'local') -> dbc.Modal:

    modal_rows = []
    if app_type == 'server':
        modal_rows.append(
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Upload(
                            id="config-upload",
                            children=html.Div(["Drag and drop or select a YAML config or bundle"]),
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

    elif app_type == 'local':
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


def get_file_path() -> str:
    """
    Request a configuration filename from the user via a pop up search window
    """
    # Open the tk window and push it to the top layer
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.focus_force()

    # Ask the user for the filename
    filetypes = [
        ("Config (*.yml, *.yaml)", "*.yml *.yaml"),
        ("Archive (*.zip, *.tar, *.tar.gz)", "*.zip *.tar *.tar.gz"),
        ("All", "*")
    ]
    file_path = filedialog.askopenfilename(parent=root, title="Select Target Model", filetypes=filetypes)
    root.destroy()
    return file_path


def build_application(initial_config: str | None = None,
                      app_type: str = 'local') -> dash.Dash:
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
        def load_config(
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
        def load_config(
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

        def write_hdf5(buffer: io.BytesIO) -> None:
            with h5py.File(buffer, "w") as h5_file:
                for key in cached_result.fields:
                    h5_file.create_dataset(
                        key,
                        data=cached_result.arrays[key],
                        dtype=np.float32,
                    )

        filename = f"{active_payload['model_name']}-{active_payload['config_hash'][:12]}.h5"
        return dcc.send_bytes(write_hdf5, filename)

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
