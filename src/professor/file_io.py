"""File and configuration payload operations used by the user interfaces."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import stat
import tarfile
import tempfile
import uuid
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from professor.vela import config

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
CONFIG_SUFFIXES = (".yaml", ".yml")
RESULT_SIZE_LIMIT = 24 * 1024 * 1024


def decode_upload(contents: str) -> bytes:
    if "," not in contents:
        raise ValueError("Uploaded file content was not a Dash data URL.")
    _, encoded = contents.split(",", 1)
    return base64.b64decode(encoded)


def is_archive(filename: str) -> bool:
    return filename.lower().endswith(ARCHIVE_SUFFIXES)


def is_config_file(filename: str) -> bool:
    return filename.lower().endswith(CONFIG_SUFFIXES)


def _validate_archive_target(root: Path, member_name: str) -> None:
    target = (root / member_name).resolve()
    resolved_root = root.resolve()
    if resolved_root not in target.parents and target != resolved_root:
        raise ValueError(f"Archive entry escapes extraction directory: {member_name}")


def safe_extract_zip(raw_data: bytes, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(io.BytesIO(raw_data)) as archive:
        for info in archive.infolist():
            _validate_archive_target(root, info.filename)
            unix_mode = info.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise ValueError(f"Archive links are not allowed: {info.filename}")
        archive.extractall(destination)


def safe_extract_tar(raw_data: bytes, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(raw_data), mode="r:*") as archive:
        members = archive.getmembers()
        for member in members:
            _validate_archive_target(root, member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Archive links and special files are not allowed: {member.name}")
        archive.extractall(destination, members=members)


def find_bundle_config(bundle_root: Path) -> Path:
    config_paths = [path for path in bundle_root.rglob("*") if path.is_file() and is_config_file(path.name)]
    if not config_paths:
        raise ValueError("Uploaded bundle does not contain a YAML config file.")
    if len(config_paths) > 1:
        raise ValueError("Uploaded bundle must contain exactly one YAML config file.")
    return config_paths[0]


def resolve_checkpoint_paths(conf: Any, base_dir: Path | None, allow_relative: bool) -> list[str]:
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


def checkpoint_mtimes(checkpoints: list[str]) -> list[float]:
    return [Path(path).stat().st_mtime for path in checkpoints]


def target_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hash_data(data: Any) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_config_payload_from_text(
    yaml_text: str,
    source: str,
    session_id: str,
    base_dir: Path | None = None,
    allow_relative: bool = False,
) -> dict[str, Any]:
    conf = OmegaConf.create(yaml_text)
    resolve_checkpoint_paths(conf, base_dir=base_dir, allow_relative=allow_relative)
    normalized_yaml = OmegaConf.to_yaml(conf, resolve=True)
    prof_config = config.Config.from_text(normalized_yaml, source=source)
    dict_conf = OmegaConf.to_container(conf, resolve=True)
    config_hash = hash_data(dict_conf)
    checkpoint_mtime_values = checkpoint_mtimes(prof_config.checkpoints)
    cache_key = hash_data(
        {
            "config_hash": config_hash,
            "checkpoints": prof_config.checkpoints,
            "checkpoint_mtimes": checkpoint_mtime_values,
            "target_device": target_device_name(),
            "half_precision": prof_config.half_precision,
            "jit": prof_config.jit,
        }
    )
    sliders = []
    for name, slider in prof_config.sliders.items():
        sliders.append(
            {
                "name": name,
                "default": slider["initial_value"],
                "min": slider["lower_bound"],
                "max": slider["upper_bound"],
                "step": 1e-3 * (slider["upper_bound"] - slider["lower_bound"]),
            }
        )

    return {
        "instance_id": str(uuid.uuid4()),
        "session_id": session_id,
        "source": source,
        "yaml_text": normalized_yaml,
        "model_name": prof_config.model_name,
        "fields": prof_config.fields,
        "sliders": sliders,
        "slider_order": list(prof_config.sliders.keys()),
        "appearance": prof_config.appearance,
        "config_hash": config_hash,
        "cache_key": cache_key,
        "checkpoints": prof_config.checkpoints,
        "checkpoint_mtimes": checkpoint_mtime_values,
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
    raw_data = decode_upload(contents)
    if is_archive(filename):
        extraction_root = Path(tempfile.mkdtemp(prefix=f"professor-dash-{session_id}-"))
        if filename.lower().endswith(".zip"):
            safe_extract_zip(raw_data, extraction_root)
        else:
            safe_extract_tar(raw_data, extraction_root)
        config_path = find_bundle_config(extraction_root)
        return build_config_payload_from_text(
            config_path.read_text(encoding="utf-8"),
            source=str(config_path),
            session_id=session_id,
            base_dir=extraction_root,
            allow_relative=True,
        )

    if not is_config_file(filename):
        raise ValueError("Upload a YAML config or an archive containing one YAML config.")

    return build_config_payload_from_text(
        raw_data.decode("utf-8"),
        source=filename,
        session_id=session_id,
        allow_relative=False,
    )


def write_hdf5(buffer: BinaryIO, fields: Iterable[str], arrays: dict[str, np.ndarray]) -> None:
    with h5py.File(buffer, "w") as h5_file:
        for key in fields:
            h5_file.create_dataset(key, data=arrays[key], dtype=np.float32)
