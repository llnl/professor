import base64
import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from professor.gui.dash_gui import (
    ResultCache,
    SliderConfig,
    build_config_payload_from_text,
    build_config_payload_from_upload,
    build_slider,
)
from professor.vela.config import Config
from professor.vela.model import Model


def _yaml_text(checkpoint: Path) -> str:
    return f"""
model:
  pytorch:
    name: Test Model
    checkpoints:
      - {checkpoint}
    devices:
      - 0
    seed: 1231
    dimensions:
      n_inputs: 1
      batch_size: 1
      y_pixels: 2
      x_pixels: 2
    invocation:
      target: torch.nn.Identity
      params: {{}}
    execution:
      half_precision: false
      jit: false

gui:
  napari:
    fields:
      - density
    sliders:
      alpha:
        lower_bound: 0.0
        upper_bound: 1.0
        initial_value: 0.5
"""


def _upload_contents(data: bytes) -> str:
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:application/octet-stream;base64,{encoded}"


def test_config_from_text_uses_existing_schema(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    cfg = Config.from_text(_yaml_text(checkpoint))

    assert cfg.model_name == "Test Model"
    assert cfg.fields == ["density"]
    assert cfg.slider_initial_values == [0.5]


def test_payloads_for_same_config_have_distinct_instances(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    yaml_text = _yaml_text(checkpoint)

    first = build_config_payload_from_text(yaml_text, "first.yml", "session-a")
    second = build_config_payload_from_text(yaml_text, "second.yml", "session-b")

    assert first["config_hash"] == second["config_hash"]
    assert first["cache_key"] == second["cache_key"]
    assert first["instance_id"] != second["instance_id"]


def test_plain_uploaded_yaml_rejects_relative_checkpoint_path():
    yaml_text = _yaml_text(Path("relative-checkpoint.pt"))

    with pytest.raises(ValueError, match="absolute checkpoint paths"):
        build_config_payload_from_upload(_upload_contents(yaml_text.encode("utf-8")), "model.yml", "session-a")


def test_uploaded_bundle_resolves_relative_checkpoint_path(tmp_path):
    yaml_text = _yaml_text(Path("checkpoint.pt"))
    archive_data = io.BytesIO()
    with zipfile.ZipFile(archive_data, "w") as archive:
        archive.writestr("config.yml", yaml_text)
        archive.writestr("checkpoint.pt", b"checkpoint")

    payload = build_config_payload_from_upload(_upload_contents(archive_data.getvalue()), "bundle.zip", "session-a")

    assert payload["fields"] == ["density"]
    assert Path(payload["checkpoints"][0]).is_absolute()
    assert Path(payload["checkpoints"][0]).exists()


def test_uploaded_bundle_rejects_escaping_paths():
    archive_data = io.BytesIO()
    with zipfile.ZipFile(archive_data, "w") as archive:
        archive.writestr("../config.yml", _yaml_text(Path("checkpoint.pt")))

    with pytest.raises(ValueError, match="escapes extraction directory"):
        build_config_payload_from_upload(_upload_contents(archive_data.getvalue()), "bundle.zip", "session-a")


class DummyModel(Model):
    def __init__(self) -> None:
        super().__init__(config=None)  # type: ignore[arg-type]

    @property
    def num_gpus(self) -> int:
        return 0

    def _load_model(self, model: Any, path: str, device: Any) -> Any:
        return model

    def _get_tensor(self, device: Any) -> Any:
        return None

    def _move_tensor_to_cpu(self, tensor: Any) -> Any:
        return tensor

    def _get_checkpoint_device(self, device_id: str) -> Any:
        return None


def test_model_collection_is_instance_scoped():
    first = DummyModel()
    second = DummyModel()

    first.models.append("loaded")  # type: ignore[arg-type]

    assert first.models == ["loaded"]
    assert second.models == []


def test_result_cache_returns_selected_large_field_only():
    cache = ResultCache()
    output = np.zeros((1, 2, 1024, 1024), dtype=np.float32)
    output[0, 1, :, :] = 2.0

    metadata = cache.put("instance-a", ["density", "pressure"], output)
    selected = cache.get_array(metadata["token"], "pressure")

    assert metadata["shape"] == [1024, 1024]
    assert metadata["fields"] == ["density", "pressure"]
    assert selected is not None
    assert selected.shape == (1024, 1024)
    assert np.all(selected == 2.0)


def test_result_cache_replaces_previous_instance_result():
    cache = ResultCache()
    first = cache.put("instance-a", ["density"], np.zeros((1, 1, 2, 2), dtype=np.float32))
    second = cache.put("instance-a", ["density"], np.ones((1, 1, 2, 2), dtype=np.float32))

    assert cache.get_array(first["token"], "density") is None
    selected = cache.get_array(second["token"], "density")
    assert selected is not None
    assert np.all(selected == 1.0)


def test_slider_marks_use_concise_float_labels():
    row = build_slider(
        SliderConfig(name="Vel", default=0.2, min=0.18, max=0.22, step=0.00004),
        instance_id="instance-a",
    )
    slider = row.children[1].children

    assert len(slider.marks) == 5
    assert list(slider.marks.values()) == ["0.18", "0.19", "0.2", "0.21", "0.22"]
    assert all(len(label.replace(".", "").replace("-", "")) <= 6 for label in slider.marks.values())
