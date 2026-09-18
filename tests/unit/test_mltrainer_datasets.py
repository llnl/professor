import h5py
import numpy as np
import pytest
import torch

from professor.mltrainer import (
    CompleteDataset,
    CompleteDatasetDivideScaling,
    CompleteDatasetOneFileSims,
)


def _write_dataset(path, inputs, fields):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("inputs", data=inputs)
        handle.create_dataset("fields", data=fields)


def test_complete_dataset_direct_parametric_slice(tmp_path):
    fields = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    filename = tmp_path / "case.h5"
    _write_dataset(filename, np.array([2.0, 4.0], dtype=np.float32), fields)
    filelist = np.array([filename.name])
    dataset = CompleteDataset(filelist, n_channels=2, path=str(tmp_path))

    inputs, full_fields = dataset[0]
    dataset.enable_parametric_slicing()
    sliced_inputs, sliced_fields = dataset[2]

    assert len(dataset) == 5
    assert dataset.pixels_z == 1
    assert dataset.n_input == 3
    torch.testing.assert_close(sliced_inputs[:-1], inputs)
    torch.testing.assert_close(sliced_fields, full_fields[..., 2])
    assert sliced_inputs[-1].item() == pytest.approx(2 / 5)


def test_dataset_switches_from_full_reads_to_direct_parametric_reads(tmp_path):
    fields = np.arange(1 * 2 * 3 * 3, dtype=np.float32).reshape(1, 2, 3, 3)
    filename = tmp_path / "case.h5"
    _write_dataset(filename, np.array([2.0], dtype=np.float32), fields)
    filelist = np.array([filename.name])

    dataset = CompleteDataset(filelist, 1, str(tmp_path))

    full_inputs, full_fields = dataset[0]
    dataset.enable_parametric_slicing()
    direct_inputs, direct_fields = dataset[2]

    torch.testing.assert_close(direct_inputs[:-1], full_inputs)
    torch.testing.assert_close(direct_fields, full_fields[..., 2])
    assert direct_inputs.shape == (2, 1, 1)
    assert direct_fields.shape == (1, 2, 3)


def test_one_file_and_scaled_datasets_share_metadata_and_slice_api(tmp_path):
    one_file_fields = np.arange(2 * 1 * 2 * 3 * 4, dtype=np.float32).reshape(2, 1, 2, 3, 4)
    one_file_name = tmp_path / "cases.h5"
    _write_dataset(one_file_name, np.array([[2.0], [4.0]], dtype=np.float32), one_file_fields)
    filelist = np.array([[one_file_name.name, 1]], dtype=object)
    one_file = CompleteDatasetOneFileSims(filelist, 1, str(tmp_path))

    scaled_fields = np.arange(1 * 2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)
    scaled_name = tmp_path / "scaled.h5"
    _write_dataset(scaled_name, np.array([2.0], dtype=np.float32), scaled_fields)
    scaled = CompleteDatasetDivideScaling(
        np.array([scaled_name.name]),
        1,
        str(tmp_path),
        torch.tensor([2.0]).view(1, 1, 1),
    )

    one_file.enable_parametric_slicing()
    scaled.enable_parametric_slicing()
    _, one_file_slice = one_file[3]
    scaled_inputs, _ = scaled[3]

    assert len(one_file) == 4
    assert one_file.pixels_z == scaled.pixels_z == 1
    torch.testing.assert_close(one_file_slice, torch.from_numpy(one_file_fields[1, ..., 3]))
    assert scaled_inputs[0].item() == pytest.approx(1.0)
    assert scaled_inputs[-1].item() == pytest.approx(3 / 4)
