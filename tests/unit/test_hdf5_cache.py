# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pickle
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from professor.hdf5_cache import HDF5FileCache  # noqa: E402


@pytest.fixture
def make_shard(tmp_path):
    """Factory writing a small dataset_type=1 style HDF5 shard file."""

    def _make(name: str, n_samples: int = 3):
        path = tmp_path / name
        rng = np.random.default_rng(hash(name) % (2**32))
        with h5py.File(path, "w") as f:
            f.create_dataset("inputs", data=rng.random((n_samples, 4), dtype=np.float32))
            f.create_dataset("fields", data=rng.random((n_samples, 1, 8, 8), dtype=np.float32))
        return str(path)

    return _make


def test_hit_returns_same_object(make_shard):
    cache = HDF5FileCache(capacity=2)
    path = make_shard("a.h5")
    f1 = cache.open_file(path)
    f2 = cache.open_file(path)
    assert f1 is f2
    assert len(cache) == 1


def test_eviction_closes_lru(make_shard):
    cache = HDF5FileCache(capacity=2)
    fa = cache.open_file(make_shard("a.h5"))
    fb = cache.open_file(make_shard("b.h5"))
    fc = cache.open_file(make_shard("c.h5"))
    assert not fa
    assert fb and fc
    assert len(cache) == 2


def test_access_refreshes_recency(make_shard):
    cache = HDF5FileCache(capacity=2)
    sa, sb = make_shard("a.h5"), make_shard("b.h5")
    fa = cache.open_file(sa)
    fb = cache.open_file(sb)
    cache.open_file(sa)  # make shard a  most recent
    cache.open_file(make_shard("c.h5"))  # evict shard b
    assert fa
    assert not fb


def test_pid_change_resets_without_close(make_shard):
    cache = HDF5FileCache(capacity=2)
    path = make_shard("a.h5")
    original_file_handle = cache.open_file(path)
    cache.pid = cache.pid + 1  # simulates a forked worker
    new_file_handle = cache.open_file(path)
    assert new_file_handle is not original_file_handle
    assert original_file_handle  # assert inherited handle was NOT closed
    assert len(cache) == 1
    original_file_handle.close()


def test_capacity_validation():
    with pytest.raises(ValueError):
        HDF5FileCache(capacity=0)


def test_cli_rejects_cache_without_dataset_type_1(monkeypatch):
    from professor.__main__ import prof_trainer

    monkeypatch.setattr(sys, "argv", ["prof-trainer", "--dataset_type", "0", "--hdf5_cache_size", "4"])
    with pytest.raises(SystemExit):
        prof_trainer()


def test_pickle_drops_handles(make_shard):
    cache = HDF5FileCache(capacity=3)
    cache.open_file(make_shard("a.h5"))
    restored = pickle.loads(pickle.dumps(cache))
    assert restored.capacity == 3
    assert len(restored) == 0


def test_dataset_reads_through_cache(make_shard, tmp_path):
    pytest.importorskip("mpi4py")
    torch = pytest.importorskip("torch")  # noqa: F841
    from professor.mltrainer import CompleteDatasetOneFileSims

    make_shard("a.h5", n_samples=3)
    make_shard("b.h5", n_samples=2)
    
    # create a dataset file list
    rows = [("a.h5", j) for j in range(3)] + [("b.h5", j) for j in range(2)]
    filelist = np.array(rows, dtype=[("f0", "U64"), ("f1", np.int64)])

    plain = CompleteDatasetOneFileSims(filelist, n_channels=1, path=str(tmp_path))
    cached = CompleteDatasetOneFileSims(filelist, n_channels=1, path=str(tmp_path), cache_size=2)
    for i in range(len(filelist)):
        x0, y0 = plain[i]
        x1, y1 = cached[i]
        assert (x0 == x1).all()
        assert (y0 == y1).all()
    assert cached.cache is not None
    assert len(cached.cache) >= 1
