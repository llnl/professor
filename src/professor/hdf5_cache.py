# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from collections import OrderedDict
from typing import Any

import h5py


class HDF5FileCache:
    """
    Per-process LRU cache of open read-only hdf5 file handles.

    Lets many samples in one shard file share a single open handle instead of
    reopening the file per sample. Handles are owned by the cache, so callers
    must not close them.

    Every dataloader worker builds its own pool, so no worker_init_fn is
    needed. Under the spawn and forkserver contexts the cache is pickled with
    only its capacity, and under fork the inherited entries are dropped without
    closing, since closing an inherited handle corrupts hdf5 state shared with
    the parent.

    Args:
        capacity (int): maximum number of files kept open per process
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity: int = capacity
        self.pid: int = -1
        self.files: "OrderedDict[str, h5py.File]" = OrderedDict()

    def open_file(self, path: str) -> h5py.File:
        pid = os.getpid()
        if pid != self.pid:
            # first use or a forked worker: drop inherited entries, never close them
            self.files = OrderedDict()
            self.pid = pid
        f = self.files.get(path)
        if f is not None:
            self.files.move_to_end(path)
            return f
        f = h5py.File(path, "r")
        self.files[path] = f
        if len(self.files) > self.capacity:
            _, evicted = self.files.popitem(last=False)
            evicted.close()
        return f

    def __len__(self) -> int:
        return len(self.files)

    def __reduce__(self) -> Any:
        # rebuild empty in the child; h5py handles cannot be pickled
        return (type(self), (self.capacity,))
