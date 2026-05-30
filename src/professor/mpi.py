# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from typing import Tuple


def load_balance_inds(n: int, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get balanced indices to use to split a tensor on multiple mpi ranks.

    Args:
        n (int): total number of indices in the tensor
        size (int): total number of mpi ranks (MPI.COMM_WORLD.size)

    Returns:
        (tuple): First item is a numpy array containing the number of indices
            that will be on each rank. Second item is a numpy array containgin
            the indices used to slice the tensor onto multple ranks.
    """
    n_sims_per_rank = np.zeros(size, dtype=np.int_)
    temp_rank = size - 1
    for i in range(n):
        n_sims_per_rank[temp_rank] += 1
        temp_rank -= 1
        if temp_rank < 0:
            temp_rank = size - 1

    rank_ordering = np.zeros(size + 1, dtype=np.int_)
    for r in range(size):
        rank_ordering[r + 1] = rank_ordering[r] + n_sims_per_rank[r]
    return n_sims_per_rank, rank_ordering
