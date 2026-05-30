# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import numpy as np
from sklearn.model_selection import KFold  # type: ignore[import-untyped]
from typing import List

path = f"/p/gpfs1/{os.getenv('USER')}/data/linear_shaped_charge"
filelist = np.genfromtxt(
    f'{path}/filelist.txt',
    dtype=None,
    encoding="UTF-8",
)
simulations = {}
# Build a dictionary, where the key is the simulation number
# and the value is a list of all of the simulation index values!
for file in filelist:
    # file[0] is the simulation number
    # file[1] is the 'index number'
    if file[0] not in simulations:
        simulations[file[0]] = [file[1]]
    else:
        simulations[file[0]].append(file[1])

n_files = len(simulations)
# we want to make a committee where we split simulations,
# and not timesteps/index!
sim_number_as_int = np.arange(n_files)
simulation_keys = np.array(list(simulations.keys()))
kf = KFold(n_splits=5, random_state=42, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(sim_number_as_int)):
    print(f"Fold {i}:")
    print(f"Training dataset index: {train_index}")
    print(f"Test dataset index: {test_index}")
    # do not do anything with test datasets!
    train_sims = simulation_keys[train_index]
    # now build a new filelist
    filelist_kf = []
    for sim in train_sims:
        for idx in simulations[sim]:
            filelist_kf.append([sim, idx])
    filelist_kf_arr = np.array(filelist_kf)
    np.savetxt(
        f'{path}/filelist_committee_{i}.txt', filelist_kf_arr, fmt="%s",
    )
