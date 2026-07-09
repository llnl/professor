# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Fast MPI generator for the analytical Fourier-modes example.
# Produces HDF5 files with content IDENTICAL to fouriermodes_study_mpi.py,
# but skips the matplotlib PNG render (PNGs are only for human viewing and
# are not used by prof-trainer). This keeps generation GPU-baseline friendly.

import argparse
import numpy as np
import h5py
from mpi4py import MPI


def make_image(sample, nx, ny, mx1, mx2, my1, my2, outdir):
    output_name = f"{outdir}/image" + str(sample).zfill(6)
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))  # uniform
    vx = mx1 * np.sin(2 * 3.1415926 * X) + mx2 * np.sin(4 * 3.1415926 * X)
    vy = my1 * np.sin(2 * 3.1415926 * Y) + my2 * np.sin(4 * 3.1415926 * Y)
    grid = vx * vy
    inputs = [mx1, mx2, my1, my2]
    with h5py.File(output_name + ".h5", mode="w") as allimages:
        allimages.create_dataset("inputs", data=inputs, dtype="f")
        allimages.create_dataset("fields", data=grid.reshape(1, nx, ny), dtype="f")


parser = argparse.ArgumentParser()
parser.add_argument("-nx", "--nx", default=512, type=int)
parser.add_argument("-ny", "--ny", default=512, type=int)
parser.add_argument("--a_min", default=-1, type=float)
parser.add_argument("--a_max", default=1, type=float)
parser.add_argument("--b_min", default=-1, type=float)
parser.add_argument("--b_max", default=1, type=float)
parser.add_argument("--c_min", default=-1, type=float)
parser.add_argument("--c_max", default=1, type=float)
parser.add_argument("--d_min", default=-1, type=float)
parser.add_argument("--d_max", default=1, type=float)
parser.add_argument("-n", "--nsamp_per_dim", default=2, type=int)
parser.add_argument("-o", "--outdir", default=".", type=str)
args = parser.parse_args()

COMM = MPI.COMM_WORLD
rank = COMM.rank
size = COMM.size

nx, ny = int(args.nx), int(args.ny)
n = args.nsamp_per_dim
a_space = np.linspace(args.a_min, args.a_max, n)
b_space = np.linspace(args.b_min, args.b_max, n)
c_space = np.linspace(args.c_min, args.c_max, n)
d_space = np.linspace(args.d_min, args.d_max, n)

if rank == 0:
    print(f"Generating {n**4} images into {args.outdir} across {size} ranks")

count = 0
for a in a_space:
    for b in b_space:
        for c in c_space:
            for d in d_space:
                if count % size == rank:
                    make_image(count, nx, ny, a, b, c, d, args.outdir)
                count += 1
                if (count % 2000 == 0) and rank == 0:
                    print(count, flush=True)
COMM.Barrier()
if rank == 0:
    print("All Done!")
