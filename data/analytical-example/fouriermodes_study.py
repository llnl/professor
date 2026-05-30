# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import numpy as np
import h5py  # type: ignore[import-untyped]
from matplotlib import pyplot as plt


def make_image(sample, nx, ny, mx1, mx2, my1, my2):
    output_name = "image" + str(sample).zfill(6)

    # Initialize grid
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))  # uniform

    vx = mx1 * np.sin(2 * 3.1415926 * X) + mx2 * np.sin(4 * 3.1415926 * X)
    vy = my1 * np.sin(2 * 3.1415926 * Y) + my2 * np.sin(4 * 3.1415926 * Y)

    grid = vx * vy

    vmin = -4
    vmax = 4

    inputs = [mx1, mx2, my1, my2]

    with h5py.File(output_name + ".h5", mode="w") as allimages:
        allimages.create_dataset(
            "inputs",
            data=inputs,
            dtype="f",
            # Note that compression greatly increases the time to load data
            # compression="gzip",
        )
        allimages.create_dataset(
            "fields",
            data=grid.reshape(1, nx, ny),
            dtype="f",
            # Note that compression greatly increases the time to load data
            # compression="gzip",
        )

    fig, ax = plt.subplots()
    ax.imshow(grid, interpolation="none", vmax=vmax, vmin=vmin)
    plt.savefig(output_name + ".png")
    plt.close()


parser = argparse.ArgumentParser()
# the image dimension is (nx,ny)
parser.add_argument("-nx", "--nx", default=512, type=int, help="Spatial Resolution X")
parser.add_argument("-ny", "--ny", default=512, type=int, help="Spatial Resolution Y")

# The 2D function is an outer product of two 1D functions
# The 1D functions are of the form
# f(x) = a sin(2 pi x) + b sin(4 pi x)
# f(y) = c sin(2 pi y) + d sin(4 pi y)
# The design paramters are the aplitudes [a,b,c,d] of the sine waves
#
# Uniform Carrtesian sampling is used, the bounds are specified by [min.max]
# The design space is hardwired to dim=4
parser.add_argument("--a_min", default=-1, type=float)
parser.add_argument("--a_max", default=1, type=float)
parser.add_argument("--b_min", default=-1, type=float)
parser.add_argument("--b_max", default=1, type=float)
parser.add_argument("--c_min", default=-1, type=float)
parser.add_argument("--c_max", default=1, type=float)
parser.add_argument("--d_min", default=-1, type=float)
parser.add_argument("--d_max", default=1, type=float)

#
parser.add_argument("-n", "--nsamp_per_dim", default=2, type=int)

args = parser.parse_args()
nx = int(args.nx)
ny = int(args.ny)

a_min = float(args.a_min)
a_max = float(args.a_max)
b_min = float(args.b_min)
b_max = float(args.b_max)

c_min = float(args.c_min)
c_max = float(args.c_max)
d_min = float(args.d_min)
d_max = float(args.d_max)
n = args.nsamp_per_dim

a_space = np.linspace(a_min, a_max, n)
b_space = np.linspace(b_min, b_max, n)
c_space = np.linspace(c_min, c_max, n)
d_space = np.linspace(d_min, d_max, n)

print("Generating", n * n * n * n, " images!")

count = 0
for a in a_space:
    for b in b_space:
        for c in c_space:
            for d in d_space:
                make_image(count, nx, ny, a, b, c, d)
                count += 1
                if np.mod(count, 100) == 0:
                    print(count)

print("All Done!")
