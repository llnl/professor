# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import argparse
import numpy as np
import h5py
from scipy.stats import qmc
from scipy.signal import medfilt
from mpi4py import MPI
import matplotlib.pyplot as plt


def plot_image(c: np.ndarray):
    vmax = np.amax(c)
    mid = np.shape(c)[0] // 2

    f, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(c[mid, :, :], vmin=0, vmax=vmax, cmap='jet')
    axes[0, 1].imshow(c[:, mid, :], vmin=0, vmax=vmax, cmap='jet')
    axes[1, 0].imshow(c[:, :, mid ], vmin=0, vmax=vmax, cmap='jet')
    v_empty = np.zeros((2, 2))
    v_empty[:] = np.nan
    ca = axes[1, 1].imshow(v_empty, vmin=0, vmax=vmax, cmap='jet')
    axes[1, 1].set_axis_off()
    cb = plt.colorbar(ca)
    cb.set_label('Concentration')
    plt.show()


def generate_sample(
        sample: int,
        x: float,
        y: float,
        z: float,
        t: float,
        v: float,
        diffusivity: float,
        resolution: int = 128,
        output_dir: str = './',
        render_image: bool = False
        ):
    # Initialize grid
    xg = np.linspace(0, 1, resolution).astype(np.float32)
    G = np.meshgrid(xg, xg, xg, indexing='ij')
    rs = (G[0] - x) ** 2 + (G[1] - y) ** 2 + (G[2] - z) ** 2

    # Generate data
    a = v * (4 * np.pi * diffusivity * t) ** -1.5
    b = rs / (4.0 * diffusivity * t)
    c = a * np.exp(-b)
    c = medfilt(c, kernel_size=3)

    if render_image:
        plot_image(c)

    # inputs = [x, y, z, t, v, diffusivity]
    inputs = [x, y, z, diffusivity]
    fname = os.path.join(output_dir, f"image_{sample:06d}.h5")
    with h5py.File(fname, mode="w") as allimages:
        allimages.create_dataset(
            "inputs",
            data=inputs,
            dtype="f",
            # Note that compression greatly increases the time to load data
            # compression="gzip",
        )
        allimages.create_dataset(
            "fields",
            data=c.reshape(1, resolution, resolution, resolution),
            dtype="f",
            # Note that compression greatly increases the time to load data
            # compression="gzip",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resolution", default=128, type=int, help="Spatial Resolution")
    parser.add_argument("-n", "--num-samples", default=100, type=int, help="Number of samples")
    parser.add_argument("-o", "--output", default='./', type=str, help='Output Directory')
    parser.add_argument("-s", "--seed", type=int, help='Random Seed')
    args = parser.parse_args()

    # Sample the target space
    sampler = qmc.LatinHypercube(d=6, seed=args.seed)
    sample = np.array(sampler.random(n=args.num_samples))
    sample_keys = ['x', 'y', 'z', 'diffusivity']
    vals = {k: sample[:, ii] for ii, k in enumerate(sample_keys)}

    # Sync values
    COMM = MPI.COMM_WORLD
    rank = COMM.rank
    size = COMM.size
    num_items_per_rank = args.num_samples // size
    if size > 1:
        broadcast_size = num_items_per_rank * size
        for k in sample_keys:
            recvbuf = np.zeros(num_items_per_rank, dtype=np.float64)
            sendbuf = np.ascontiguousarray(vals[k][:broadcast_size])
            COMM.Scatter(sendbuf, recvbuf, root=0)
            if (rank == size - 1) and (broadcast_size < args.num_samples):
                recvbuf = np.concatenate([recvbuf, vals[k][broadcast_size:]], axis=0)
            vals[k] = recvbuf

    # Generate data
    os.makedirs(args.output, exist_ok=True)
    if rank == 0:
        print('Generating samples')

    rank_offset = rank * num_items_per_rank
    for ii in range(len(vals['x'])):
        generate_sample(
            ii + rank_offset,
            vals['x'][ii],
            vals['y'][ii],
            vals['z'][ii],
            1.0,
            1.0,
            0.4 * vals['diffusivity'][ii] + 0.1,
            args.resolution,
            output_dir=args.output
        )

    COMM.Barrier()
    if rank == 0:
        print('Done!')


if __name__ == '__main__':
    main()
