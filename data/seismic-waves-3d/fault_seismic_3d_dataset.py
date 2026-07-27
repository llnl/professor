# Copyright 2026, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generate far-field 3D seismic wave propagation examples.

Each sample is a point double-couple fault source in a homogeneous isotropic
1 km cube.

    inputs:  sampled source/material parameters plus the output time
    fields:  displacement field with shape (3, nx, ny, nz)

One file is written for each realization/time-step pair.

The displacement uses the far-field moment tensor approximation

    uP = eR * (eR.M.eR) * O(t - R / vp) / (4 pi rho vp^3 R)
    uS = (I - eR eR).M.eR * O(t - R / vs) / (4 pi rho vs^3 R)

where O(t) is the moment-rate source time function.  Points very close to the
source are outside the far-field regime and are left at zero by a configurable
exclusion radius.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from mpi4py import MPI
from numba import njit, prange
from scipy.stats import qmc

INPUT_NAMES = (
    "source_x",
    "source_y",
    "source_z",
    "strike",
    "dip",
    "rake",
    "vp",
    "vs",
    "density",
    "time",
)

LABEL_NAMES = (
    "displacement_x",
    "displacement_y",
    "displacement_z",
)


@njit(cache=True)
def gaussian_moment_rate(t, duration):
    """Unit-area Gaussian moment-rate pulse centered at 3 sigma."""
    sigma = duration / 6.0
    center = 0.5 * duration
    z = (t - center) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


@njit(cache=True)
def moment_tensor_from_fault(strike, dip, rake, moment):
    """Double-couple moment tensor for ENU coordinates.

    x is east, y is north, and z is up.  Strike is clockwise from north, dip is
    from horizontal, and rake is measured in the fault plane from strike.
    """
    ss = np.sin(strike)
    cs = np.cos(strike)
    sd = np.sin(dip)
    cd = np.cos(dip)
    sr = np.sin(rake)
    cr = np.cos(rake)

    strike_vec = np.empty(3, dtype=np.float64)
    strike_vec[0] = ss
    strike_vec[1] = cs
    strike_vec[2] = 0.0

    down_dip_vec = np.empty(3, dtype=np.float64)
    down_dip_vec[0] = cs * cd
    down_dip_vec[1] = -ss * cd
    down_dip_vec[2] = -sd

    normal_vec = np.empty(3, dtype=np.float64)
    normal_vec[0] = -cs * sd
    normal_vec[1] = ss * sd
    normal_vec[2] = -cd

    slip_vec = np.empty(3, dtype=np.float64)
    for i in range(3):
        slip_vec[i] = cr * strike_vec[i] + sr * down_dip_vec[i]

    mt = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            mt[i, j] = moment * (slip_vec[i] * normal_vec[j] + slip_vec[j] * normal_vec[i])
    return mt


@njit(parallel=True, cache=True)
def far_field_displacement(
    points,
    times,
    source,
    strike,
    dip,
    rake,
    vp,
    vs,
    density,
    moment,
    duration,
    exclusion_radius,
):
    npoints = points.shape[0]
    nt = times.shape[0]
    out = np.zeros((nt, 3, npoints), dtype=np.float32)
    mt = moment_tensor_from_fault(strike, dip, rake, moment)

    inv_p = 1.0 / (4.0 * np.pi * density * vp * vp * vp)
    inv_s = 1.0 / (4.0 * np.pi * density * vs * vs * vs)

    for ip in prange(npoints):
        rx = points[ip, 0] - source[0]
        ry = points[ip, 1] - source[1]
        rz = points[ip, 2] - source[2]
        radius = np.sqrt(rx * rx + ry * ry + rz * rz)
        if radius <= exclusion_radius:
            continue

        er0 = rx / radius
        er1 = ry / radius
        er2 = rz / radius

        mer0 = mt[0, 0] * er0 + mt[0, 1] * er1 + mt[0, 2] * er2
        mer1 = mt[1, 0] * er0 + mt[1, 1] * er1 + mt[1, 2] * er2
        mer2 = mt[2, 0] * er0 + mt[2, 1] * er1 + mt[2, 2] * er2
        radial_amp = er0 * mer0 + er1 * mer1 + er2 * mer2

        p0 = er0 * radial_amp
        p1 = er1 * radial_amp
        p2 = er2 * radial_amp
        s0 = mer0 - p0
        s1 = mer1 - p1
        s2 = mer2 - p2

        scale_p = inv_p / radius
        scale_s = inv_s / radius
        tp = radius / vp
        ts = radius / vs

        for it in range(nt):
            t = times[it]
            op = gaussian_moment_rate(t - tp, duration) * scale_p
            os = gaussian_moment_rate(t - ts, duration) * scale_s
            out[it, 0, ip] = p0 * op + s0 * os
            out[it, 1, ip] = p1 * op + s1 * os
            out[it, 2, ip] = p2 * op + s2 * os

    return out


def build_points(nx, ny, nz, domain_size):
    x = np.linspace(0.0, domain_size, nx, dtype=np.float64)
    y = np.linspace(0.0, domain_size, ny, dtype=np.float64)
    z = np.linspace(0.0, domain_size, nz, dtype=np.float64)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    return x, y, z, points


def sample_parameter_table(args):
    if args.nsamples == 0:
        return np.empty((0, 9), dtype=np.float64)

    sampler = qmc.LatinHypercube(d=9, seed=args.seed)
    unit = sampler.random(n=args.nsamples)
    margin = args.source_margin
    source_min = margin
    source_max = args.domain_size - margin

    lower = np.array(
        [
            source_min,
            source_min,
            source_min,
            0.0,
            args.dip_min,
            -np.pi,
            args.vp_min,
            args.vp_vs_min,
            args.density_min,
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            source_max,
            source_max,
            source_max,
            2.0 * np.pi,
            args.dip_max,
            np.pi,
            args.vp_max,
            args.vp_vs_max,
            args.density_max,
        ],
        dtype=np.float64,
    )

    scaled = qmc.scale(unit, lower, upper)
    table = scaled.copy()
    table[:, 7] = scaled[:, 6] / scaled[:, 7]
    return table


def input_scaling_bounds(args, time_upper):
    margin = args.source_margin
    source_min = margin
    source_max = args.domain_size - margin
    vs_min = args.vp_min / args.vp_vs_max
    vs_max = args.vp_max / args.vp_vs_min

    lower = np.array(
        [
            source_min,
            source_min,
            source_min,
            0.0,
            args.dip_min,
            -np.pi,
            args.vp_min,
            vs_min,
            args.density_min,
            args.t_min,
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            source_max,
            source_max,
            source_max,
            2.0 * np.pi,
            args.dip_max,
            np.pi,
            args.vp_max,
            vs_max,
            args.density_max,
            time_upper,
        ],
        dtype=np.float64,
    )
    return lower, upper


def scale_inputs_to_unit_range(inputs, lower, upper):
    return (inputs - lower) / (upper - lower)


def threshold_fields_for_write(fields):
    fields[np.abs(fields) < 1.0e-3] = 0.0


def update_ranges(inputs, fields, input_min, input_max, label_min, label_max):
    input_min[:] = np.minimum(input_min, inputs)
    input_max[:] = np.maximum(input_max, inputs)

    flattened_fields = fields.reshape(fields.shape[0], -1)
    label_min[:] = np.minimum(label_min, np.min(flattened_fields, axis=1))
    label_max[:] = np.maximum(label_max, np.max(flattened_fields, axis=1))


def yaml_float(value):
    if np.isfinite(value):
        return format(float(value), ".9g")
    return "null"


def write_scaling_yaml(filename, input_min, input_max, label_min, label_max):
    with open(filename, mode="w", encoding="utf-8") as scaling_file:
        scaling_file.write("inputs:\n")
        for index, name in enumerate(INPUT_NAMES):
            scaling_file.write(f"  {name}:\n")
            scaling_file.write(f"    min: {yaml_float(input_min[index])}\n")
            scaling_file.write(f"    max: {yaml_float(input_max[index])}\n")

        scaling_file.write("labels:\n")
        for index, name in enumerate(LABEL_NAMES):
            scaling_file.write(f"  {name}:\n")
            scaling_file.write(f"    min: {yaml_float(label_min[index])}\n")
            scaling_file.write(f"    max: {yaml_float(label_max[index])}\n")


def write_sample(
    filename,
    inputs,
    fields,
    inputs_scaled,
    input_lower,
    input_upper,
):
    with h5py.File(filename, mode="w") as h5:
        inputs_dataset = h5.create_dataset("inputs", data=inputs.astype(np.float32), dtype="f")
        inputs_dataset.attrs["scaled_to_unit_range"] = inputs_scaled
        if inputs_scaled:
            inputs_dataset.attrs["unit_range_lower"] = input_lower.astype(np.float32)
            inputs_dataset.attrs["unit_range_upper"] = input_upper.astype(np.float32)
        h5.create_dataset(
            "fields",
            data=fields,
            dtype="f",
            compression="lzf",
            shuffle=True,
        )


def write_png(filename, fields):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    z_index = fields.shape[3] // 2
    magnitude = np.sqrt(np.sum(fields * fields, axis=0))
    center_slice = magnitude[:, :, z_index]

    fig, ax = plt.subplots()
    image = ax.imshow(
        center_slice.T,
        origin="lower",
        interpolation="none",
        aspect="equal",
    )
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(image, ax=ax, label="scaled displacement magnitude")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HDF5 samples of far-field 3D seismic waves.")
    parser.add_argument("-n", "--nsamples", default=16, type=int)
    parser.add_argument("-o", "--output-dir", default="fault_seismic_3d_data")
    parser.add_argument("--seed", default=12345, type=int)

    parser.add_argument("-nx", "--nx", default=32, type=int)
    parser.add_argument("-ny", "--ny", default=32, type=int)
    parser.add_argument("-nz", "--nz", default=32, type=int)
    parser.add_argument("--domain-size", default=1000.0, type=float)
    parser.add_argument("--source-margin", default=50.0, type=float)
    parser.add_argument("--exclusion-radius", default=25.0, type=float)

    parser.add_argument("--t-min", default=0.01, type=float)
    parser.add_argument("--t-max", default=0.20, type=float)
    parser.add_argument("--dt", default=0.01, type=float)
    parser.add_argument("--source-duration", default=0.04, type=float)

    parser.add_argument("--vp-min", default=2500.0, type=float)
    parser.add_argument("--vp-max", default=6500.0, type=float)
    parser.add_argument("--vp-vs-min", default=1.65, type=float)
    parser.add_argument("--vp-vs-max", default=1.85, type=float)
    parser.add_argument("--density-min", default=1800.0, type=float)
    parser.add_argument("--density-max", default=3300.0, type=float)
    parser.add_argument("--dip-min", default=float(np.deg2rad(5.0)), type=float)
    parser.add_argument("--dip-max", default=float(np.deg2rad(90.0)), type=float)
    parser.add_argument("--moment", default=1.0e13, type=float)
    parser.add_argument("--displacement-scale", default=40.0, type=float)
    parser.add_argument(
        "--scale-inputs",
        action="store_true",
        help="write each HDF5 inputs value min-max scaled to the unit range",
    )
    parser.add_argument("--write-png", action="store_true")

    return parser.parse_args()


def validate_args(args):
    if args.nx <= 1 or args.ny <= 1 or args.nz <= 1:
        raise ValueError("nx, ny, and nz must all be greater than 1")
    if args.nsamples < 0:
        raise ValueError("nsamples must be non-negative")
    if args.domain_size <= 0.0:
        raise ValueError("domain-size must be positive")
    if not (0.0 <= args.source_margin < 0.5 * args.domain_size):
        raise ValueError("source-margin must be in [0, domain-size / 2)")
    if args.exclusion_radius < 0.0:
        raise ValueError("exclusion-radius must be non-negative")
    if args.dt <= 0.0 or args.t_max < args.t_min:
        raise ValueError("time settings must satisfy dt > 0 and t-max >= t-min")
    if args.vp_min <= 0.0 or args.vp_max <= args.vp_min:
        raise ValueError("vp bounds must satisfy 0 < vp-min < vp-max")
    if args.vp_vs_min <= 1.0 or args.vp_vs_max <= args.vp_vs_min:
        raise ValueError("vp/vs bounds must satisfy 1 < min < max")
    if args.density_min <= 0.0 or args.density_max <= args.density_min:
        raise ValueError("density bounds must satisfy 0 < min < max")
    if not (0.0 < args.dip_min < args.dip_max <= 0.5 * np.pi):
        raise ValueError("dip bounds must satisfy 0 < dip-min < dip-max <= pi/2")
    if args.source_duration <= 0.0:
        raise ValueError("source-duration must be positive")
    if args.moment <= 0.0:
        raise ValueError("moment must be positive")
    if args.displacement_scale <= 0.0:
        raise ValueError("displacement-scale must be positive")
    if args.scale_inputs and args.t_max <= args.t_min:
        raise ValueError("scaled inputs require t-max > t-min")


def main():
    args = parse_args()
    validate_args(args)

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    times = np.arange(args.t_min, args.t_max + 0.5 * args.dt, args.dt)
    _, _, _, points = build_points(args.nx, args.ny, args.nz, args.domain_size)
    parameter_table = sample_parameter_table(args)
    input_lower, input_upper = input_scaling_bounds(args, max(args.t_max, times[-1]))
    local_input_min = np.full(len(INPUT_NAMES), np.inf, dtype=np.float64)
    local_input_max = np.full(len(INPUT_NAMES), -np.inf, dtype=np.float64)
    local_label_min = np.full(len(LABEL_NAMES), np.inf, dtype=np.float64)
    local_label_max = np.full(len(LABEL_NAMES), -np.inf, dtype=np.float64)

    if rank == 0:
        print(args, flush=True)
        print(
            "Generating",
            args.nsamples,
            "realizations and",
            args.nsamples * times.size,
            "time-step cases on grid",
            (args.nx, args.ny, args.nz),
            "with",
            times.size,
            "time steps on",
            size,
            "MPI ranks",
            flush=True,
        )

    for sample in range(args.nsamples):
        if sample % size != rank:
            continue

        params = parameter_table[sample]
        source = params[:3]
        strike = params[3]
        dip = params[4]
        rake = params[5]
        vp = params[6]
        vs = params[7]
        density = params[8]
        fields_flat = far_field_displacement(
            points,
            times,
            source,
            strike,
            dip,
            rake,
            vp,
            vs,
            density,
            args.moment,
            args.source_duration,
            args.exclusion_radius,
        )
        fields_flat *= args.displacement_scale

        for it, time_value in enumerate(times):
            case = sample * times.size + it
            fields = fields_flat[it].reshape(3, args.nx, args.ny, args.nz)
            inputs = np.array(
                [
                    source[0],
                    source[1],
                    source[2],
                    strike,
                    dip,
                    rake,
                    vp,
                    vs,
                    density,
                    time_value,
                ],
                dtype=np.float64,
            )
            if args.scale_inputs:
                inputs = scale_inputs_to_unit_range(inputs, input_lower, input_upper)
            inputs = inputs.astype(np.float32)
            threshold_fields_for_write(fields)
            update_ranges(
                inputs,
                fields,
                local_input_min,
                local_input_max,
                local_label_min,
                local_label_max,
            )
            filename = output_dir / ("case" + str(case).zfill(8) + ".h5")
            write_sample(
                filename,
                inputs,
                fields,
                args.scale_inputs,
                input_lower,
                input_upper,
            )
            if args.write_png:
                write_png(filename.with_suffix(".png"), fields)

        if rank == 0:
            print(f"{sample}/{args.nsamples}", flush=True)

    global_input_min = np.empty_like(local_input_min)
    global_input_max = np.empty_like(local_input_max)
    global_label_min = np.empty_like(local_label_min)
    global_label_max = np.empty_like(local_label_max)
    comm.Reduce(local_input_min, global_input_min, op=MPI.MIN, root=0)
    comm.Reduce(local_input_max, global_input_max, op=MPI.MAX, root=0)
    comm.Reduce(local_label_min, global_label_min, op=MPI.MIN, root=0)
    comm.Reduce(local_label_max, global_label_max, op=MPI.MAX, root=0)

    comm.Barrier()
    if rank == 0:
        write_scaling_yaml(
            output_dir / "scaling.yaml",
            global_input_min,
            global_input_max,
            global_label_min,
            global_label_max,
        )
        print("Done!", flush=True)


if __name__ == "__main__":
    main()
