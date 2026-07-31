from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray
from scipy.stats import qmc


FloatArray = NDArray[np.float32]
Float64Array = NDArray[np.float64]
Grid = Tuple[FloatArray, FloatArray, FloatArray]
SourcePositions = NDArray[np.float32]


def generate_field(
    source_positions: SourcePositions,
    source_strengths: FloatArray,
    diffusivity: float,
    time: float,
    grid: Grid,
    normalize: bool = True,
) -> FloatArray:
    if diffusivity <= 0:
        raise ValueError("diffusivity must be positive")
    if time <= 0:
        raise ValueError("time must be positive")

    x, y, z = grid
    field = np.zeros_like(x, dtype=np.float64)

    denominator = 4.0 * diffusivity * time
    prefactor = denominator ** (-1.5)

    for position, strength in zip(source_positions, source_strengths):
        sx, sy, sz = position

        radius_squared = (
            (x - sx) ** 2
            + (y - sy) ** 2
            + (z - sz) ** 2
        )

        field += strength * prefactor * np.exp(
            -radius_squared / denominator
        )

    if normalize:
        total_strength = float(np.sum(source_strengths))

        if total_strength <= 0:
            raise ValueError("total source strength must be positive")

        field /= total_strength

    return field.astype(np.float32)


def make_grid(
    resolution: int,
    domain_min: float = 0.0,
    domain_max: float = 1.0,
) -> Grid:
    if resolution < 2:
        raise ValueError("resolution must be at least 2")

    axis: FloatArray = np.linspace(
        domain_min,
        domain_max,
        resolution,
        dtype=np.float32,
    )

    return np.meshgrid(axis, axis, axis, indexing="ij")


def sample_parameter_distribution(
    num_samples: int,
    num_sources: int,
    seed: Optional[int] = None,
    domain_min: float = 0.1,
    domain_max: float = 0.9,
    diffusivity_min: float = 0.05,
    diffusivity_max: float = 0.5,
    time: float = 1.0,
    vary_strengths: bool = False,
    strength_min: float = 0.5,
    strength_max: float = 1.5,
) -> Tuple[FloatArray, float]:
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if num_sources < 1:
        raise ValueError("num_sources must be positive")
    if domain_min >= domain_max:
        raise ValueError("domain_min must be less than domain_max")
    if diffusivity_min <= 0 or diffusivity_min >= diffusivity_max:
        raise ValueError("invalid diffusivity range")
    if time <= 0:
        raise ValueError("time must be positive")

    dimensions = 3 * num_sources + 1

    if vary_strengths:
        dimensions += num_sources

    sampler = qmc.LatinHypercube(d=dimensions, seed=seed)
    unit_samples = sampler.random(n=num_samples)

    parameters = np.zeros(
        (num_samples, dimensions),
        dtype=np.float32,
    )

    column = 0

    for _ in range(num_sources):
        parameters[:, column] = (
            domain_min
            + unit_samples[:, column]
            * (domain_max - domain_min)
        )
        column += 1

        parameters[:, column] = (
            domain_min
            + unit_samples[:, column]
            * (domain_max - domain_min)
        )
        column += 1

        parameters[:, column] = (
            domain_min
            + unit_samples[:, column]
            * (domain_max - domain_min)
        )
        column += 1

    parameters[:, column] = (
        diffusivity_min
        + unit_samples[:, column]
        * (diffusivity_max - diffusivity_min)
    )
    column += 1

    if vary_strengths:
        parameters[:, column:] = (
            strength_min
            + unit_samples[:, column:]
            * (strength_max - strength_min)
        )

    return parameters, time


def unpack_parameters(
    parameters: FloatArray,
    num_sources: int,
    time: float,
) -> Tuple[SourcePositions, FloatArray, float, float]:
    position_count = 3 * num_sources

    positions = parameters[:position_count].reshape(
        num_sources,
        3,
    )

    diffusivity = float(parameters[position_count])
    strengths_start = position_count + 1

    if parameters.size > strengths_start:
        strengths = parameters[strengths_start:]
    else:
        strengths = np.ones(num_sources, dtype=np.float32)

    return positions, strengths, diffusivity, time


def write_sample(
    filename: Path,
    parameters: FloatArray,
    field: FloatArray,
) -> None:
    with h5py.File(filename, "w") as output:
        output.create_dataset("inputs", data=parameters)
        output.create_dataset(
            "fields",
            data=field[None, ...],
            dtype="float32",
        )


def write_sample_png(
    filename: Path,
    field: FloatArray,
) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        matplotlib_config_dir = Path("/tmp") / f"matplotlib-{os.getuid()}"
        matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(matplotlib_config_dir)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    midplane = field.shape[0] // 2
    slices = (
        (field[midplane, :, :], "x midplane", "y", "z"),
        (field[:, midplane, :], "y midplane", "x", "z"),
        (field[:, :, midplane], "z midplane", "x", "y"),
    )
    vmin = float(np.min(field))
    vmax = float(np.max(field))

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
        constrained_layout=True,
    )

    for axis, (image, title, xlabel, ylabel) in zip(axes, slices):
        plot = axis.imshow(
            image.T,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=(0.0, 1.0, 0.0, 1.0),
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

    figure.colorbar(
        plot,
        ax=axes,
        shrink=0.8,
        label="field",
    )
    figure.savefig(filename, dpi=150)
    plt.close(figure)


def generate_samples(args: argparse.Namespace) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        parameters, time = sample_parameter_distribution(
            num_samples=args.num_samples,
            num_sources=args.num_sources,
            seed=args.seed,
            domain_min=args.source_min,
            domain_max=args.source_max,
            diffusivity_min=args.diffusivity_min,
            diffusivity_max=args.diffusivity_max,
            time=args.time,
            vary_strengths=args.vary_strengths,
            strength_min=args.strength_min,
            strength_max=args.strength_max,
        )
    else:
        parameters = None
        time = None

    parameters = comm.bcast(parameters, root=0)
    time = comm.bcast(time, root=0)

    if parameters is None or time is None:
        raise RuntimeError("Failed to broadcast parameters")

    grid = make_grid(args.resolution)

    local_indices = np.array_split(
        np.arange(args.num_samples),
        size,
    )[rank]

    if rank == 0:
        print(
            f"Generating {args.num_samples} samples "
            f"with {args.num_sources} source(s)"
        )

    for global_index in local_indices:
        sample_parameters = parameters[global_index]

        positions, strengths, diffusivity, sample_time = (
            unpack_parameters(
                sample_parameters,
                args.num_sources,
                time,
            )
        )

        field = generate_field(
            source_positions=positions,
            source_strengths=strengths,
            diffusivity=diffusivity,
            time=sample_time,
            grid=grid,
            normalize=not args.no_normalization,
        )

        filename = output_dir / f"image_{global_index:06d}.h5"

        write_sample(
            filename=filename,
            parameters=sample_parameters,
            field=field,
        )

        if args.write_png:
            png_filename = output_dir / f"image_{global_index:06d}.png"
            write_sample_png(
                filename=png_filename,
                field=field,
            )

    comm.Barrier()

    if rank == 0:
        print("Done")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate analytical 3D multi-source diffusion fields."
    )

    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=64,
        help="Spatial resolution per dimension",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples",
    )
    parser.add_argument(
        "-k",
        "--num-sources",
        type=int,
        default=3,
        help="Number of diffusion sources",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./diffusion_data",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=1.0,
        help="Diffusion time",
    )
    parser.add_argument(
        "--diffusivity-min",
        type=float,
        default=0.01,
        help="Minimum diffusivity",
    )
    parser.add_argument(
        "--diffusivity-max",
        type=float,
        default=0.1,
        help="Maximum diffusivity",
    )
    parser.add_argument(
        "--source-min",
        type=float,
        default=0.1,
        help="Minimum source coordinate",
    )
    parser.add_argument(
        "--source-max",
        type=float,
        default=0.9,
        help="Maximum source coordinate",
    )
    parser.add_argument(
        "--vary-strengths",
        action="store_true",
        help="Sample independent source strengths",
    )
    parser.add_argument(
        "--strength-min",
        type=float,
        default=0.5,
        help="Minimum source strength",
    )
    parser.add_argument(
        "--strength-max",
        type=float,
        default=1.5,
        help="Maximum source strength",
    )
    parser.add_argument(
        "--no-normalization",
        action="store_true",
        help="Keep physical amplitude instead of normalizing",
    )
    parser.add_argument(
        "--write-png",
        action="store_true",
        help="Render x, y, and z midplane slices for each sample",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    generate_samples(args)


if __name__ == "__main__":
    main()
