# Far-Field 3D Seismic Waves From a Point Fault

## Description

`fault_seismic_3d_dataset.py` generates synthetic, three-component seismic
displacement fields for a point double-couple source in a homogeneous,
isotropic medium. The fields are evaluated on a regular Cartesian grid in a
cubic domain. Coordinates use an east-north-up (ENU) convention: `x` is east,
`y` is north, and `z` is up.
This script uses the far-field assumption and a gaussian pulse for the moment rate function.

Note: this case produces data that are challenging to fit with typical ML models due to zero-value inflation.
These are being used to test in-development features to improve professor scaling/training performance.


## Sampled parameters

Each realization samples eight parameters with a Latin hypercube:

| Parameter | Default range | Meaning |
| --- | ---: | --- |
| `source_x`, `source_y`, `source_z` | 0-1000 m | Source position within the cube |
| `strike` | 0 to 2 pi rad | Clockwise from north |
| `dip` | 5 to 90 degrees | Down from horizontal; CLI bounds are in radians |
| `rake` | -pi to pi rad | Slip direction in the fault plane, measured from strike |
| `vp` | 2500 to 6500 m/s | P-wave speed |
| `vp_vs` | 1.65 to 1.85 | Ratio \(v_p/v_s\) |

Density, scalar moment, source duration, and the final displacement multiplier
are fixed across a run and can be set on the command line. Their defaults are
2500 kg/m³, \(10^{13}\) N m, 0.04 s, and 40, respectively.

## Running the generator

To generate the data you can run the following:

```bash
python data/seismic-waves-3d/fault_seismic_3d_dataset.py \
    --nsamples 16 \
    --output-dir fault_seismic_3d_data
```

The script uses MPI and may also be launched on multiple ranks, for example:

```bash
mpiexec -n 4 python \
    data/seismic-waves-3d/fault_seismic_3d_dataset.py \
    --nsamples 64 \
    --output-dir fault_seismic_3d_data
```

The principal options are:

```text
-n, --nsamples N              Number of source realizations (default: 16)
-o, --output-dir DIR          Output directory
-nx NX, -ny NY, -nz NZ       Number of grid points along each axis
--domain-size METERS          Cube side length (default: 1000)
--t-min SEC                   First output time (default: 0.01)
--t-max SEC                   Last requested output time (default: 0.20)
--dt SEC                      Output time step (default: 0.01)
--source-duration SEC         Gaussian pulse duration (default: 0.04)
--vp-min MPS, --vp-max MPS    P-wave speed bounds
--vp-vs-min R, --vp-vs-max R  P-to-S speed-ratio bounds
--density KG_PER_M3           Homogeneous density
--dip-min RAD, --dip-max RAD  Fault-dip bounds
--moment NM                   Scalar seismic moment
--scale-inputs                Normalize model inputs
--write-png                   Write a displacement-magnitude preview per case
--split-z                     Write each z plane as a separate 2D case
```

Run the script with `--help` for the complete command-line help.

## Output format

The output times are generated from `t-min` through `t-max` at increments of
`dt`, including the upper endpoint when it lies on the resulting time grid.
For each realization and output time, the script writes an hdf5 format file containing:

- `inputs`: a float32 vector with
  `[source_x, source_y, source_z, strike, dip, rake, vp, vp_vs, time]`;
- `fields`: a compressed float32 array with shape `(3, nx, ny, nz)`, ordered as
  the x-, y-, and z-displacement components.

With `--split-z`, each z plane is stored in its own file. In that mode,
`fields` has shape `(3, nx, ny)`, and normalized `z_position` in the range
`[0, 1]` is appended to `inputs`. A run therefore creates
`nsamples * ntimes` HDF5 cases normally, or
`nsamples * ntimes * nz` cases with `--split-z`.

The output directory also contains `scaling.yaml`, which records the observed minimum and
maximum of each stored input and displacement component.

To create the file list used by Professor, run:

```bash
cd /path/to/fault_seismic_3d_data
ls -1 *.h5 > filelist.txt
```
