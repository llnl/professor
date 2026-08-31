# 3D Point-Source Diffusion

## Description

`point_source_diffusion.py` generates analytical 3D scalar diffusion fields
from one or more instantaneous point sources in a unit cube. For source \(i\), with position
\(\mathbf{x}_i\), strength \(Q_i\), diffusivity \(D\), and snapshot time
\(t\):

\[
C(\mathbf{x},t) = \sum_{i=1}^{N_s}
Q_i (4Dt)^{-3/2}
\exp\!\left(-\frac{\lVert\mathbf{x}-\mathbf{x}_i\rVert^2}{4Dt}\right).
\]


## Sampled parameters

The script uses a Latin-hypercube sampler to generate one shared diffusivity
and a 3D position for every source in each sample. By default it generates
three sources with:

| Parameter | Default range |
| --- | ---: |
| Diffusivity, `D` | 0.01 to 0.1 |
| Each source coordinate | 0.1 to 0.9 |
| Snapshot time | 1.0 (fixed for the run) |
| Each source strength | 1.0 (fixed) |

With `--vary-strengths`, each source strength is also sampled independently
from `--strength-min` to `--strength-max`, whose defaults are 0.5 and 1.5.
The snapshot time is fixed across the dataset and is not stored in the input
vector.

## Running the generator

To generate the dataset, run the following:

```bash
python data/point-source-diffusion-3d/point_source_diffusion.py \
    --num-samples 100 \
    --output diffusion_data
```

The data can also be generated using MPI.  For example:

```bash
mpiexec -n 4 python \
    data/point-source-diffusion-3d/point_source_diffusion.py \
    --num-samples 400 \
    --output diffusion_data
```

The command-line options for the data generation script include:

```text
-r, --resolution R           Grid points per dimension (default: 64)
-n, --num-samples N          Number of samples (default: 100)
-k, --num-sources K          Sources in each sample (default: 3)
-o, --output DIR             Output directory (default: ./diffusion_data)
--time T                     Snapshot time (default: 1.0)
--diffusivity-min D          Minimum sampled diffusivity (default: 0.01)
--diffusivity-max D          Maximum sampled diffusivity (default: 0.1)
--source-min X               Minimum sampled source coordinate (default: 0.1)
--source-max X               Maximum sampled source coordinate (default: 0.9)
--vary-strengths             Sample an independent strength for each source
--strength-min Q             Minimum sampled strength (default: 0.5)
--strength-max Q             Maximum sampled strength (default: 1.5)
--normalize-parameters       Scale values written to `inputs` into [0, 1]
--write-png                  Render three midplane slices for every sample
-h, --help                   Show the complete command-line help
```


## Output format

Each sample is written as an hdf5 format file withe following entries:

- `inputs`: a vector ordered as
  `[diffusivity, source_0_x, source_0_y, source_0_z, ...]`;
- `fields`: the scalar field with shape `(1, R, R, R)`.

When `--vary-strengths` is enabled, the input vector also has the source strength terms at the end: `[diffusivity, source_0_x, source_0_y, source_0_z, ..., source_0_strength, source_1_strength, ...]` .
The output directory also contains `parameter_ranges.yaml`, which records the range of the as-written model input parameters.

To create the file list used by Professor, run:

```bash
cd /path/to/diffusion_data
ls -1 *.h5 > filelist.txt
```
