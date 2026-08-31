# 3D data generation

The point-source generator and its physics, parameters, and HDF5 schema are
described in `data/point-source-diffusion-3d/README.md`. This page focuses on
creating a dataset suitable for the 3D training examples.

An example generated with `--write-png` shows the three orthogonal midplanes
of a typical three-source field:

![Three midplane slices through a synthetic diffusion field](images/example_data_slices.png)

## Generate the training dataset

The training jobs in this tutorial target 128 points per spatial dimension.
The following examples create 10,000 samples. Adjust the sample count and
resolution for the storage and compute available at your site.

=== "Flux"

    ```bash
    REPO_ROOT=/path/to/professor
    DATASET_DIR=/path/to/data/point_diffusion

    mkdir -p "${DATASET_DIR}"
    flux run -n 32 python \
        "${REPO_ROOT}/data/point-source-diffusion-3d/point_source_diffusion.py" \
        --num-samples 10000 \
        --resolution 128 \
        --normalize-parameters \
        --output "${DATASET_DIR}"

    find "${DATASET_DIR}" -maxdepth 1 -type f -name '*.h5' -printf '%f\n' \
        | sort > "${DATASET_DIR}/filelist.txt"
    ```

=== "Generic MPI"

    ```bash
    REPO_ROOT=/path/to/professor
    DATASET_DIR=/path/to/data/point_diffusion

    mkdir -p "${DATASET_DIR}"
    mpiexec -n 8 python \
        "${REPO_ROOT}/data/point-source-diffusion-3d/point_source_diffusion.py" \
        --num-samples 10000 \
        --resolution 128 \
        --normalize-parameters \
        --output "${DATASET_DIR}"

    find "${DATASET_DIR}" -maxdepth 1 -type f -name '*.h5' -printf '%f\n' \
        | sort > "${DATASET_DIR}/filelist.txt"
    ```


!!! tip

    We recommend running with the `--normalize-parameters` option for this tutorial.
    This can improve model scaling and performance.

## Verify the dataset

For the default three sources, each HDF5 file should contain:

| Dataset | Expected shape | Contents |
| --- | --- | --- |
| `inputs` | `(10,)` | Diffusivity, then x/y/z for sources 0, 1, and 2 |
| `fields` | `(1, 128, 128, 128)` | One scalar concentration-like diffusion field |

You can check a file without loading the full dataset using a simple python script:

```python
import glob
import h5py

dataset = glob.glob("/path/to/data/point_diffusion/*.h5")
sample = dataset[0]
with h5py.File(sample, "r") as h5:
    print(sample.name, h5["inputs"].shape, h5["fields"].shape)
```


Continue with [ML training](ml-training.md).
