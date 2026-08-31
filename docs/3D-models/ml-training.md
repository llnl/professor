# 3D ML training

The generated synthetic dataset should now contain a set of HDF5 files and a
corresponding list of target files called `filelist.txt`. This section covers
the relevant Professor 3D generator settings and provides complete example
Flux scripts.


## 3D Generator Options

Similar to the 2D tutorial, we will use the `prof-trainer` utility to train a
3D generator model. These are the most common command-line options needed for
this tutorial:

| Argument | Purpose in this tutorial |
| --- | --- |
| `--generator-type` | Selects `3D-triplane` or `3D-voxel` |
| `--x_kernel`, `--y_kernel`, `--z_kernel` | Sets the initial learned spatial extent; all are 4 for the cubic data |
| `--upscale-type` | Uses `nearest` interpolation followed by convolution in these examples; `transpose` and `linear` are also supported |
| `--max_feature`, `--min_feature` | Bounds intermediate channel counts and controls much of the memory/capacity tradeoff |
| `--keys concentration` | Declares one output-channel name for logs and visualization; the HDF5 dataset itself remains named `fields` |
| `--dataset_type 0` | Reads one realization from each HDF5 file |
| `--run_directory` | Places checkpoints and logs in a predictable directory and resumes from its newest checkpoint |
| `--vis-config` | Creates a configuration for `prof-dash-gui` and updates its checkpoint path during training |
| `--batch_multiplier N` | Increases the effective batch size through accumulation |


In the following section, we will cover two different types of 3D generator models.


## 3D Triplane Generator Model

At its core, the 3D Triplane generator contains a set of child 2D generator models that are tasked with creating
projections of the data along the XY, XZ, and YZ planes.
The final layers of the model broadcast these projections together and then re-constructs the target image.
This approach significantly reduces the memory requirements of the model during training and model evaluation.

Here is an example submission script for the Triplane model:

```bash title="point_diffusion_triplane.flux"
--8<-- "examples/3D-models/point_diffusion_triplane.flux"
```


## 3D Voxel Generator Model

The 3D Voxel generator is an extension of the 2D generator model into 3D space.
This type of model has the potential for more predictive power than the triplane generator,
but is slower and requires significantly more memory during training.
To address this challenge, this model does the following:

1) Implements separable 3D convolutions for deep model layers.  Instead of a single expensive 3D convolution operation, this strategy splits the convolution into
   two steps: a (1, X, Y, Z) spatial convolution and a (C, 1, 1, 1) convolution along the channel.  This greatly improves the speed of the operation at a minor cost to model complexity.
2) Implements batch accumulation.  3D tensors require a large amount of memory during training, which can limit the maximum batch size.  To address this, we use the `--batch-accumulation`
   option to increase the effective batch size of the problem and maintain training stability.


Here is an example submission script for the Voxel model:

```bash title="point_diffusion_voxel.flux"
--8<-- "examples/3D-models/point_diffusion_voxel.flux"
```


!!! tip

    If the default batch size for this problem causes your system to run out of memory,
    try reducing the value and increasing the value of `--batch_multiplier`



## Expected outputs

If the model training was successful, your model output directory should contain some of the following:

- `*.pt` files contain model and optimizer checkpoints.
- `events.out.tfevents.*` contains TensorBoard logs.
- `model.info` records the layer-by-layer model summary and memory estimate.
- `args.txt` records the training arguments.
- `point_diffusion.yaml` is the visualization configuration.


Continue with [ML visualization](ml-visualization.md).
