# ML training

You now have a valid dataset that you can begin a distributed training run!

## Submitting your distributed training batch job

=== "tuolumne.llnl.gov"

    ```bash title="Here is an example flux batch job for Tuolumne"
    --8<-- "examples/analytical/analytical.flux"
    ```

    Save this example as analytical.flux.

    If you are still in an interactive allocation, you can begin training in it by

    ```bash
    bash analytical.flux
    ```
    otherwise sbumit the batch job with
    ```bash
    flux batch analytical.flux
    ```
    Which should take about 30 minutes to run to completion.


=== "generic"

    ```bash title="Here is an example Slurm job"
    --8<-- "examples/analytical/analytical.sh"
    ```

    Save this example as analytical.sh.

    This should take about 30 minutes to run to completion.



## Expected output

As the job is running, you will see some print out like the following. The following is a detailed layer-by-layer explanation of your ML model. It details the size of the intermediate representation at every layer, as well as the number of parameters associated with each layer. You can see this ML model has a total of 61,419,266 parameters.

```text
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Generator                                [42, 1, 512, 512]         --
├─Sequential: 1-1                        [42, 1, 512, 512]         --
│    └─Identity: 2-1                     [42, 4, 1, 1]             --
│    └─ConvTranspose2d: 2-2              [42, 1024, 4, 4]          65,536
│    └─BatchNorm2d: 2-3                  [42, 1024, 4, 4]          2,048
│    └─ReLU: 2-4                         [42, 1024, 4, 4]          --
│    └─ConvTranspose2d: 2-5              [42, 1024, 8, 8]          16,777,216
│    └─BatchNorm2d: 2-6                  [42, 1024, 8, 8]          2,048
│    └─ReLU: 2-7                         [42, 1024, 8, 8]          --
│    └─ConvTranspose2d: 2-8              [42, 1024, 16, 16]        16,777,216
│    └─BatchNorm2d: 2-9                  [42, 1024, 16, 16]        2,048
│    └─ReLU: 2-10                        [42, 1024, 16, 16]        --
│    └─ConvTranspose2d: 2-11             [42, 1024, 32, 32]        16,777,216
│    └─BatchNorm2d: 2-12                 [42, 1024, 32, 32]        2,048
│    └─ReLU: 2-13                        [42, 1024, 32, 32]        --
│    └─ConvTranspose2d: 2-14             [42, 512, 64, 64]         8,388,608
│    └─BatchNorm2d: 2-15                 [42, 512, 64, 64]         1,024
│    └─ReLU: 2-16                        [42, 512, 64, 64]         --
│    └─ConvTranspose2d: 2-17             [42, 256, 128, 128]       2,097,152
│    └─BatchNorm2d: 2-18                 [42, 256, 128, 128]       512
│    └─ReLU: 2-19                        [42, 256, 128, 128]       --
│    └─ConvTranspose2d: 2-20             [42, 128, 256, 256]       524,288
│    └─BatchNorm2d: 2-21                 [42, 128, 256, 256]       256
│    └─ReLU: 2-22                        [42, 128, 256, 256]       --
│    └─ConvTranspose2d: 2-23             [42, 1, 512, 512]         2,048
│    └─AlphaLinear: 2-24                 [42, 1, 512, 512]         2
==========================================================================================
Total params: 61,419,266
Trainable params: 61,419,266
Non-trainable params: 0
Total mult-adds (T): 5.30
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 10977.02
Params size (MB): 245.68
Estimated Total Size (MB): 11222.70
==========================================================================================
```

Also note that the above printout estimates how much total VRAM is required to do training with a batch size of 42 is 11222.70 MB per GPU. These are just ballpark estimates, but they are incredibly helpful for tweaking the batch size to maximize GPU memory (thus helping you optimally load the GPUs).

As the training is happening you'll see printout like the following:

```text
Number of steps in one epoch: 60
0000 0.525711 0.000000 4.674845 0.000000 172.295547 0.000000 0.000000 0.000000 59.671207
0001 0.276103 0.000000 0.183241 0.000000 8.001334 0.000000 0.000000 0.000000 56.302436
0002 0.262850 0.000000 0.167671 0.000000 5.933118 0.000000 0.000000 0.000000 56.320300
0003 0.252026 0.000000 0.156788 0.000000 4.254933 0.000000 0.000000 0.000000 56.026898
0004 0.212558 0.000000 0.122664 0.000000 3.897465 0.000000 0.000000 0.000000 55.930470
0005 0.178089 0.000000 0.098191 0.000000 3.169892 0.000000 0.000000 0.000000 55.333469
0006 0.159863 0.000000 0.084373 0.000000 3.261051 0.000000 0.000000 0.000000 54.952650
0007 0.082685 0.000000 0.022560 0.000000 2.944603 0.000000 0.000000 0.000000 54.824785
0008 0.054999 0.000000 0.007944 0.000000 2.799326 0.000000 0.000000 0.000000 54.966991
0009 0.046733 0.000000 0.006224 0.000000 2.911853 0.000000 0.000000 0.000000 54.687456
0010 0.041503 0.000000 0.004772 0.000000 2.198126 0.000000 0.000000 0.000000 54.696358
0011 0.038149 0.000000 0.004192 0.000000 2.334234 0.000000 0.000000 0.000000 54.692887
```

The columns from left to right are:

-   epoch number
-   Mean absolute error on training data
-   N/A
-   Mean squared error on training data
-   N/A
-   L-infinity error on training data
-   N/A
-   N/A
-   N/A
-   Time per epoch

So notice how all of those errors are going down every epoch? Your ML model is learning the mapping of the beta parameters to the full-field solutions!

In the folder where the distributed training is being run

=== "tuolumne.llnl.gov"

    `/p/lustre5/$USER/models/analytical/runs`

=== "generic"

    `/lustre/scratch1/$USER/data/analytical_data`

you will see a folder `filelist_42_1024_128_1e-5` which contains the following

```text
|-- 0000.pt
|-- 0005.pt
|-- 0010.pt
|-- 0015.pt
|-- 0020.pt
|-- 0024.pt
|-- args.txt
|-- events.out.tfevents.1730496525.lassen24.63595.0
|-- events.out.tfevents.1730496525.lassen24.63596.0
|-- events.out.tfevents.1730496525.lassen24.63597.0
|-- events.out.tfevents.1730496525.lassen24.63598.0
|-- events.out.tfevents.1730496576.lassen24.65166.0
|-- events.out.tfevents.1730496576.lassen24.65167.0
|-- events.out.tfevents.1730496576.lassen24.65168.0
|-- events.out.tfevents.1730496576.lassen24.65169.0
`-- model.info
```

Those files have the following meaning:

-   `.pt` files are checkpoints of the ML model containing both the model weights and optimizer state
-   `args.txt` is the command line arguments use for training
-   `events` files are the logs for tensorboard
-   `model.info` is a print out of the layer by layer details of the model

If you see all this, your training has successfully completed!
