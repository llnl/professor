# 3D Point Source Diffusion

## Description

Consider the 3D point source diffusion equation:


$$
  C(r,t) = \frac{Q}{(4 \pi D t)^{3/2}} exp \left( - \frac{r^2}{4Dt} \right)
$$

where $C$ is concentration, $t$ is time, $Q$ is the initial quantity, and $D$ is diffusivity.
This example uses the *point_source_diffusion.py* script to generate an ensemble of images for the above parameters and a source location in a unit-cube.
The generated images are of size (R, R, R, 1), and the ensemble is selected using a Latin-Hypercube sampler.


```
usage: python point_source_diffusion.py [--num-samples N] [--output-folder F] [--output F]

options:
  -h, --help               Show this help message and exit
  -n N, --num-samples N    Number of random samples to generate
  -o F, --output F         Path to place the resulting files
  -r R, --resolution R     Image resolution
```


After generating the target data files, you may need to create the filelist.txt file that professor uses to index the results.
To do so, do the following:

```
cd /path/to/dataset
ls -1 *.h5 > filelist.txt
```
