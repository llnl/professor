# 3D fault dislocation example

## Description

Okada (1992) provides a set of closed-form analytical expressions for the deformation field surrounding a fault in an infinite, elastic half-space.
A fault dislocation is described via displacement along the strike, dip, and normal directions of the fault surface.
The resulting field is often complicated and is not necessarily symmetric, which makes this an excellent test case for the 3D generator.


The generate_fault_dislocation_data.py script can be used to generate a series 128x128x128x3 images using a Latin-Hypercube sampler.
Note: this script requires the **cutde** python package, which can be obtained via pip (e.g.: "python -m pip install cutde").
For large dataset requests, this process may take several minutes to run, so we recommend that you run this in an allocation.


```
usage: generate_fault_dislocation_data.py [--num-samples N] [--output-folder F]

options:
  -h, --help               Show this help message and exit
  -n N, --num-samples N    Number of random samples to generate
  -o F, --output-folder F  Path to place the resulting files
```


After generating the target data files, you may need to create the filelist.txt file that professor uses to index the results.
To do so, do the following:

```
cd /path/to/dataset
ls -1 *.h5 > filelist.txt
```
