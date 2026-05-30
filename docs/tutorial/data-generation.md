# Data generation

There are two scripts to generate the data for the analytical expression. They are located in the professor repo in [data/analytical-example](data/analytical-example)

-   fouriermodes_study.py
-   fouriermodes_study_mpi.py

with the mpi version implementing an embarrassingly parallel generation.

We will generate the dataset with 10 samples per dimension for a total of 10,000 image arrays.

## fouriermodes_study_mpi.py

You can grab the latest file from the repo, or quickly make a new file with this contents.


```python title="Example data generation script"
--8<-- "data/analytical-example/fouriermodes_study_mpi.py:4:"
```



## executing the mpi version

=== "tuolumne.llnl.gov"

    We will request an interactive allocation, make a dataset folder on lustre, cd into it, create the data, then create the filelist.

    ```bash
    # request an interactive allocation
    flux alloc -N 1 -q pdebug

    # make a dataset folder
    mkdir -p /p/lustre5/$USER/analytical_data_test
    cd /p/lustre5/$USER/analytical_data_test

    # generate the data
    flux run -N 1 -n 64 /usr/workspace/prof/mlvenv/bin/python /usr/workspace/prof/professor/data/analytical-example/fouriermodes_study_mpi.py --nsamp_per_dim 10

    # create the filelist.txt
    ls -1 *.h5 > filelist.txt
    ```

    You can type `exit` or press `ctrl + d` to leave the allocation.

=== "generic"

    CD into the directory you want to create the dataset, then execute the python file

    ```bash
    # make a dataset folder
    mkdir -p analytical_data
    cd analytical_data

    # generate the data
    mpirun -n 8 python fouriermodes_study_mpi.py --nsamp_per_dim 10

    # create the filelist.txt
    ls -1 *.h5 > filelist.tx
    ```


You should now have 10,000 .h5 files, .png files, and a filelist.txt in that directory! Each .h5 file contains an image array that is `1 x 512 x 512`.

## Notes about compression

With HDF5 it is possible to store your datasets with compression filters such as gzip. For example:

```python
with h5py.File(output_name + ".h5", mode="w") as allimages:
    allimages.create_dataset(
        "inputs",
        data=inputs,
        dtype="f",
        compression="gzip",
    )
```

While compression can greatly reduce the overall storage of these datasets (in some cases by 50%), it unfortunately creates a huge CPU burden when loading the data. The training paradigm used in this work is data distributed, which continuously loads data from storage. You do not want every data read to require CPU time to decompresses the data, as the CPU decompression can quickly become the bottleneck in your training. We have observed cases where using a light gzip filter can slow down the training time of an epoch by up to 4x.

It is thus recommended to not use compression in the hdf5 training datasets.
