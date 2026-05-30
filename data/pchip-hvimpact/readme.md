# Ensemble of PCHIP high velocity impact simulations

![Image_of_pchip_hvimpact_00](images/density_y_00.png){width=25%}
![Image_of_pchip_hvimpact_00](images/density_y_25.png){width=25%}
![Image_of_pchip_hvimpact_00](images/density_y_50.png){width=25%}


## Cite This Work

Jekel, C.F., Sterbentz, D.M., Stitt, T.M., Mocz, P., Rieben, R.N., White, D.A. and Belof, J.L., 2024. Machine learning visualization tool for exploring parameterized hydrodynamics. Machine Learning: Science and Technology, 5(4), p.045048. https://iopscience.iop.org/article/10.1088/2632-2153/ad8daa/meta 

```bib
@article{jekel2024machine,
  title={Machine learning visualization tool for exploring parameterized hydrodynamics},
  author={Jekel, CF and Sterbentz, DM and Stitt, TM and Mocz, P and Rieben, RN and White, DA and Belof, JL},
  journal={Machine Learning: Science and Technology},
  volume={5},
  number={4},
  pages={045048},
  year={2024},
  publisher={IOP Publishing}
}
```

## Description

The high velocity impact studies consist of an initially stationary copper target with a perturbation machined into the right-hand side (the copper-air interface), and a copper impactor with velocity of 2 km/s. As the shock wave reaches the interface perturbations, vorticity deposition occurs along the interface due to misalignments between pressure and density gradients at the perturbations. This creates the RMI that generally results in the jetting of the copper target material. The impactor is 1x9 cm and the target is nominally 0.5x9cm. These dimensions and velocities are chosen to be compatible with the two-stage gas gun at LLNL's High Explosive Applications Facility (HEAF)[1,2]. Whereas in actual HEAF experiments the impactor/target are circular with 9cm diameter, the simulations are 2D. The simulations begin at time t=0 with the impactor and target in contact with a discontinuous velocity. 

An example of the experimental setup of the PCHIP impact is shown here

![Example_pchip_setup](images/pchip_rhs.png){width=40%}

The target-air interface is parameterized with four parameters defining a Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) [3]. The PCHIP parameter range was [-0.25, 0.25] cm.

The hydrodynamic solutions were computed using a nominal mesh of 144x144 quadratic (`Q_2 Q_1`) elements, the mesh was morphed to have conformal interfaces between air and copper. During the simulation the fields density, velocity x, velocity y, energy, pressure, and the material indicator are projected onto a 1024x1024 Cartesian image and exported at 51 uniform timesteps from 0 to 15 μs.

[1] Michael Armstrong, Jeffrey Nguyen, Sylvie Aubry, William Schill, Jonathan Belof, and Hector Lorenzana. Use of shock wavefront curvature to modulate rmi jet growth. Bulletin of the American Physical Society, 67, 2022.

[2] Jeffrey Nguyen, Sylvie Aubry, Michael Armstrong, Andrew Hoff, Jonathan Belof, Hector Lorenzana, Matthew Staska, and Brandon LaLone. Modulation of richtmyer-meshkov instability in gas gun experiments. Bulletin of the American Physical Society, 67, 2022.

[3] Dane M. Sterbentz, Charles F. Jekel, Daniel A. White, Sylvie Aubry, Hector E. Lorenzana, and Jonathan L. Belof. Design optimization for Richtmyer–Meshkov instability suppression at  shock-compressed material interfaces. Physics of Fluids, 34(8):082109, 08 2022. ISSN 1070-6631. doi:10.1063/5.0100100. URL https://doi.org/10.1063/5.0100100.

## Scope And Content

The dataset is stored as 2,985 hdf5 files using gzip compression which takes up 2.1 TiB on disk. The files are named `simulation_NUMBER.h5`.

The shape of the full concatenated dataset is (2985, 51, 6, 1024, 1024) float32 values.

Each h5 file has two keys: `inputs` and `fields`.

The index description of `inputs`:

0. First PCHIP parameter in cm.
1. Second PCHIP parameter in cm.
2. Third PCHIP parameter in cm.
3. Fourth PCHIP parameter in cm.
4. The impact velocity in cm/μs.
5. Time of simulation in μs. 

Each `field` is a `1024 x 1024` image array of float32 values. The index description of `fields`: 

0. density 
1. velocity x 
2. velocity y 
3. energy 
4. pressure
5. materials 

H5py can be used to quickly access the data as multidimensional numpy arrays.

```python
import h5py

with h5py.File('simulation_0000.h5','r') as f:
    inputs = f['inputs'][:]
    fields = f['fields'][:]

print(inputs.shape)
print(fields.shape)
```
which should print:
```
(51, 6)
(51, 6, 1024, 1024)
```

## Obtain data

There are 5 archive files, each 450 GB. The files have the following names and sha256sums. The files will be available online.

0. pchip_hvimpact.tar.gz00
1. pchip_hvimpact.tar.gz01
2. pchip_hvimpact.tar.gz02
2. pchip_hvimpact.tar.gz03
2. pchip_hvimpact.tar.gz04

To untar this data, run the following command
```
cat pchip_hvimpact.tar.gz* | tar xzpvf -
```

## Author

C. F. Jekel, D. M. Sterbentz, T. M. Stitt, P. Mocz, R. N. Rieben, D. A. White, and J. L. Belof

## Funding

This work was supported by the LLNL-LDRD Program under Project No. 21-SI-006.
