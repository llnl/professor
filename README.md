# Professor
```
 ____  ____   ___   _____  ___  _____ _____  ___   ____  
|    \|    \ /   \ |     |/  _]/ ___// ___/ /   \ |    \ 
|  o  )  D  )     ||   __/  [_(   \_(   \_ |     ||  D  )
|   _/|    /|  O  ||  |_|    _]\__  |\__  ||  O  ||    / 
|  |  |    \|     ||   _]   [_ /  \ |/  \ ||     ||    \ 
|  |  |  .  \     ||  | |     |\    |\    ||     ||  .  \
|__|  |__|\_|\___/ |__| |_____| \___| \___| \___/ |__|\_|
                                                         
```
Professor (commonly abbreviated as Prof) is a person who professes to be an expert in some art or science.

Professor is a tool to help you study complicated physical phenomena by providing tools to 1) fit machine learning models to 2D image arrays from simulations and 2) interactively explore these machine learning models in real time. Professor is most useful when studying ensembles of simulations.

A typically workflow would look like:
1. A user is interested in how parameters A, B, C, & D influence some complicated physics
2. User setups up parameterized simulations to study ABCD and the results of these simulations to be image arrays (a 2d matrix of float32 values)
3. User runs an ensemble of simulations studying ABCD creating a dataset of image arrays
4. User runs `prof-trainer` to fit a machine learning model to learn the mapping from [A,B,C,D] to the image arrays
5. User then uses `prov-vis` to interactively explore the machine learning model in real time, gaining their insight into how those parameters influence the physics
6. Go profess your idea about ABCD!

## Installation

Clone and then cd into this repo.

The default installation only checks the minimal dependencies. This process was setup to allow users to install some parts (e.g. just distributed training, or just machine learning visualization).

To install just the machine learning training:
```bash
python -m pip install .[trainer]
```

To install just the gui machine learning visualiztion:
```bash
python -m pip install .[gui]
```

To install both the gui and machine learning training:
```bash
python -m pip install .[all]
```

For developers, we recommend to install `dev` which installs everything and a few developer tools.
```bash
python -m pip install -e .[dev]
```

## Requirements
The basic requirements are:
```
torch>=1.12.1
torchinfo>=1.8.0
omegaconf>=2.3.0
schema>=0.7.5
torchlayers
kornia
wheel
```
## Datasets

Some example datasets will be hosted TBD. This will allow users to reproduce our machine learning model results, and to experiment with other models that may improve accuracy.

[rayleigh-taylor-single](data/rayleigh-taylor-single) is a 328 GiB dataset of shape (2000, 51, 6, 768, 256) studying the Rayleigh-Taylor instability of two fluids mixing.

[pchip-hvimpact](data/pchip-hvimpact) is a 2.1 TiB dataset of shape (2000, 51, 6, 1024, 1024) studying the Richtmyer-Meshkov instability as a shockwave passes through copper. 

## Examples

### Training

The [examples](examples) folder contains example batch scripts to fit the machine learning model on the Lassen HPC. These training runs utilized distributed data parallelism to train the ML model using many GPUs using `prof-trainer`.

The distributed training will create folders like the following:
```
runs
├── Jul10_14-08-27_rzadams1003GEN_NGPU_4_LR_0.0019200000000000003_Batch_48_Loss_l1_MaxFeat_1024_MinFeat_128_keys_density,velocity_x,velocity_y,pressure,energy,materials
│   ├── 0000.pt
│   ├── 0020.pt
│   ├── 0040.pt
│   ├── 0060.pt
│   ├── 0080.pt
│   ├── 0099.pt
│   ├── args.txt
│   ├── events.out.tfevents.1720645707.rzadams1003.1180058.0
│   └── model.info
├── Jun22_11-50-13_rzvernal14GEN_NGPU_40_LR_0.019200000000000002_Batch_48_Loss_l1_MaxFeat_4096_MinFeat_256_keys_density,velocity_x,velocity_y,pressure,energy,materials
│   ├── 0000.pt
│   ├── 0020.pt
│   ├── 0040.pt
│   ├── args.txt
│   ├── events.out.tfevents.1719082213.rzvernal14.901987.0
│   └── model.info
├── Jun22_11-50-14_rzvernal24GEN_NGPU_40_LR_0.019200000000000002_Batch_48_Loss_l1_MaxFeat_4096_MinFeat_512_keys_density,velocity_x,velocity_y,pressure,energy,materials
│   ├── 0000.pt
│   ├── 0020.pt
│   ├── 0040.pt
│   ├── args.txt
│   ├── events.out.tfevents.1719082214.rzvernal24.3706721.0
│   └── model.info
```
where the `.pt` files are model checkpoints, `args.txt` are the command line arguments that went into `prof-trainer`, `model.info` is the layer by layer information of the machine learning model, and `events*` are the tensorboard training history.

You can use tensorboard to visualize the results from many training runs with
```
tensorboard --logdir runs
```

### Visualization

The [configs](configs) folder contains example yaml files to allow you to visualize the machine learning models. These can be run like the following:
```bash
prof-vis configs/configs/pchip.yaml
```
You must point the `checkpoint` key within the yaml file to the path of the machine learning checkpoint, as well as provide details for the expected inputs to the model.

Example model checkpoints will be hosted TBD.

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

## Funding

This work was supported by the LLNL-LDRD Program under Project No. 21-SI-006.

## License

see [LICENSE](https://github.com/LLNL/professor/blob/main/LICENSE) and [NOTICE](https://github.com/LLNL/professor/blob/main/NOTICE)

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

LLNL-CODE-2015991
