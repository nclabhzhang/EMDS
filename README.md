# EMDS
This repository provides the implemention for the paper Equivariant Score-based Generative Diffusion Framework for 3D Molecules.

Please cite our paper if our datasets or code are helpful to you ~ 😊

## Requirements
* Python 3.7
* pytorch 1.12.0
* rdkit 2022.3.5
* pyparsing 2.4.7


## Dataset
* QM9: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz


## Running Experiments
### Preparations
To preprocess the 3D molecular dataset QM9 for training model, run the following command:
```bash
python data/preprocess_3d.py
python data/split_generators.py
```

### Configs
The configurations are provided on the config/ directory in YAML format.

### Training
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --type train --config train --seed 42
```

### Generation and Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --type sample --config sample
```

## Acknowledgements
EMDS builds upon the source code from the projects [GDSS](https://github.com/harryjo97/gdss), [G-SphereNet](https://github.com/divelab/DIG/tree/dig-stable) and [EDM](https://github.com/ehoogeboom/e3_diffusion_for_molecules).

We thank their contributors and maintainers!

## Citation
Please cite our paper if our datasets or code are helpful to you.

H. Zhang, Y. Liu, X. Liu, C. Wang, and M. Guo, "Equivariant Score-based Generative Diffusion Framework for 3D Molecules", BMC Bioinformatics 25, 203 (2024), DOI: 10.1186/s12859-024-05810-w

