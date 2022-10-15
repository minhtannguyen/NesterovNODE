# Nesterov Neural Ordinary Differential Equations

This is the official implementation of Nesterov Neural Ordinary Differential Equations. 

## Main requirements
The code base in in Python. The following packages are required:
- torch
- torchvision
- torchdiffeq
- tqdm
- imageio
- einops
These packages can be installed with the following command:
```
pip install torch torchvision torchdiffeq tqdm imageio einops
```
Additionally, if you want to run the visualization code, the ```numpy```, ```pandas```, ```matplotlib``` are also required. These are well-supported package, so if there's any problem with the installation process, please refer to the official installation guide.

## Details for each experiments can be found in
### MNIST
```
mnist/README.md
```
### CIFAR
```
cifar/README.md
```
### Point Cloud
```
point-cloud/README.md
```
### Walker 2D
```
walker2d/README.md
```
### Silverbox Initialization
```
README-sb.md
```