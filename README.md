# Cupybara
<p align="center">
  <img src="https://github.com/jbrhm/CudaLibrary/blob/main/data/Cupybara.png"/>
</p>

## Installation
- Prerequisites:
- CMake **>=3.28** (you may have to install the kitware gpg key)
- Clone and `cd` into the repo
- `chmod +x ./build.sh`
- `./build.sh`

### Possible Errors:
- Make sure to this script is run in an environment managed by pip

## Usage:
For usages look in the testing folder for how to interface with the cupybara library

## Performance
**Performance Measured On 1000x1000 Matrices**
**Hardware: Ryzen 5 3600 & NVIDIA RTX 3070**

Cupybara Python Front End:
7.742296567166108 GFLOPS

Pytorch Python Front End:
0.5086737813983089 GFLOPS

Cupybara CUDA Back End:
421354 GFLOPS

**Performance Measured On 200x1 Vectors**
Cupybara Python Front End AVX:
2.755241411022795e-06 GFLOPS

Pytorch Python Front End:
1.6040844970771753e-06 GFLOPS

## Uninstall
- Run `pip uninstall cupybara`
- Run `sudo apt remove cupybara`
