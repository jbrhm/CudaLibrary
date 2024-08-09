# Cupybara
<p align="center">
  <img src="https://github.com/jbrhm/CudaLibrary/blob/main/data/Cupybara.png"/>
</p>

## Installation
- Clone and `cd` into the repo
- `chmod +x ./build.sh`
- `./build.sh`

## Uninstall
- Run `pip uninstall cupybara`
- Run `sudo apt remove cupybara`

## Performance
**Performance Measured On 1000x1000 Matrices**

Cupybara Python Front End:
2.201327628700629 GFLOPS

Pytorch Python Front End:
0.40395909001952424 GFLOPS

Cupybara CUDA Back End:
421354 GFLOPS
