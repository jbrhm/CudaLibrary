# Cupybara
<p align="center">
  <img src="https://github.com/jbrhm/CudaLibrary/blob/main/data/Cupybara.png"/>
</p>

## Installation
- Download the latest release of Cupybara using the release tab on the right
- Cd to the directory the wheel and package were downloaded
- Run `sudo pip install --force-reinstall cupybara-1.0.2-py3-none-any.whl`
- Run `sudo apt install -f ./cupybara_1.0.2.deb`

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
