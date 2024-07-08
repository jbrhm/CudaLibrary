import torch
import time
from enum import Enum
from pycublas import Matrix

SIZE = 1000

ITERATIONS = 1000

class Backends(Enum):
    cupybara = 1
    torch = 2

backend = Backends.torch

begin = time.time()
if backend == Backends.torch:
    # Create the tensors
    tensor1 = torch.rand(SIZE, SIZE)
    tensor2 = torch.rand(SIZE, SIZE)

    # Do the multiplications
    for i in range(0, ITERATIONS):
        torch.matmul(tensor1, tensor2)

elif backend == Backends.cupybara:
    # Create the matrices
    mat1 = Matrix.identity(SIZE, SIZE)
    mat2 = Matrix.identity(SIZE, SIZE)
    mat3 = Matrix.identity(SIZE, SIZE)

    # Do the multiplications
    for i in range(0, ITERATIONS):
        Matrix.multiply(mat1, mat2, mat3)


end = time.time()

GFLOPS = ((2 * SIZE * SIZE * SIZE) / (end - begin)) * 1e-9; 

print(f"Pytorch had {GFLOPS} GFLOPS")