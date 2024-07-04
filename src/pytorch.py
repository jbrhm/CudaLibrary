import torch
import time

SIZE = 40

ITERATIONS = 1000

tensor1 = torch.rand(SIZE, SIZE)
tensor2 = torch.rand(SIZE, SIZE)

begin = time.time()

for i in range(0, ITERATIONS):
    torch.matmul(tensor1, tensor2)

end = time.time()

GFLOPS = ((2 * SIZE * SIZE * SIZE) / (end - begin)) * 1e-9; 

print(f"Pytorch had {GFLOPS} GFLOPS")