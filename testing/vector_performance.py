from cupybara.cupybara import Vector

# Regular imports
import time
from enum import Enum

class Backends(Enum):
    host = 1
    avx = 2

backend = Backends.avx

SIZE = 200

ITERATIONS = 10000

vec1 = Vector.zeros(SIZE)
vec2 = Vector.zeros(SIZE)
vec3 = Vector.zeros(SIZE)

# Sync the vectors to AVX
if backend == Backends.avx:
    vec1.syncAVX()
    vec2.syncAVX()
    vec3.syncAVX()

begin = time.time()

# Do the multiplications
for i in range(0, ITERATIONS):
    Vector.add(vec1, vec2, vec3)

end = time.time()

#         vvv one add + one assign = 2 flops
GFLOPS = ((2 * SIZE) / (end - begin)) * 1e-9; 

print(f"{GFLOPS} GFLOPS")
