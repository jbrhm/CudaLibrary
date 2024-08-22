from cupybara.cupybara import Vector

# Regular imports
import time
from enum import Enum

SIZE = 200

ITERATIONS = 1000

begin = time.time()

vec1 = Vector.zeros(SIZE)
vec2 = Vector.zeros(SIZE)
vec3 = Vector.zeros(SIZE)

# Sync the vectors to AVX
vec1.syncAVX()
vec2.syncAVX()
vec3.syncAVX()

# Do the multiplications
for i in range(0, ITERATIONS):
    Vector.add(vec1, vec2, vec3)

end = time.time()

#         vvv one add + one assign = 2 flops
GFLOPS = ((2 * SIZE) / (end - begin)) * 1e-9; 

print(f"{GFLOPS} GFLOPS")
