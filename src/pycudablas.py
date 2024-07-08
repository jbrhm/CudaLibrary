from ctypes import CDLL, c_longlong
  
lib = CDLL('./build/libcudablas.so') 

# new_matrix
lib.new_matrix.restype = c_longlong

# sync
lib.sync.argtypes = (c_longlong,)

# print
lib.print.argtypes = (c_longlong,)

# load the library 
class Matrix:
    # This is a pointer to a matrix class in C/C++/CUDA
    matrix: c_longlong

    def __init__(self):
        self.matrix = lib.new_matrix(4, 4)

    def print(self):
        lib.print(self.matrix)

    def sync(self):
        lib.sync(self.matrix)

    # Overloaded Matrix Multiplication
    def __mul__(self, other):
        print("Not implemented yet")

mat = Matrix()
mat.print()
