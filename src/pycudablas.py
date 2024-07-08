from ctypes import CDLL, c_longlong, c_uint, POINTER, c_float

import numpy as np
  
lib = CDLL('./build/libcudablas.so') 

# new_matrix
lib.new_matrix.argtypes = (c_uint, c_uint)
lib.new_matrix.restype = c_longlong

# new_matrix_data
lib.new_matrix_from_data.argtypes = (c_uint, c_uint, POINTER(c_float))
lib.new_matrix_from_data.restype = c_longlong

# sync
lib.sync.argtypes = (c_longlong,)

# print
lib.print.argtypes = (c_longlong,)

# load the library 
class Matrix:
    # This is a pointer to a matrix class in C/C++/CUDA
    matrix: c_longlong

    def __init__(self, _matrix):
        self.matrix = _matrix
        pass
    
    # creates an identity matrix
    @classmethod
    def identity(cls):
        matrix = lib.new_matrix(4, 4)
        return Matrix(matrix)

    # data is a np array of floats
    @classmethod
    def from_data(cls, data):
        mat_data = data.ctypes.data_as(POINTER(c_float)).contents
        M = data.shape[0]
        N = data.shape[1]
        matrix = lib.new_matrix_from_data(M, N, mat_data)
        return Matrix(matrix)

    # Prints the current matrix on the CPU
    def print(self):
        lib.print(self.matrix)

    # Syncs the matrix from the GPU to the CPU
    # THIS SHOULD BE DONE BEFORE DOING ANY CPU OPERATIONS like PRINT
    def sync(self):
        lib.sync(self.matrix)

    # Overloaded Matrix Multiplication
    def __mul__(self, other):
        print("Not implemented yet")

mat1 = Matrix.identity()
mat1.print()

data = np.array([[2,1],
                 [1,1]], dtype=np.float32)

mat2 = Matrix.from_data(data)
mat2.print()


