from ctypes import CDLL, c_longlong, c_uint, POINTER, c_float

import numpy as np

from cupybara.cupybara_paths import CupybaraPaths
  
lib = CDLL(CupybaraPaths.cupybara_libs) 

# new_matrix
lib.new_matrix.argtypes = (c_uint, c_uint)
lib.new_matrix.restype = c_longlong

# new_matrix_data
lib.new_matrix_from_data.argtypes = (c_uint, c_uint, POINTER(c_float))
lib.new_matrix_from_data.restype = c_longlong
# release Matrix data
lib.matrix_release.argtypes = (c_longlong,)

# sync
lib.matrix_sync.argtypes = (c_longlong,)

# print
lib.matrix_print.argtypes = (c_longlong,)

# Multiply
lib.matrix_multiply.argtypes = (c_longlong, c_longlong, c_longlong)
lib.matrix_multiply.restype = c_longlong

# load the library 
class Matrix:
    # This is a pointer to a matrix class in C/C++/CUDA
    matrix: c_longlong

    def __init__(self, _matrix):
        self.matrix = _matrix
        pass

    def __del__(self):
        lib.matrix_release(self.matrix)
    
    # creates an identity matrix
    @classmethod
    def identity(cls, M, N):
        matrix = lib.new_matrix(M, N)
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
        lib.matrix_print(self.matrix)

    # Syncs the matrix from the GPU to the CPU
    # THIS SHOULD BE DONE BEFORE DOING ANY CPU OPERATIONS like PRINT
    def sync(self):
        lib.matrix_sync(self.matrix)

    # Overloaded Matrix Multiplication
    @staticmethod
    def multiply(A, B, C):
        lib.matrix_multiply(A.matrix, B.matrix, C.matrix)

# new_vector
lib.new_vector.argtypes = (c_uint,)
lib.new_vector.restype = c_longlong

# new_matrix_from_data
lib.new_vector_from_data.argtypes = (c_uint, POINTER(c_float))
lib.new_vector_from_data.restype = c_longlong

# release vector data
lib.vector_release.argtypes = (c_longlong,)

# vector_syncHost
lib.vector_syncHost.argtypes = (c_longlong,)

# vector_syncAVX
lib.vector_syncAVX.argtypes = (c_longlong,)

# print
lib.vector_print.argtypes = (c_longlong,)

# add
lib.vector_add.argtypes = (c_longlong, c_longlong, c_longlong)
lib.vector_add.restype = c_longlong

# Parrallelized Vector
class Vector:
    # This is a pointer to a vector class in C/C++/CUDA
    vector: c_longlong

    def __init__(self, _vector):
        self.vector = _vector
        pass

    def __del__(self):
        lib.vector_release(self.vector)
    
    # creates a vector of all zeros
    @classmethod
    def zeros(cls, n):
        vector = lib.new_vector(n)
        return Vector(vector)

    # data is a np array of floats
    @classmethod
    def from_data(cls, data):
        vec_data = data.ctypes.data_as(POINTER(c_float)).contents
        n = data.shape[0]
        vector = lib.new_vector_from_data(n, vec_data)
        return Vector(vector)

    # Prints the current vector on the CPU
    def print(self):
        lib.vector_print(self.vector)

    # Syncs the vector from the GPU/AVX to the CPU
    # THIS SHOULD BE DONE BEFORE DOING ANY CPU OPERATIONS like PRINT
    def syncHost(self):
        lib.vector_syncHost(self.vector)

    # Syncs the vector to the AVX
    def syncAVX(self):
        lib.vector_syncAVX(self.vector)

    # Overloaded Matrix Multiplication
    @staticmethod
    def add(A, B, C):
        lib.vector_add(A.vector, B.vector, C.vector)
