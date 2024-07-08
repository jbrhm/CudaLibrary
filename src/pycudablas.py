from ctypes import CDLL, c_void_p, c_longlong
  
# load the library 
print("Loading Library...")
lib = CDLL('./build/libcudablas.so') 

lib.new_matrix.restype = c_longlong
lib.sync.argtypes = (c_longlong,)
lib.print.argtypes = (c_longlong,)

print("Creating Matrix...")
matrix = lib.new_matrix(4, 4)

print("Python Matrix Value:")
print(matrix)

print("Syncing Matrix...")
lib.sync(matrix)

print("Python Matrix Value:")
print(matrix)

print("Printing Matrix...")
lib.print(matrix)

#https://docs.python.org/3/library/ctypes.html#variable-sized-data-types
#https://www.geeksforgeeks.org/using-pointers-in-python-using-ctypes/
#https://www.geeksforgeeks.org/how-to-call-c-c-from-python/
