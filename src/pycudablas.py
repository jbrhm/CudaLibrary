from ctypes import cdll 
  
# load the library 
print("Loading Library...")
lib = cdll.LoadLibrary('./build/libcudablas.so') 

print("Creating Matrix...")
matrix = lib.new_matrix(4, 4)

print("Syncing Matrix...")
lib.sync(matrix)

print("Printing Matrix...")
lib.print(matrix)

#https://docs.python.org/3/library/ctypes.html#variable-sized-data-types
#https://www.geeksforgeeks.org/using-pointers-in-python-using-ctypes/
#https://www.geeksforgeeks.org/how-to-call-c-c-from-python/
