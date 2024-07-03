cmake_minimum_required(VERSION 3.27)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-18")
set(CMAKE_C_COMPILER "/usr/bin/clang-18")
set(CMAKE_CUDA_FLAGS -allow-unsupported-compiler)

project(matrix 
	VERSION 1.0.0
	LANGUAGES CXX C CUDA
)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS ./src/*.c*)

message("We are building using ${srcs}")

add_executable(mat ${srcs})

target_include_directories(mat PUBLIC .)

set_target_properties(mat PROPERTIES LINKER_LANGUAGE CXX)
