cmake_minimum_required(VERSION 3.28)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc")
#set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
#set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CUDA_FLAGS -allow-unsupported-compiler)

project(cupybara 
	VERSION 1.0.2
	LANGUAGES CXX C CUDA
)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS ./src/*.c*)

message("We are building using ${srcs}")

# Testing Executable

add_executable(cupybaraTest ${srcs})

set_property(TARGET cupybaraTest PROPERTY CXX_STANDARD 20)

target_include_directories(cupybaraTest PRIVATE utils/)

target_link_libraries(cupybaraTest PRIVATE cublas)

target_compile_options(cupybaraTest PUBLIC -g3 -mavx -march=znver2)

# Shared Library

add_library(cupybara SHARED ${srcs})

set_property(TARGET cupybara PROPERTY CXX_STANDARD 20)

target_include_directories(cupybara PRIVATE utils/)

target_link_libraries(cupybara PRIVATE cublas)

target_compile_options(cupybara PUBLIC -O3 -mavx -march=znver2)

include(cmake/installation.cmake)

include(cmake/packing.cmake)
