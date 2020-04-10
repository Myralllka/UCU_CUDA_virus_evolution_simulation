cmake_minimum_required(VERSION 3.16)
project(cuda_sandbox CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_sandbox main.cu
        src/vec_add.cu include/vec_add.cuh
        )

set_target_properties(
        cuda_sandbox
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)