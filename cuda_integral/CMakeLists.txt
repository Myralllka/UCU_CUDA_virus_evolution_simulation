cmake_minimum_required(VERSION 3.16)
project(cuda_integral CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_integral main.cu)

set_target_properties(
        cuda_integral
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
