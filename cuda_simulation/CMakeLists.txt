cmake_minimum_required(VERSION 3.16)
project(cuda_simulation CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_GENERATOR_TOOLSET "cuda=8.0,${CMAKE_GENERATOR_TOOLSET}")

## PVS Studio
# include(PVS-Studio.cmake)
# SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif ()

find_package(Boost COMPONENTS filesystem program_options system REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
include_directories("includes")

file(GLOB_RECURSE SRC "include/*.h" "include/*.cuh" "src/*.cpp" "src/*.cu")
add_executable(cuda_simulation main.cu ${SRC})

set_target_properties(
        cuda_simulation
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda_simulation LINK_PUBLIC ${Boost_LIBRARIES})


add_executable(cuda_test cuda_test.cu
        includes/file_interface/parser_exception.h
        includes/file_interface/conf_parser.h src/file_interface/conf_parser.cpp)
set_target_properties(
        cuda_test
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda_test LINK_PUBLIC ${Boost_LIBRARIES})

#pvs_studio_add_target(TARGET integration_mt.analyze ALL
#        ANALYZE cuda_simulation
#        OUTPUT FORMAT errorfile
#        MODE GA:1,2,3 OP:1,2,3 GA:1,2,3 OP:1,2,3 64:1,2,3 CS:1,2,3 MISRA:1,2,3
#        LOG target_${PROJECT_NAME}.err)
#####################################
