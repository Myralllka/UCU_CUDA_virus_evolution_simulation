cmake_minimum_required(VERSION 3.16)
project(cuda_integral CUDA CXX)

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
include_directories("include")

add_executable(cuda_integral main.cu
        include/cuda_impl/cuda_integrate.cuh src/cuda_impl/cuda_integrate.cu
        include/cuda_impl/cuda_assert.cuh
        )
set_target_properties(
        cuda_integral
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda_integral LINK_PUBLIC ${Boost_LIBRARIES})


#file(GLOB_RECURSE SRC "include/*.h" "include/*.cuh" "src/*.cpp" "src/*.cu")
#add_executable(cuda_test cuda_resource_test.cu
#        include/option_parser/ConfigFileOpt.h src/option_parser/ConfigFileOpt.cpp)
#set_target_properties(
#        cuda_test
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(cuda_test LINK_PUBLIC ${Boost_LIBRARIES})

#pvs_studio_add_target(TARGET integration_mt.analyze ALL
#        ANALYZE integration_mt
#        OUTPUT FORMAT errorfile
#        MODE GA:1,2,3 OP:1,2,3 GA:1,2,3 OP:1,2,3 64:1,2,3 CS:1,2,3 MISRA:1,2,3
#        LOG target_${PROJECT_NAME}.err)
#####################################
