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
find_package(Threads REQUIRED)

file(GLOB_RECURSE SRC "include/*.h" "src/*.cpp")

add_executable(cuda_integral main.cu
        include/cuda/cuda_integrate.cuh
        include/cuda/cuda_assert.cuh
        ${SRC})
set_target_properties(
        cuda_integral
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda_integral LINK_PUBLIC ${Boost_LIBRARIES})

#add_executable(integration_mt main.cpp ${SRC})
#target_link_libraries(integration_mt LINK_PUBLIC ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries(integration_mt LINK_PUBLIC ${Boost_LIBRARIES})

add_executable(cuda_resource_test cuda_resource_test.cu
        include/option_parser/ConfigFileOpt.h src/option_parser/ConfigFileOpt.cpp)
set_target_properties(
        cuda_resource_test
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda_resource_test LINK_PUBLIC ${Boost_LIBRARIES})

#pvs_studio_add_target(TARGET integration_mt.analyze ALL
#        ANALYZE integration_mt
#        OUTPUT FORMAT errorfile
#        MODE GA:1,2,3 OP:1,2,3 GA:1,2,3 OP:1,2,3 64:1,2,3 CS:1,2,3 MISRA:1,2,3
#        LOG target_${PROJECT_NAME}.err)
#####################################
