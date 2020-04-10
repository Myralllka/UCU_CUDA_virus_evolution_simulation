cmake_minimum_required(VERSION 3.15)
project(CUDA_myralllka)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)

# PVS Studio
#include(PVS-Studio.cmake)
#SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif ()


file(GLOB_RECURSE SRC "includes/*.h" "src/*.cpp")
add_executable(CUDA_myralllka first.cu ${SRC})

#####################################
## create own target for library
#pvs_studio_add_target(TARGET ${PROJECT_NAME}.analyze ALL
#        ANALYZE count_number_of_all_words
#        OUTPUT FORMAT errorfile
#        MODE GA:1,2,3 OP:1,2,3 GA:1,2,3 OP:1,2,3 64:1,2,3 CS:1,2,3 MISRA:1,2,3
#        LOG target_${PROJECT_NAME}.err)

#####################################