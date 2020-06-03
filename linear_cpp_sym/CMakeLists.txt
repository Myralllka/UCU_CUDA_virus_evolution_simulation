cmake_minimum_required(VERSION 3.16)
project(linear_cpp_sym)

set(CMAKE_CXX_STANDARD 17)

# PVS Studio
#include(PVS-Studio.cmake)
#SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif ()
set(CMAKE_BUILD_TYPE  Release)
#find_package(Threads REQUIRED)
#set(Boost_USE_STATIC_LIBS OFF) # enable dynamic linking
#set(Boost_USE_MULTITHREAD ON)  # enable maltreating
find_package(Boost COMPONENTS filesystem program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES("includes")

file(GLOB_RECURSE SRC "includes/*.h" "src/*.cpp")

add_executable(linear_cpp_sym main.cpp ${SRC})
TARGET_LINK_LIBRARIES(linear_cpp_sym LINK_PUBLIC ${Boost_LIBRARIES})
#TARGET_LINK_LIBRARIES(linear_cpp_sym LINK_PUBLIC ${CMAKE_THREAD_LIBS_INIT})

#add_executable(test test.cpp)
#TARGET_LINK_LIBRARIES(test LINK_PUBLIC ${CMAKE_THREAD_LIBS_INIT})
#TARGET_LINK_LIBRARIES(test LINK_PUBLIC ${Boost_LIBRARIES})

#####################################
## create own target for library
#pvs_studio_add_target(TARGET ${PROJECT_NAME}.analyze ALL
#        ANALYZE linear_cpp_sym
#        OUTPUT FORMAT errorfile
#        MODE GA:1,2,3 OP:1,2,3 GA:1,2,3 OP:1,2,3 64:1,2,3 CS:1,2,3 MISRA:1,2,3
#        LOG target_${PROJECT_NAME}.err)
#####################################
