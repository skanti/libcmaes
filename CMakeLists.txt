cmake_minimum_required(VERSION 3.0)
project(cmaes CXX)

set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++14 -fopenmp -DNDEBUG")

set(CMAKE_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

set(SOURCE_DIR ./src)

set(SOURCE_FILES_CPP
        ${SOURCE_DIR}/Engine.cpp
        ${SOURCE_DIR}/MathKernels.cpp
        ${SOURCE_DIR}/Parameter.cpp
        )

set(SOURCE_FILES_H
        ${SOURCE_DIR}/Engine.h
        ${SOURCE_DIR}/Parameter.h
        ${SOURCE_DIR}/CMAESTypes.h
        ${SOURCE_DIR}/AlignedAllocator.h
        ${SOURCE_DIR}/MathKernels.h)


set(MKL_DIR /opt/intel/compilers_and_libraries/linux/mkl)


include_directories(${SOURCE_DIR} ${MKL_DIR}/include)
link_directories(${MKL_DIR}/lib/intel64_lin/ ${MKL_DIR}/lib/intel64)
add_library(cmaes ${SOURCE_FILES_CPP} ${SOURCE_FILES_H})
target_link_libraries(cmaes mkl_intel_lp64 mkl_core mkl_intel_thread pthread m dl)

add_executable(cmaes_example ./example/main.cpp)
target_link_libraries(cmaes_example cmaes)
