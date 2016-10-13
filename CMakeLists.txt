cmake_minimum_required(VERSION 3.0)
project(cmaes CXX)

set(CMAKE_CXX_COMPILER "/usr/local/bin/icpc")

set(OPTIMIZATION_FLAGS "-O3 -march=native -m64")
set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS} ${OPTIMIZATION_FLAGS} -std=c++11 -DNDEBUG")

set(CMAKE_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

set(SOURCE_DIR ./src)

set(SOURCE_FILES_CPP
        ${SOURCE_DIR}/cpp/Engine.cpp
        ${SOURCE_DIR}/cpp/MathKernels.cpp
        ${SOURCE_DIR}/cpp/Parameter.cpp
        )

set(SOURCE_FILES_H
        ${SOURCE_DIR}/cpp/Engine.h
        ${SOURCE_DIR}/cpp/Parameter.h
        ${SOURCE_DIR}/cpp/CMAESTypes.h
        ${SOURCE_DIR}/cpp/AlignedAllocator.h
        src/cpp/MathKernels.h)


set(MKL_DIR /usr/local/intel/compilers_and_libraries/mac/mkl)

include_directories(${SOURCE_DIR}/cpp ${MKL_DIR}/include)
link_directories(${MKL_DIR}/lib)
add_library(cmaes ${SOURCE_FILES_CPP} ${SOURCE_FILES_H})
target_link_libraries(cmaes mkl_intel_lp64 mkl_core mkl_intel_thread iomp5 pthread m dl)

add_executable(cmaes_example ./example/main.cpp)
target_link_libraries(cmaes_example cmaes)
