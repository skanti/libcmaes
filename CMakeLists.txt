cmake_minimum_required(VERSION 3.0)
project(libcmaes CXX)

set(CMAKE_VERBOSE_MAKEFILE 0)
if (LINUX)
    set(CMAKE_CXX_COMPILER "icpc")
elseif (APPLE)
    set(CMAKE_CXX_COMPILER "/usr/local/bin/icpc")
endif ()

set(OPTIMIZATION_FLAGS "")
set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS} ${OPTIMIZATION_FLAGS} -std=c++11")

set(CMAKE_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

set(SOURCE_DIR ./src)
set(SOURCE_FILES_CPP
        ${SOURCE_DIR}/cpp/CMAES.cpp
        ${SOURCE_DIR}/cpp/MathKernels.cpp
        ${SOURCE_DIR}/cpp/Parameters.cpp
        )

set(SOURCE_FILES_H
        ${SOURCE_DIR}/cpp/CMAES.h
        ${SOURCE_DIR}/cpp/Parameters.h
        ${SOURCE_DIR}/cpp/Data.h
        ${SOURCE_DIR}/cpp/Model.h
        ${SOURCE_DIR}/cpp/Types.h
        ${SOURCE_DIR}/cpp/AlignedAllocator.h
        ${SOURCE_DIR}/cpp/GnuplotIostream.h
        src/cpp/MathKernels.h)


if (LINUX)
elseif (APPLE)
    set(MKL_DIR /usr/local/intel/compilers_and_libraries/mac/mkl)
endif ()

include_directories(${SOURCE_DIR}/cpp ${MKL_DIR}/include)
link_directories(${MKL_DIR}/lib)
add_library(cmaes ${SOURCE_FILES_CPP} ${SOURCE_FILES_H})
target_link_libraries(cmaes mkl_intel_lp64 mkl_core mkl_sequential pthread m dl)

add_executable(libcmaes_example ./example/main.cpp)
target_link_libraries(libcmaes_example cmaes)

if (APPLE)
    add_custom_command(TARGET libcmaes_example POST_BUILD
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/send_remote.sh
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif ()