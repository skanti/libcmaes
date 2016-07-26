cmake_minimum_required(VERSION 3.5)
project(libcmaes CXX)

set(CMAKE_VERBOSE_MAKEFILE 0)
set(CMAKE_CXX_COMPILER "/usr/local/bin/icpc")
#set(OPTIMIZATION_FLAGS "-O3 -qopenmp  -march=native -mtune=native")
set(OPTIMIZATION_FLAGS)
set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS} ${OPTIMIZATION_FLAGS} -std=c++14 ${DEFINES}")

set(CMAKE_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

set(SOURCE_DIR ./src)
set(SOURCE_FILES_CPP
        ${SOURCE_DIR}/cpp/CMAES.cpp
        ${SOURCE_DIR}/cpp/Parameters.cpp)

set(SOURCE_FILES_H
        ${SOURCE_DIR}/cpp/CMAES.h
        ${SOURCE_DIR}/cpp/Parameters.h
        ${SOURCE_DIR}/cpp/Data.h
        ${SOURCE_DIR}/cpp/Model.h
        ${SOURCE_DIR}/cpp/Types.h
        #${SOURCE_DIR}/cpp/GnuplotIostream.h
        src/cpp/SolverPool.h)


# dependencies
set(ARMADILLO_DIR /usr/local/Cellar/armadillo/7.200.2)
set(BOOST_DIR /usr/local/Cellar/boost/1.60.0_2)
#set(EIGEN_DIR /usr/local/Cellar/eigen/3.2.8)

include_directories(${SOURCE_DIR}/cpp ${ARMADILLO_DIR}/include)
link_directories(${ARMADILLO_DIR}/lib)
add_library(cmaes ${SOURCE_FILES_CPP} ${SOURCE_FILES_H})
target_link_libraries(cmaes armadillo)

# test
#link_directories(${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
add_executable(libcmaes_example ./example/main.cpp)
target_link_libraries(libcmaes_example cmaes)