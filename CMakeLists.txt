cmake_minimum_required(VERSION 3.0)
project(cmaes CXX)

set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++14 -fopenmp -DNDEBUG")

include_directories("/usr/local/include/libcmaes/")

find_package(Eigen3 3.3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(CMAKE_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")


set(SOURCE_DIR ./src)

set(SOURCE_FILES_CPP
        ${SOURCE_DIR}/Engine.cpp
        ${SOURCE_DIR}/Parameter.cpp
        )

set(SOURCE_FILES_H
        ${SOURCE_DIR}/Engine.h
        ${SOURCE_DIR}/Parameter.h
        ${SOURCE_DIR}/Common.h)


include_directories(./src)

add_library(cmaes ${SOURCE_FILES_CPP} ${SOURCE_FILES_H})
target_link_libraries(cmaes)

add_executable(cmaes_example ./example/point_registration.cpp)
target_link_libraries(cmaes_example cmaes)
