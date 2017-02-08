cmake_minimum_required(VERSION 3.0)
project(cmaes CXX)
 
 #-DNDEBUG

set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++14 -fopenmp")

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
        ${SOURCE_DIR}/Common.h
        ${SOURCE_DIR}/MathKernels.h)


include_directories(./src)

#add_library(cmaes ${SOURCE_FILES_CPP} ${SOURCE_FILES_H})
#target_link_libraries(cmaes)

add_executable(cmaes_example ./example/main.cpp ${SOURCE_FILES_H})
#target_link_libraries(cmaes_example cmaes)
