cmake_minimum_required(VERSION 3.5)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

project(NeuralNetwork)
file(GLOB SOURCES "src/*.cpp" "main.cpp")
add_executable(NeuralNetwork ${SOURCES})

include_directories(include)

set_property(TARGET NeuralNetwork PROPERTY CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-no-pie -Wall -Wshadow -Wextra -pedantic")

target_compile_options(NeuralNetwork PRIVATE -Wall)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "neuralnet")