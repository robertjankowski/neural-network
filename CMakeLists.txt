cmake_minimum_required(VERSION 2.8.9)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

project(NeuralNetwork)

include_directories("${PROJECT_SOURCE_DIR}/include")
file(GLOB SOURCES "src/*.cpp" "*.cpp")
add_executable(NeuralNetwork ${SOURCES})

set_property(TARGET NeuralNetwork PROPERTY CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS -no-pie)

target_compile_options(NeuralNetwork PRIVATE -Wall)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "neuralnet")