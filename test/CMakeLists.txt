cmake_minimum_required(VERSION 3.5)

project(NeuralNetworkTest)
file(GLOB SOURCES "../src/*.cpp" test.cpp)
add_executable(NeuralNetworkTest ${SOURCES})
include_directories(../include)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "test_neuralnet")