cmake_minimum_required(VERSION 2.8.9)

project(NeuralNetwork)

include_directories("${PROJECT_SOURCE_DIR}/include")
file(GLOB SOURCES "src/*.cpp" "*.cpp")
add_executable(NeuralNetwork ${SOURCES})

set_property(TARGET NeuralNetwork PROPERTY CXX_STANDARD 11)

target_compile_options(NeuralNetwork PRIVATE -Wall)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "neuralnet")