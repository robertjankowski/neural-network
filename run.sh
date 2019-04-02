#!/bin/bash
rm -r CMakeFiles # clean cmake
cmake .
make
bin/neuralnet $(pwd)"/iris.data" 100 2 0.1 # default parameters 