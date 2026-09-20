#!/usr/bin/env bash

# 单纯 cpp 文件可以仅使用 c++ 进行编译，但是目前的 module 涉及到 thrust 的使用，因此需要使用 nvcc 进行编译

#c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) ${1} -o example$(python3-config --extension-suffix)
nvcc -O3 -shared -std=c++11 -Xcompiler -fPIC $(python3 -m pybind11 --includes) ${1} -o example$(python3-config --extension-suffix)