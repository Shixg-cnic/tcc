cmake_minimum_required(VERSION 3.18)

project(TccSpmm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80 90)

add_executable(swtcf_convert_run
    src/swtcf_convert_main.cu
    src/swtcf_convert.cu
)

target_include_directories(swtcf_convert_run PRIVATE include)

add_executable(swtcf_spmm_run
    src/swtcf_spmm_main.cu
    src/swtcf_spmm.cu
    src/swtcf_convert.cu
)

target_include_directories(swtcf_spmm_run PRIVATE include)