cmake_minimum_required(VERSION 3.18)

project(TccSpmm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 90)


add_executable(tcf_convert
    src/main.cpp
    src/format.cpp
)

target_include_directories(tcf_convert PRIVATE include)

add_executable(mma_test
    src/mma_test.cu
)

target_include_directories(mma_test PRIVATE include)

add_executable(tcf_spmm_run
    src/tcf_spmm_main.cu
    src/tcf_spmm.cu
    src/format.cpp
)

add_executable(tcf_spmm_cc_run
    src/tcf_spmm_cc_main.cu
    src/tcf_spmm_cc.cu
    src/format.cpp
)

target_include_directories(
    tcf_spmm_cc_run
    PRIVATE include
)

target_include_directories(tcf_spmm_run PRIVATE include)
add_executable(tcf_spmm_hybrid_run
    src/tcf_spmm_hybrid_main.cu
    src/tcf_spmm_hybrid.cu
    src/format.cpp
)

target_include_directories(
    tcf_spmm_hybrid_run
    PRIVATE include
)

add_executable(swtcf_convert_run
    src/swtcf_convert_main.cu
    src/swtcf_convert.cu
)

target_include_directories(
    swtcf_convert_run
    PRIVATE include
)