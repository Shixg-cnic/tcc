#include <torch/extension.h>
#include <vector>
#include <memory>
#include <iostream>
#include "class.h" 
#include "utils.h"
#include "spmm_launcher.h"

#include <nvToolsExt.h>

namespace py = pybind11;

std::vector<void*> MatCs_;

cudaStream_t stream_bcsr;
cudaStream_t stream_bcsc;

//#define use_mixed_memory_access
#define use_global_memory_access

#ifdef use_mixed_memory_access
torch::Tensor forward_tensorcore_mixed(
    vint bcsr_num_rows, vint bcsr_num_cols, vint bcsr_nnz, vint bcsr_num_TCBlocks, vint bcsr_num_rowWindows,
    torch::Tensor bcsr_rowWindowOffset, torch::Tensor bcsr_tcOffset, torch::Tensor bcsr_sparseA2B, torch::Tensor bcsr_tcLocalBit, 
    vint bcsc_num_rows, vint bcsc_num_cols, vint bcsc_nnz, vint bcsc_num_TCBlocks, vint bcsc_num_colWindows,
    torch::Tensor bcsc_colWindowOffset, torch::Tensor bcsc_tcOffset, torch::Tensor bcsc_sparseA2C, torch::Tensor bcsc_tcLocalBit, 
    int64_t cpu_float_feature_len, int32_t gpu_node_capacity, int32_t feature_dim,
    torch::Tensor id_map, py::capsule cpu_feature_cache_ptr, torch::Tensor cache_search_map, py::capsule gpu_feature_cache_ptr,
    const int32_t device_id, py::capsule denseC) {

    //std::printf("\n[Info] first layer forward\n"); 

    // bcsr 构造
    vint *raw_bcsr_rowWindowOffset = bcsr_rowWindowOffset.data_ptr<vint>();
    vint *raw_bcsr_tcOffset = bcsr_tcOffset.data_ptr<vint>();
    vint *raw_bcsr_sparseA2B = bcsr_sparseA2B.data_ptr<vint>();
    TCLOCAL_TYPE *raw_bcsr_tcLocalBit = bcsr_tcLocalBit.data_ptr<TCLOCAL_TYPE>();

    BCSR<MAT_VAL_TYPE> bcsr(
        bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows, 
        raw_bcsr_rowWindowOffset,
        raw_bcsr_tcOffset,
        raw_bcsr_sparseA2B,
        raw_bcsr_tcLocalBit
    );

    //std::printf("[Info] training-bcsr: \n");
    //std::cout << bcsr.toString() << std::endl;

    // bcsc 构造
    vint *raw_bcsc_colWindowOffset = bcsc_colWindowOffset.data_ptr<vint>();
    vint *raw_bcsc_tcOffset = bcsc_tcOffset.data_ptr<vint>();
    vint *raw_bcsc_sparseA2C = bcsc_sparseA2C.data_ptr<vint>();
    TCLOCAL_TYPE *raw_bcsc_tcLocalBit = bcsc_tcLocalBit.data_ptr<TCLOCAL_TYPE>();

    BCSC<MAT_VAL_TYPE> bcsc(
        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows, 
        raw_bcsc_colWindowOffset,
        raw_bcsc_tcOffset,
        raw_bcsc_sparseA2C,
        raw_bcsc_tcLocalBit
    );

    //std::printf("[Info] training-bcsc: \n"); 
    //std::cout << bcsc.toString() << std::endl;

    // spmm
    int32_t* raw_id_map = id_map.data_ptr<int32_t>();
    int32_t* raw_cache_search_map = cache_search_map.data_ptr<int32_t>();
    float* raw_cpu_feature_cache_ptr = static_cast<float*>(cpu_feature_cache_ptr.get_pointer()); 
    float** raw_gpu_feature_cache_ptr = static_cast<float**>(gpu_feature_cache_ptr.get_pointer()); 

    //MAT_VAL_TYPE *d_MatC;
    MAT_VAL_TYPE *d_MatC = static_cast<MAT_VAL_TYPE*>(denseC.get_pointer());
    launch_spmm_kernel(bcsr, bcsc, raw_id_map, raw_cpu_feature_cache_ptr, raw_cache_search_map, raw_gpu_feature_cache_ptr, d_MatC, gpu_node_capacity, feature_dim, device_id, stream_bcsr, stream_bcsc); // 目前 denseB 数据还可以模拟生成，后续需要加上 cpu_float_feature 和 gpu_float_feature 数据提取真实 feature 数据
    //cudaDeviceSynchronize();
    cudaStreamSynchronize(stream_bcsr);
    cudaStreamSynchronize(stream_bcsc);

    // denseMatC 计算结果转换为 tensor 数据返回。
    int current_dev = -1;
    cudaGetDevice(&current_dev);
    auto device = "cuda:" + std::to_string(current_dev);

    //std::printf("device-%d\n", current_dev);

    auto output = torch::from_blob(
        d_MatC, 
        {(long long)(bcsr_num_rows) * feature_dim}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );

    //MatCs_.push_back(d_MatC);

    return output;
}
#endif

#ifdef use_global_memory_access
torch::Tensor forward_tensorcore_mixed(
    vint bcsr_num_rows, vint bcsr_num_cols, vint bcsr_nnz, vint bcsr_num_TCBlocks, vint bcsr_num_rowWindows,
    torch::Tensor bcsr_rowWindowOffset, torch::Tensor bcsr_tcOffset, torch::Tensor bcsr_sparseA2B, torch::Tensor bcsr_tcLocalBit, 
    vint bcsc_num_rows, vint bcsc_num_cols, vint bcsc_nnz, vint bcsc_num_TCBlocks, vint bcsc_num_colWindows,
    torch::Tensor bcsc_colWindowOffset, torch::Tensor bcsc_tcOffset, torch::Tensor bcsc_sparseA2C, torch::Tensor bcsc_tcLocalBit, 
    int64_t cpu_float_feature_len, int32_t gpu_node_capacity, int32_t feature_dim,
    torch::Tensor id_map, py::capsule cpu_feature_cache_ptr, torch::Tensor cache_search_map, py::capsule gpu_feature_cache_ptr,
    torch::Tensor x,   
    const int32_t device_id, py::capsule denseC) {

    //std::printf("\n[Info] first layer forward\n"); 

    // bcsr 构造
    vint *raw_bcsr_rowWindowOffset = bcsr_rowWindowOffset.data_ptr<vint>();
    vint *raw_bcsr_tcOffset = bcsr_tcOffset.data_ptr<vint>();
    vint *raw_bcsr_sparseA2B = bcsr_sparseA2B.data_ptr<vint>();
    TCLOCAL_TYPE *raw_bcsr_tcLocalBit = bcsr_tcLocalBit.data_ptr<TCLOCAL_TYPE>();

    BCSR<MAT_VAL_TYPE> bcsr(
        bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows, 
        raw_bcsr_rowWindowOffset,
        raw_bcsr_tcOffset,
        raw_bcsr_sparseA2B,
        raw_bcsr_tcLocalBit
    );

    //std::printf("[Info] training-bcsr: \n");
    //std::cout << bcsr.toString() << std::endl;

    // bcsc 构造
    vint *raw_bcsc_colWindowOffset = bcsc_colWindowOffset.data_ptr<vint>();
    vint *raw_bcsc_tcOffset = bcsc_tcOffset.data_ptr<vint>();
    vint *raw_bcsc_sparseA2C = bcsc_sparseA2C.data_ptr<vint>();
    TCLOCAL_TYPE *raw_bcsc_tcLocalBit = bcsc_tcLocalBit.data_ptr<TCLOCAL_TYPE>();

    BCSC<MAT_VAL_TYPE> bcsc(
        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows, 
        raw_bcsc_colWindowOffset,
        raw_bcsc_tcOffset,
        raw_bcsc_sparseA2C,
        raw_bcsc_tcLocalBit
    );

    //std::printf("[Info] training-bcsc: \n"); 
    //std::cout << bcsc.toString() << std::endl;

    // spmm
    int32_t* raw_id_map = id_map.data_ptr<int32_t>();
    int32_t* raw_cache_search_map = cache_search_map.data_ptr<int32_t>();
    float* raw_cpu_feature_cache_ptr = static_cast<float*>(cpu_feature_cache_ptr.get_pointer()); 
    float** raw_gpu_feature_cache_ptr = static_cast<float**>(gpu_feature_cache_ptr.get_pointer()); 

    // matB & matC 准备
    MAT_VAL_TYPE *raw_MatB = x.data_ptr<MAT_VAL_TYPE>();
    //MAT_VAL_TYPE *d_MatC;
    MAT_VAL_TYPE *d_MatC = static_cast<MAT_VAL_TYPE*>(denseC.get_pointer());
    launch_spmm_kernel(bcsr, bcsc, raw_MatB, raw_id_map, raw_cpu_feature_cache_ptr, raw_cache_search_map, raw_gpu_feature_cache_ptr, d_MatC, gpu_node_capacity, feature_dim, device_id, stream_bcsr, stream_bcsc); // 目前 denseB 数据还可以模拟生成，后续需要加上 cpu_float_feature 和 gpu_float_feature 数据提取真实 feature 数据

    std::string nvtx_msg = "device sync";
    nvtxRangePush(nvtx_msg.c_str());
    //cudaDeviceSynchronize();
    nvtxRangePop();

    nvtx_msg = "bcsr stream sync";
    nvtxRangePush(nvtx_msg.c_str());
    //cudaStreamSynchronize(stream_bcsr);
    nvtxRangePop();

    nvtx_msg = "bcsc stream sync";
    nvtxRangePush(nvtx_msg.c_str());
    //cudaStreamSynchronize(stream_bcsc);
    nvtxRangePop();

    // denseMatC 计算结果转换为 tensor 数据返回。
    int current_dev = -1;
    cudaGetDevice(&current_dev);
    auto device = "cuda:" + std::to_string(current_dev);

    //std::printf("device-%d\n", current_dev);

    auto output = torch::from_blob(
        d_MatC, 
        {(long long)(bcsr_num_rows) * feature_dim}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );

    //MatCs_.push_back(d_MatC);

    return output;
}
#endif

torch::Tensor forward_tensorcore(
    vint bcsr_num_rows, vint bcsr_num_cols, vint bcsr_nnz, vint bcsr_num_TCBlocks, vint bcsr_num_rowWindows,
    torch::Tensor bcsr_rowWindowOffset, torch::Tensor bcsr_tcOffset, torch::Tensor bcsr_sparseA2B, torch::Tensor bcsr_tcLocalBit, 
    vint bcsc_num_rows, vint bcsc_num_cols, vint bcsc_nnz, vint bcsc_num_TCBlocks, vint bcsc_num_colWindows,
    torch::Tensor bcsc_colWindowOffset, torch::Tensor bcsc_tcOffset, torch::Tensor bcsc_sparseA2C, torch::Tensor bcsc_tcLocalBit, 
    int32_t feature_dim, torch::Tensor x, const int32_t device_id, py::capsule denseC) {

    //std::printf("\n[Info] second layer forward\n"); 

    // bcsr 构造
    vint *raw_bcsr_rowWindowOffset = bcsr_rowWindowOffset.data_ptr<vint>();
    vint *raw_bcsr_tcOffset = bcsr_tcOffset.data_ptr<vint>();
    vint *raw_bcsr_sparseA2B = bcsr_sparseA2B.data_ptr<vint>();
    TCLOCAL_TYPE *raw_bcsr_tcLocalBit = bcsr_tcLocalBit.data_ptr<TCLOCAL_TYPE>();

    BCSR<MAT_VAL_TYPE> bcsr(
        bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows, 
        raw_bcsr_rowWindowOffset,
        raw_bcsr_tcOffset,
        raw_bcsr_sparseA2B,
        raw_bcsr_tcLocalBit
    );

    //std::printf("[Info] training-bcsr: \n");
    //std::cout << bcsr.toString() << std::endl;

    // bcsc 构造
    vint *raw_bcsc_colWindowOffset = bcsc_colWindowOffset.data_ptr<vint>();
    vint *raw_bcsc_tcOffset = bcsc_tcOffset.data_ptr<vint>();
    vint *raw_bcsc_sparseA2C = bcsc_sparseA2C.data_ptr<vint>();
    TCLOCAL_TYPE *raw_bcsc_tcLocalBit = bcsc_tcLocalBit.data_ptr<TCLOCAL_TYPE>();

    BCSC<MAT_VAL_TYPE> bcsc(
        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows, 
        raw_bcsc_colWindowOffset,
        raw_bcsc_tcOffset,
        raw_bcsc_sparseA2C,
        raw_bcsc_tcLocalBit
    );

    //std::printf("[Info] training-bcsc: \n"); 
    //std::cout << bcsc.toString() << std::endl;

    // matB & matC 准备
    MAT_VAL_TYPE *raw_MatB = x.data_ptr<MAT_VAL_TYPE>();
    //MAT_VAL_TYPE *d_MatC;
    MAT_VAL_TYPE *d_MatC = static_cast<MAT_VAL_TYPE*>(denseC.get_pointer());

    //thrust::device_vector<MAT_VAL_TYPE> d_MatB(raw_MatB, raw_MatB + bcsc_num_cols * feature_dim);
    //thrust::host_vector<MAT_VAL_TYPE> h_MatB = d_MatB;
    //std::printf("[Info] second layer feature data: { ");
    //for (int32_t i = 0; i < h_MatB.size(); i++) {
    //    if (i && i % 256 == 0) {
    //        std::cout << std::endl;
    //    }
    //    std::cout << h_MatB[i] << ", ";
    //}
    //std::cout << std::endl;

    // spmm
    launch_spmm_kernel(bcsr, bcsc, raw_MatB, d_MatC, feature_dim, device_id, stream_bcsr, stream_bcsc); // 目前 denseB 数据还可以模拟生成，后续需要加上 cpu_float_feature 和 gpu_float_feature 数据提取真实 feature 数据
    //cudaDeviceSynchronize();
    //cudaStreamSynchronize(stream_bcsr);
    //cudaStreamSynchronize(stream_bcsc);

    // denseMatC 计算结果转换为 tensor 数据返回。
    int current_dev = -1;
    cudaGetDevice(&current_dev);
    auto device = "cuda:" + std::to_string(current_dev);

    //std::printf("device-%d\n", current_dev);

    auto output = torch::from_blob(
        d_MatC, 
        {(long long)(bcsr_num_rows) * feature_dim}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );

    //MatCs_.push_back(d_MatC);

    return output;
}

void read_cpu_feature_cache(py::capsule py_cpu_feature_cache_ptr, int32_t cpu_feature_len) {
    float* cpu_feature_cache_ptr = static_cast<float*>(py_cpu_feature_cache_ptr.get_pointer());

    thrust::host_vector<float> h_cpu_feature_cache(cpu_feature_cache_ptr, cpu_feature_cache_ptr + cpu_feature_len);

    std::printf("h_cpu_feature_cache: { ");
    for (int i = 0; i < cpu_feature_len; i++) {
        std::printf("%f, ", h_cpu_feature_cache[i]);
    }
    std::printf("}\n");
}

void read_gpu0_feature_cache(py::capsule py_gpu_feature_cache_ptr) {
    float* gpu_feature_cache_ptr = static_cast<float*>(py_gpu_feature_cache_ptr.get_pointer());
    thrust::device_vector<float> d_gpu_feature_cache(gpu_feature_cache_ptr, gpu_feature_cache_ptr + 100);
    thrust::host_vector<float> h_gpu_feature_cache = d_gpu_feature_cache;

    std::printf("gpu feature: { ");
    for (int i = 0; i < h_gpu_feature_cache.size(); i++) {
        std::printf("%f, ", h_gpu_feature_cache[i]);
    }
    std::printf("}\n");
}

void initialize(int device_id) {
    cudaSetDevice(device_id);

    cudaStreamCreate(&stream_bcsr);
    cudaStreamCreate(&stream_bcsc);
}

std::vector<py::capsule> space_malloc(int device_id, vint denseC_size) {
    std::vector<py::capsule> ret;

    cudaSetDevice(device_id);

    // 目前对第二层结果进行写入时似乎同时存在着对第一层计算结果的读取，所以保险起见还是把两个空间分开
    MAT_VAL_TYPE *d_MatC, *d_MatC_2;

    cudaMalloc(&d_MatC, sizeof(MAT_VAL_TYPE) * denseC_size);
    cudaMemset(d_MatC, 0, denseC_size * sizeof(MAT_VAL_TYPE));

    cudaMalloc(&d_MatC_2, sizeof(MAT_VAL_TYPE) * denseC_size);
    cudaMemset(d_MatC_2, 0, denseC_size * sizeof(MAT_VAL_TYPE));

    ret.push_back(py::capsule(static_cast<void*>(d_MatC), "denseC"));
    ret.push_back(py::capsule(static_cast<void*>(d_MatC_2), "denseC_2"));

    MatCs_.push_back(d_MatC);
    MatCs_.push_back(d_MatC_2);

    return ret;
}

void finalize(int device_id) {
    cudaSetDevice(device_id);

    // 清除 stream
    cudaStreamDestroy(stream_bcsr);
    cudaStreamDestroy(stream_bcsc);

    // 释放空间
    for (int i = 0; i < MatCs_.size(); i++) {
        cudaFree(MatCs_[i]);
    }
    MatCs_.clear();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_tensorcore_mixed", &forward_tensorcore_mixed,
        "mixed memory access SpMM(first layer of gcn)");

    m.def("forward_tensorcore", &forward_tensorcore,
        "normal memory access SpMM");

    m.def("read_cpu_feature_cache", &read_cpu_feature_cache,
        "read_cpu_feature_cache");

    m.def("read_gpu0_feature_cache", &read_gpu0_feature_cache,
        "read_gpu0_feature_cache");

    m.def("initialize", &initialize,
        "initialize");

    m.def("space_malloc", &space_malloc, 
        "malloc space for train");

    m.def("finalize", &finalize,
        "finalize");
}