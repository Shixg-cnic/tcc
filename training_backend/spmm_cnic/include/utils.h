#pragma once

#include <cuda_runtime.h>
#include <string>

// 锁函数
#ifdef __CUDA__
__device__ void acquireLock(int* lock);
__device__ void releaseLock(int* lock);
#else
inline void acquireLock(int* lock);
inline void releaseLock(int* lock);
#endif

// 错误检查
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, int line);

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line);

// 检查宏定义
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while(0)
#define CUDA_CHECK(x) do { cudaError_t err = x; if (err != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(err)); } } while(0)

#define CHECK_CUDA_ERROR(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while(0)


#define CHECK_LAST_CUDA_ERROR() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while(0)

// 打印CUDA设备信息
void printCudaInfo();

// 辅助函数
std::string match_filename(std::string s);
void init_vec(const int rows, const int cols, float** Mat);
void init_vec1(const int nnz, float* Mat, float val);
void init_vecB(const int rows, const int cols, float* Mat, float val);

// 定义GpuTimer结构
struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
    GpuTimer();
    ~GpuTimer();
    void Start();
    void Stop();
    float Elapsed();
};

// 定义Dur结构
typedef uint64_t clocktype;
struct Dur {
    clocktype begin;
    clocktype end;
    int smid;
    Dur(clocktype x, clocktype y, int outsm);
};

// 自定义类型
typedef struct {
    float f1, f2, f3, f4, f5, f6, f7, f8;
} float8;
