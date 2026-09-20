#include "utils.h"
#include <cstdio>
#include <iostream>

// CUDA设备锁函数实现
#ifdef __CUDA__
__device__ void acquireLock(int* lock) {
    while (atomicCAS(lock, 0, 1) != 0) {
        // spin
    }
}

__device__ void releaseLock(int* lock) {
    atomicExch(lock, 0);
}
#else
inline void acquireLock(int* lock) {
    while (__sync_val_compare_and_swap(lock, 0, 1) != 0) {
        // spin
    }
}

inline void releaseLock(int* lock) {
    __sync_lock_release(lock);
}
#endif

// 错误检查实现
void check(cudaError_t err, const char* const func, const char* const file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s: %d\n", cudaGetErrorString(err), file, line);
        fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
    }
}

void checkLast(const char* const file, const int line) {
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s at %s: %d\n", cudaGetErrorString(err), file, line);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
    }
}

// 打印CUDA设备信息实现
void printCudaInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024.0 * 1024.0));
        printf("   WarpSize:   %d\n", deviceProps.warpSize);
        printf("   shMem/Blk:  %d\n", deviceProps.sharedMemPerBlock);
        printf("   L2Size:     %.0f MB\n", static_cast<float>(deviceProps.l2CacheSize) / (1024.0 * 1024.0));
    }
    printf("---------------------------------------------------------\n");
}

// 文件名匹配实现
std::string match_filename(std::string s) {
    int last_slash = s.rfind('/') + 1;
    std::string suffix = s.substr(last_slash, s.size() - last_slash);
    return suffix;
}

// 初始化向量实现
void init_vec(const int rows, const int cols, float** Mat) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Mat[i][j] = 1.0;
        }
    }
}

void init_vec1(const int nnz, float* Mat, float val) {
    for (int i = 0; i < nnz; ++i) {
        Mat[i] = val;
    }
}

void init_vecB(const int rows, const int cols, float* Mat, float val) {
    int t = 1;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; ++j) {
            Mat[i * cols + j] = t;
        }
        t += 1;
    }
}

// GpuTimer实现
GpuTimer::GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

GpuTimer::~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void GpuTimer::Start() {
    cudaEventRecord(start);
}

void GpuTimer::Stop() {
    cudaEventRecord(stop);
}

float GpuTimer::Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
}

// Dur结构构造函数实现
Dur::Dur(clocktype x, clocktype y, int outsm) : begin(x), end(y), smid(outsm) {}
