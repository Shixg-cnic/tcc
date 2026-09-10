#include "mma_tf32.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace {
    
constexpr int TILE_ROWS = 8;
constexpr int K_DIM = 8;
constexpr int FEATURE_DIM = 16;
constexpr int TILE_COUNT = 2;
constexpr int OUTPUT_ROWS = TILE_ROWS * TILE_COUNT;


__global__ void mmaTestkernel(
    const float* matrixA,
    const float* matrixX,
    float* matrixY
) {
    const int lane = threadIdx.x;
    const int group = lane >> 2;        // group_id 0,1,2,3,4,5,6,7
    const int threadInGroup = lane & 3; // local_id

    float mmaA[4];

    // 按照物理地址去读
    mmaA[0] = matrixX[threadInGroup * FEATURE_DIM + group];
    mmaA[1] = matrixX[threadInGroup * FEATURE_DIM + group + 8];
    mmaA[2] = matrixX[(threadInGroup + 4) * FEATURE_DIM + group];
    mmaA[3] = matrixX[(threadInGroup + 4) * FEATURE_DIM + group + 8];

    

    for(int tile = 0; tile < TILE_COUNT; tile++) {
        float mmaB[2];
        float mmaC[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int tileAoffset = tile * TILE_ROWS * K_DIM;
        mmaB[0] = matrixA[tileAoffset + group * K_DIM + threadInGroup];
        mmaB[1] = matrixA[tileAoffset + group * K_DIM + threadInGroup + 4];

        tcc::mmaTf32M16N8K8(mmaA, mmaB, mmaC);
        const int outputRowBase = tile * TILE_ROWS;

        matrixY[(outputRowBase + threadInGroup * 2) * FEATURE_DIM + group] = mmaC[0];
        matrixY[(outputRowBase + threadInGroup * 2 + 1) * FEATURE_DIM + group] = mmaC[1];
        matrixY[(outputRowBase + threadInGroup * 2) * FEATURE_DIM + group + 8] = mmaC[2];
        matrixY[(outputRowBase + threadInGroup * 2 + 1) * FEATURE_DIM + group + 8] = mmaC[3];


    }


}
} //namespace

int main() {
    std::vector<float> matrixA(OUTPUT_ROWS * K_DIM, 0.0f);
    std::vector<float> matrixX(K_DIM * FEATURE_DIM, 0.0f);
    std::vector<float> matrixY(OUTPUT_ROWS * FEATURE_DIM, 0.0f);
    std::vector<float> expected(OUTPUT_ROWS * FEATURE_DIM, 0.0f);

    for(int row = 0; row < OUTPUT_ROWS; ++row) {
        const int tile = row / TILE_ROWS;
        const int localRow = row % TILE_ROWS;
        matrixA[row * K_DIM + localRow] = static_cast<float>(tile + 1);
        matrixA[row * K_DIM + (localRow + 1) % K_DIM] = static_cast<float>(tile + 2);
    }

    for(int row = 0; row < K_DIM; ++row) {
        for(int feature = 0; feature < FEATURE_DIM; ++feature) {
            matrixX[row * FEATURE_DIM + feature] = static_cast<float>((row + 1) * 10 + feature);
        }
    }

    for(int row = 0; row < OUTPUT_ROWS; ++row) {
        for(int feature = 0; feature < FEATURE_DIM; ++feature) {
            for(int k = 0; k < K_DIM; ++k) {
                expected[row * FEATURE_DIM + feature] += matrixA[row * K_DIM + k] * matrixX[k * FEATURE_DIM + feature];
            }
        }
    }

    float* deviceA = nullptr;
    float* deviceX = nullptr;
    float* deviceY = nullptr;

    cudaMalloc(&deviceA, matrixA.size() * sizeof(float));
    cudaMalloc(&deviceX, matrixX.size() * sizeof(float));
    cudaMalloc(&deviceY, matrixY.size() * sizeof(float));

    cudaMemcpy(deviceA, matrixA.data(), matrixA.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceX, matrixX.data(), matrixX.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(deviceY, 0, matrixY.size() * sizeof(float));

    mmaTestkernel<<<1, 32>>>(deviceA, deviceX, deviceY);
    cudaDeviceSynchronize();

    cudaMemcpy(matrixY.data(), deviceY, matrixY.size() * sizeof(float), cudaMemcpyDeviceToHost);

    bool passed = true;
    for(int row = 0; row < OUTPUT_ROWS; ++row) {
        for(int feature = 0; feature < FEATURE_DIM; ++feature) {
            const int index = row * FEATURE_DIM + feature;
            if(std::fabs(matrixY[index] - expected[index]) > 1.0e-4f) {
                std::cout << "Mismatch at (" << row << ", " << feature << "): GPU=" << matrixY[index] << ", CPU=" << expected[index] << '\n';
                passed = false;
            }
        }
    }

    cudaFree(deviceA);
    cudaFree(deviceX);
    cudaFree(deviceY);

    if(passed) {
        std::cout << "Two-tile X-reuse MMA test passed\n";
        return 0;
    }

    std::cout << "Two-tile X-reuse MMA test failed\n";
    return 1;
}
