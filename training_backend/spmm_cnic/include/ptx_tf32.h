#pragma once

#include <cuda_runtime.h>
#include "config.h"

// 定义类型  common.h 文件中已出现
// using vint = int;
// using MAT_PTR_TYPE = int;
// using MAT_VAL_TYPE = float;

// WARP TOOLS
// __device__ __forceinline__ vint lane_id();
// __device__ __forceinline__ int lane_mask_lt();
// __device__ __forceinline__ vint warp_id();

// // MEMORY OPERATIONS
// __device__ __forceinline__ void async_copy(MAT_PTR_TYPE shared_addr, const MAT_VAL_TYPE* val);
// __device__ __forceinline__ void async_copy_idx(MAT_PTR_TYPE shared_addr, const vint* val);
// __device__ __forceinline__ MAT_VAL_TYPE load_fp32_from_global(const MAT_VAL_TYPE* a);
// __device__ __forceinline__ MAT_VAL_TYPE load_fp32_from_global_cs(const MAT_VAL_TYPE* a);
// __device__ __forceinline__ MAT_VAL_TYPE load_fp32_from_global2shared(const MAT_VAL_TYPE* a) ;
// __device__ __forceinline__ vint load_int_from_global(const vint* a);
// __device__ __forceinline__ void store_fp32_to_global(MAT_VAL_TYPE* a, MAT_VAL_TYPE v);
// __device__ __forceinline__ MAT_VAL_TYPE load_fp32_from_shared1(const MAT_PTR_TYPE a);
// __device__ __forceinline__ float4 vector_fetch_fp32V4(const float4* ptr);
// __device__ __forceinline__ float2 vector_fetch_fp32V2(const float2* ptr);
// __device__ __forceinline__ MAT_VAL_TYPE load_int_from_shared(const MAT_PTR_TYPE a);
// __device__ __forceinline__ float2 ld_shared_float2(uint a);
// __device__ __forceinline__ float4 ld_shared_float4(uint a);
// __device__ __forceinline__ uint getSMId();


// __device__ __forceinline__
// void tf32_m16n8k8(MAT_VAL_TYPE* MatA, MAT_VAL_TYPE* MatB, MAT_VAL_TYPE* MatC);

// __device__ __forceinline__
// void tf32_m16n8k8_detail(
//     MAT_VAL_TYPE MatA0, 
//     MAT_VAL_TYPE MatA1, 
//     MAT_VAL_TYPE MatA2, 
//     MAT_VAL_TYPE MatA3, 
//     MAT_VAL_TYPE MatB0,
//     MAT_VAL_TYPE MatB1, 
//     MAT_VAL_TYPE C0,
//     MAT_VAL_TYPE C1,
//     MAT_VAL_TYPE C2,
//     MAT_VAL_TYPE C3
// );

// __device__ __forceinline__
// void tf32_m16n8k4(MAT_VAL_TYPE* MatA, MAT_VAL_TYPE* MatB, MAT_VAL_TYPE* MatC);

// __device__ __forceinline__
// void wait_group();

__device__ __forceinline__ MAT_VAL_TYPE load_fp32_from_global2shared(const MAT_VAL_TYPE* a) {
    MAT_VAL_TYPE r;
    asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(r) : "l"(a));
    return r;
}
__device__ __forceinline__
void tf32_m16n8k8(MAT_VAL_TYPE* MatA, MAT_VAL_TYPE* MatB, MAT_VAL_TYPE* MatC) {
    vint const* A   = reinterpret_cast<vint const*>(MatA);
    vint const* B   = reinterpret_cast<vint const*>(MatB);
    float* C        = reinterpret_cast<float*>(MatC);

    asm volatile(
        "cvt.rna.tf32.f32 %4, %4;\n"
        "cvt.rna.tf32.f32 %5, %5;\n"
        "cvt.rna.tf32.f32 %6, %6;\n"
        "cvt.rna.tf32.f32 %7, %7;\n"
        "cvt.rna.tf32.f32 %8, %8;\n"
        "cvt.rna.tf32.f32 %9, %9;\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%0, %1, %2, %3};\n"
        :"+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])      // output
        :"r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
         "r"(B[0]), "r"(B[1])
    );
}

__device__ __forceinline__
void fp64_m16n8k8(float* MatA, float* MatB, float* MatC) {
    double A[4], B[2], C[4];  // 使用 double 进行计算

    // 1. 转换输入数据 float -> double
    A[0] = static_cast<double>(MatA[0]);
    A[1] = static_cast<double>(MatA[1]);
    A[2] = static_cast<double>(MatA[2]);
    A[3] = static_cast<double>(MatA[3]);

    B[0] = static_cast<double>(MatB[0]);
    B[1] = static_cast<double>(MatB[1]);

    C[0] = static_cast<double>(MatC[0]);
    C[1] = static_cast<double>(MatC[1]);
    C[2] = static_cast<double>(MatC[2]);
    C[3] = static_cast<double>(MatC[3]);

    // 2. 执行 double 精度的 MMA 计算
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%0, %1, %2, %3};\n"
        :"+d"(C[0]), "+d"(C[1]), "+d"(C[2]), "+d"(C[3])      // output
        :"d"(A[0]), "d"(A[1]), "d"(A[2]), "d"(A[3]),
         "d"(B[0]), "d"(B[1])
    );

    // 3. 转换计算结果 double -> float
    MatC[0] = static_cast<float>(C[0]);
    MatC[1] = static_cast<float>(C[1]);
    MatC[2] = static_cast<float>(C[2]);
    MatC[3] = static_cast<float>(C[3]);
}

__device__ __forceinline__
void tf32_m16n8k8_detail(
    MAT_VAL_TYPE MatA0, 
    MAT_VAL_TYPE MatA1, 
    MAT_VAL_TYPE MatA2, 
    MAT_VAL_TYPE MatA3, 
    MAT_VAL_TYPE MatB0,
    MAT_VAL_TYPE MatB1, 
    MAT_VAL_TYPE C0,
    MAT_VAL_TYPE C1,
    MAT_VAL_TYPE C2,
    MAT_VAL_TYPE C3
) {
    vint const A0   = static_cast<vint const>(MatA0);
    vint const A1   = static_cast<vint const>(MatA1);
    vint const A2   = static_cast<vint const>(MatA2);
    vint const A3   = static_cast<vint const>(MatA3);
    vint const B0   = static_cast<vint const>(MatB0);
    vint const B1   = static_cast<vint const>(MatB1);
    // float C0        = static_cast<float>(MatC0);
    // float C1        = static_cast<float>(MatC1);
    // float C2        = static_cast<float>(MatC2);
    // float C3        = static_cast<float>(MatC3);

    asm volatile(
        "cvt.rna.tf32.f32 %4, %4;\n"
        "cvt.rna.tf32.f32 %5, %5;\n"
        "cvt.rna.tf32.f32 %6, %6;\n"
        "cvt.rna.tf32.f32 %7, %7;\n"
        "cvt.rna.tf32.f32 %8, %8;\n"
        "cvt.rna.tf32.f32 %9, %9;\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%0, %1, %2, %3};\n"
        :"+f"(C0), "+f"(C1), "+f"(C2), "+f"(C3)      // output
        :"r"(A0), "r"(A1), "r"(A2), "r"(A3),
         "r"(B0), "r"(B1)
    );
}

__device__ __forceinline__
void tf32_m16n8k4(MAT_VAL_TYPE* MatA, MAT_VAL_TYPE* MatB, MAT_VAL_TYPE* MatC) {
    vint const* A   = reinterpret_cast<vint const*>(MatA);
    vint const* B   = reinterpret_cast<vint const*>(MatB);
    float *C        = reinterpret_cast<float*>(MatC);

    asm volatile(
        "cvt.rna.tf32.f32 %4, %4;\n"
        "cvt.rna.tf32.f32 %5, %5;\n"
        "cvt.rna.tf32.f32 %6, %6;\n"
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32"
        "{%0, %1, %2, %3},"
        "{%4, %5},"
        "{%6},"
        "{%0, %1, %2, %3};\n"
        :"+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])      // output
        :"r"(A[0]), "r"(A[1]),
         "r"(B[0])
    );
}

__device__ __forceinline__
void wait_group() {
    asm volatile(
        "cp.async.commit_group;\n"
        "cp.async.wait_group 0;\n"
        ::
    );
}



// ======================== WARP TOOLS ========================

__device__ __forceinline__ vint lane_id() {
    unsigned r;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(r));
    return r;
}

__device__ __forceinline__ int lane_mask_lt() {
    int mask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
}

// __device__ __forceinline__ vint warp_id() {
//     return threadIdx.x >> 5;
// }




// ====================== MEMORY OPERATIONS ======================

__device__ __forceinline__ void async_copy(MAT_PTR_TYPE shared_addr, const MAT_VAL_TYPE* val) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shared_addr), "l"(val));
}

__device__ __forceinline__ void async_copy_idx(MAT_PTR_TYPE shared_addr, const vint* val) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shared_addr), "l"(val));
}

__device__ __forceinline__ MAT_VAL_TYPE load_fp32_from_global(const MAT_VAL_TYPE* a) {
    MAT_VAL_TYPE r;
    asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(r) : "l"(a));
    return r;
}

__device__ __forceinline__ MAT_VAL_TYPE load_fp32_from_global_cs(const MAT_VAL_TYPE* a) {
    MAT_VAL_TYPE r;
    asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(r) : "l"(a));
    return r;
}



__device__ __forceinline__ vint load_int_from_global(const vint* a) {
    int r;
    asm volatile("ld.global.cv.s32 %0, [%1];" : "=r"(r) : "l"(a));
    return r;
}

__device__ __forceinline__ void store_fp32_to_global(MAT_VAL_TYPE* a, MAT_VAL_TYPE v) {
    asm volatile("st.global.wt.f32 [%0], %1;" :: "l"(a), "f"(v));
}

__device__ __forceinline__ MAT_VAL_TYPE load_fp32_from_shared1(const MAT_PTR_TYPE a) {
    MAT_VAL_TYPE r;
    asm volatile("ld.shared.cs.f32 %0, [%1];" : "=f"(r) : "r"(a));
    return r;
}

__device__ __forceinline__ float4 vector_fetch_fp32V4(const float4* ptr) {
    float4 ret;
    asm volatile(
        "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
        : "l"(ptr));
    return ret;
}

__device__ __forceinline__ float2 vector_fetch_fp32V2(const float2* ptr) {
    float2 ret;
    asm volatile(
        "ld.global.v2.f32 {%0, %1}, [%2];"
        : "=f"(ret.x), "=f"(ret.y)
        : "l"(ptr));
    return ret;
}

__device__ __forceinline__ MAT_VAL_TYPE load_int_from_shared(const MAT_PTR_TYPE a) {
    vint r;
    asm volatile("ld.shared.cs.s32 %0, [%1];" : "=r"(r) : "r"(a));
    return r;
}

__device__ __forceinline__ float2 ld_shared_float2(uint a) {
    float2 v;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(a * 4));
    return v;
}

__device__ __forceinline__ float4 ld_shared_float4(uint a) {
    float4 v;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "r"(a * 4));
    return v;
}

__device__ __forceinline__ uint getSMId() {
    uint smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}
