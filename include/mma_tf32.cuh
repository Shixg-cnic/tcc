#pragma once

#include <cstdint>

namespace tcc {

__device__ __forceinline__
void mmaTf32M16N8K8(
    const float a[4],
    const float b[2],
    float c[4]
) {
    const auto* aBits = reinterpret_cast<const std::uint32_t*>(a);
    const auto* bBits = reinterpret_cast<const std::uint32_t*>(b);

    asm volatile(
        "cvt.rna.tf32.f32 %4, %4;\n"
        "cvt.rna.tf32.f32 %5, %5;\n"
        "cvt.rna.tf32.f32 %6, %6;\n"
        "cvt.rna.tf32.f32 %7, %7;\n"
        "cvt.rna.tf32.f32 %8, %8;\n"
        "cvt.rna.tf32.f32 %9, %9;\n"

        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"

        : "+f"(c[0]),
          "+f"(c[1]),
          "+f"(c[2]),
          "+f"(c[3])

        : "r"(aBits[0]),
          "r"(aBits[1]),
          "r"(aBits[2]),
          "r"(aBits[3]),
          "r"(bBits[0]),
          "r"(bBits[1])

        
    );


}


}