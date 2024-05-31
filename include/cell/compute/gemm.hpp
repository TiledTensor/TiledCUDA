#pragma once
#include "cuda_utils.hpp"

#include <cute/tensor.hpp>

namespace tiledcuda::cell::compute {

template <typename TensorA, typename TensorB, typename TensorAcc,
          typename TiledMma>
DEVICE void gemm(const TiledMma& mma, const TensorA& a, const TensorB& b,
                 TensorAcc& acc) {
    cute::gemm(mma, a, b, acc);  // compute
}

template <typename RegA, typename RegB, typename RegC>
DEVICE void wmma(const RegA& a, const RegB& b, RegC& c) {
    const uint32_t* rA = reinterpret_cast<const uint32_t*>(a.data());
    const uint32_t* rB = reinterpret_cast<const uint32_t*>(b.data());
    uint32_t* rC = reinterpret_cast<uint32_t*>(c.mutable_data());

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, "
        "%4, %5}, {%6, %7}, {%8, %9};\n"
        : "=r"(rC[0]), "=r"(rC[1])
        : "r"(rA[0]), "r"(rA[1]), "r"(rA[2]), "r"(rA[3]), "r"(rB[0]),
          "r"(rB[1]), "r"(rC[0]), "r"(rC[1]));
}

}  // namespace tiledcuda::cell::compute
