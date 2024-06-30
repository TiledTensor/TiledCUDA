#pragma once

#include "cuda_utils.hpp"
#include "types/layout.hpp"
#include "types/tile_shape.hpp"

namespace tiledcuda::cell::compute {

namespace tl = tiledcuda::cell::tile_layout;

template <typename TensorA, typename TensorB, typename TensorAcc,
          typename TiledMma>
DEVICE void gemm(const TiledMma& mma, const TensorA& a, const TensorB& b,
                 TensorAcc& acc) {
    cute::gemm(mma, a, b, acc);
}

namespace detail {

// functor to warp ptx wmma instruction
// see the below document for various choices and detailed parameters of the
// wmma PTX instruction.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
template <typename RegTileA, typename RegTileB, typename RegTileC,
          typename InstShape>
struct Gemm {
    DEVICE void operator()(const RegTileA& a, const RegTileB& b, RegTileC& c);
};

// partial specialization for wmma 16x16x16
template <typename RegTileA, typename RegTileB, typename RegTileC>
struct Gemm<RegTileA, RegTileB, RegTileC, InstShape<16, 16, 16>> {
    DEVICE void operator()(const RegTileA& a, const RegTileB& b, RegTileC& c) {
        static constexpr int sc0 = RegTileA::kRows / 2;  // dim m
        static constexpr int sc1 = RegTileA::kCols / 4;  // dim k
        static_assert(RegTileB::kCols / 4 == sc1,
                      "Mismatched k-dimension for operand A and B.");

        const uint32_t* ra = reinterpret_cast<const uint32_t*>(a.data());
        const uint32_t* rb = reinterpret_cast<const uint32_t*>(b.data());
        float* rc = c.mutable_data();

        auto base_tile_wmma = [&](const uint32_t* A, const uint32_t* B,
                                  float* C) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10, %11, %12, %13};\n"
                : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]),
                  "r"(B[2]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10, %11, %12, %13};\n"
                : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[1]),
                  "r"(B[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
        };

        for (int i = 0; i < sc0; ++i) {
            for (int j = 0; j < sc1; ++j) {  // accumulate along the k dimension
                base_tile_wmma(ra, rb, rc);
                ra += 4;
                rb += 4;
            }
            rc += 8;
        }
    }
};

}  // namespace detail

template <typename RegTileA, typename RegTileB, typename RegTileC>
DEVICE void gemm_(const RegTileA& a, const RegTileB& b, RegTileC& c) {
    detail::Gemm<RegTileA, RegTileB, RegTileC, InstShape<16, 16, 16>> gemm;
    gemm(a, b, c);
}

}  // namespace tiledcuda::cell::compute
