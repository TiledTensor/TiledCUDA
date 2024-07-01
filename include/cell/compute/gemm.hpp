#pragma once

#include "cell/traits/base.hpp"
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
    using BaseShape =
        tiledcuda::cell::traits::BaseTileShape<typename RegTileA::DType>;

    using InType = typename RegTileA::DType;
    using OutType = typename RegTileC::DType;

    static_assert(std::is_same<InType, cutlass::half_t>::value,
                  "Only half precision is supported for now.");
    static_assert(std::is_same<InType, typename RegTileB::DType>::value,
                  "Mismatched data type for operand A and B.");
    static_assert(std::is_same<OutType, float>::value, "Output must be float.");

    // A register tile's contiguous dimension maps to the fragment's contiguous
    // dimension, and the strided dimension maps to the strided dimension.
    static constexpr int ms = RegTileA::kRows / BaseShape::sub_row;
    static constexpr int ns = RegTileB::kCols / BaseShape::sub_row;
    static constexpr int ks = RegTileA::kCols / BaseShape::sub_col;

    // FIXME (haruhi): This check requires operand B to have a column-major
    // layout of [K, N], meaning the k dimension is contiguous in memory.
    static_assert(RegTileB::kRows / BaseShape::sub_col == ks,
                  "Mismatched k-dimension for operand A and B.");

    static_assert(ms && ns && ks, "Invalid tile shapes for GEMM.");

    DEVICE void tile_wmma(const uint32_t* A, const uint32_t* B, float* C) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[2]),
              "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[1]), "r"(B[3]),
              "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    }

    DEVICE void operator()(const RegTileA& a, const RegTileB& b, RegTileC& c) {
        const uint32_t* ra = reinterpret_cast<const uint32_t*>(a.data());
        const uint32_t* rb = reinterpret_cast<const uint32_t*>(b.data());
        float* rc = c.mutable_data();

        int offset_a = 0, offset_b = 0, offset_c = 0;
        for (int i = 0; i < ms; ++i) {
            for (int j = 0; j < ns; ++j) {
                offset_c = (i * ns + j) * 8;
                for (int k = 0; k < ks; ++k) {
                    offset_a = (i * ks + k) * 4;
                    offset_b = (j * ks + k) * 4;

                    tile_wmma(&ra[offset_a], &rb[offset_b], &rc[offset_c]);
                }
            }
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
