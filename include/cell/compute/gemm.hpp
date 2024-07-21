#pragma once

#include "cell/traits/base.hpp"
#include "cuda_utils.hpp"
#include "types/layout.hpp"
#include "types/tile_shape.hpp"

namespace tiledcuda::cell::compute {
namespace tl = tile_layout;

template <typename TensorA, typename TensorB, typename TensorAcc,
          typename TiledMma>
DEVICE void gemm(const TiledMma& mma, const TensorA& a, const TensorB& b,
                 TensorAcc& acc) {
    cute::gemm(mma, a, b, acc);
}

namespace detail {

/// @brief: Functor to warp wmma PTX instruction. See the below document for
///         various choices and detailed parameters of the wmma PTX instruction.
///         https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
template <typename RegTileA, typename RegTileB, typename RegTileC>
struct Gemm {
    using InTypeA = typename RegTileA::DType::DType;
    using InTypeB = typename RegTileB::DType::DType;
    using OutType = typename RegTileC::DType::DType;

    using BaseShape = traits::BaseTileShape<InTypeA>;

    static_assert(std::is_same_v<InTypeA, cutlass::half_t> ||
                      std::is_same_v<InTypeA, __half>,
                  "This GEMM implementation supports only half-precision as "
                  "the input element type.");
    static_assert(std::is_same_v<OutType, float>,
                  "The output type must be float.");
    static_assert(std::is_same_v<InTypeA, InTypeB>,
                  "Mismatched data type for operand A and B.");
    static_assert(RegTileB::kRows == RegTileA::kCols,
                  "Mismatched k-dimension for operand A and B.");

    static constexpr int kMs = RegTileA::kRows;
    static constexpr int kNs = RegTileB::kCols;
    static constexpr int kKs = RegTileA::kCols;
    static_assert(kMs && kNs && kKs, "Invalid tile shapes for GEMM.");

    DEVICE void operator()(const RegTileA& a, const RegTileB& b, RegTileC& c) {
        if (thread0()) {
            printf("GEMM: ms = %d, ns = %d,  ks = %d\n", kMs, kNs, kKs);

            printf("A:\n");
            a.dump_value();

            printf("\nB:\n");
            b.dump_value();
            printf("\n");
        }

        for (int i = 0; i < kMs; ++i) {
            for (int j = 0; j < kNs; ++j) {
                for (int k = 0; k < kKs; ++k) {
                    if (thread0()) {
                        printf("(%d, %d, %d):\n", i, j, k);

                        printf("rA:\n");
                        a(i, k).dump_value();

                        printf("\nrB:\n");
                        b(k, j).dump_value();
                        printf("\n");
                    }

                    tile_wmma(a(i, k).data(), b(k, j).data(),
                              c(i, j).mutable_data());
                }
            }
        }
    }

  private:
    DEVICE void tile_wmma(const InTypeA* ra, const InTypeB* rb, float* C) {
        const uint32_t* A = reinterpret_cast<const uint32_t*>(ra);
        const uint32_t* B = reinterpret_cast<const uint32_t*>(rb);

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
};

}  // namespace detail

template <typename RegTileA, typename RegTileB, typename RegTileC>
DEVICE void gemm_(const RegTileA& a, const RegTileB& b, RegTileC& c) {
    detail::Gemm<RegTileA, RegTileB, RegTileC> gemm;
    gemm(a, b, c);
}

}  // namespace tiledcuda::cell::compute
