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
        static constexpr int sc0 = RegTileA::kRows / 4;
        static constexpr int sc1 = RegTileA::kCols / 2;

        static_assert(RegTileB::kRows / 4 == sc0 && RegTileC::kRows / 4,
                      "dimension mismatch");
        static_assert(RegTileB::kCols / 2 == sc1 && RegTileC::kCols / 2,
                      "dimension mismatch");

        const uint32_t* ra = reinterpret_cast<const uint32_t*>(a.data());
        const uint32_t* rb = reinterpret_cast<const uint32_t*>(b.data());
        uint32_t* rc = reinterpret_cast<uint32_t*>(c.mutable_data());

        // the data distribution of this specific wmma instruction over thread's
        // registers can be found in pp.19 of this document:
        // https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf

        auto base_tile_wmma = [&](const uint32_t* ra, const uint32_t* rb,
                                  uint32_t* rc) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, "
                "%2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(rc[0]), "=r"(rc[1]), "=r"(rc[2]), "=r"(rc[3])
                : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]),
                  "r"(rb[2]), "r"(rc[0]), "r"(rc[1]), "r"(rc[2]), "r"(rc[3]));

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, "
                "%2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(rc[4]), "=r"(rc[5]), "=r"(rc[6]), "=r"(rc[7])
                : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[1]),
                  "r"(rb[3]), "r"(rc[4]), "r"(rc[5]), "r"(rc[6]), "r"(rc[7]));
        };

        for (int i = 0; i < sc0; ++i) {
            for (int j = 0; j < sc1; ++j) {
                base_tile_wmma(ra, rb, rc);
                ra += 4;
                rb += 4;
                rc += 8;
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
