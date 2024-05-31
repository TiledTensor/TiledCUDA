#pragma once
#include "cuda_utils.hpp"
#include "types/tile_shape.hpp"

#include <cute/tensor.hpp>

namespace tiledcuda::cell::compute {

template <typename TensorA, typename TensorB, typename TensorAcc,
          typename TiledMma>
DEVICE void gemm(const TiledMma& mma, const TensorA& a, const TensorB& b,
                 TensorAcc& acc) {
    cute::gemm(mma, a, b, acc);
}

template <const int kM, const int kN, const int kK>
using InstShape = TileShape<kM, kN, kK>;

namespace detail {

// functor to warp ptx wmma instruction
template <typename RegTileA, typename RegTileB, typename RegTileC,
          typename InstShape>
struct Gemm {
    DEVICE void operator()(const RegTileA& a, const RegTileB& b, RegTileC& c);
};

// partial specialization for wmma 16x16x16
template <typename RegTileA, typename RegTileB, typename RegTileC>
struct Gemm<RegTileA, RegTileB, RegTileC, InstShape<16, 16, 16>> {
    DEVICE void operator()(const RegTileA& a, const RegTileB& b, RegTileC& c) {
        const uint32_t* reg_a = reinterpret_cast<const uint32_t*>(a.data());
        const uint32_t* reg_b = reinterpret_cast<const uint32_t*>(b.data());
        uint32_t* reg_c = reinterpret_cast<uint32_t*>(c.mutable_data());

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, "
            "%3, "
            "%4, %5}, {%6, %7}, {%8, %9};\n"
            : "=r"(reg_c[0]), "=r"(reg_c[1])
            : "r"(reg_a[0]), "r"(reg_a[1]), "r"(reg_a[2]), "r"(reg_a[3]),
              "r"(reg_b[0]), "r"(reg_b[1]), "r"(reg_c[0]), "r"(reg_c[1]));

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, "
            "%3, "
            "%4, %5}, {%6, %7}, {%8, %9};\n"
            : "=r"(reg_c[2]), "=r"(reg_c[3])
            : "r"(reg_a[0]), "r"(reg_a[1]), "r"(reg_a[2]), "r"(reg_a[3]),
              "r"(reg_b[1]), "r"(reg_b[2]), "r"(reg_c[2]), "r"(reg_c[3]));
    }
};

}  // namespace detail

template <typename RegTileA, typename RegTileB, typename RegTileC>
DEVICE void wmma(const RegTileA& a, const RegTileB& b, RegTileC& c) {
    detail::Gemm<RegTileA, RegTileB, RegTileC, InstShape<16, 16, 16>> gemm;
    gemm(a, b, c);
}

}  // namespace tiledcuda::cell::compute
