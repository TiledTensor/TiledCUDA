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
        const uint32_t* ra = reinterpret_cast<const uint32_t*>(a.data());
        const uint32_t* rb = reinterpret_cast<const uint32_t*>(b.data());
        uint32_t* rc = reinterpret_cast<uint32_t*>(c.mutable_data());

        // the data distribution of this specific wmma instruction over thread's
        // registers can be found in pp.19 of this document:
        // https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, "
            "%3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(rc[0]), "=r"(rc[1]), "=r"(rc[2]), "=r"(rc[3])
            : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]),
              "r"(rb[2]), "r"(rc[0]), "r"(rc[1]), "r"(rc[2]), "r"(rc[3]));

        if (threadIdx.x == 0) {
            const half* reg1 = reinterpret_cast<const half*>(a.data());
            const half* reg2 = reinterpret_cast<const half*>(b.data());
            const float* reg3 = reinterpret_cast<const float*>(c.data());

            printf("[A-tid = %d]:\t", threadIdx.x);
            for (int i = 0; i < 8; ++i) printf("%.2f, ", __half2float(reg1[i]));

            printf("\npart 1, [B-tid = %d]:\t", threadIdx.x);
            printf("%.2f, %.2f, %.2f, %.2f\n", __half2float(reg2[0]),
                   __half2float(reg2[1]), __half2float(reg2[4]),
                   __half2float(reg2[5]));

            printf("part 1, [C-tid = %d]:\t", threadIdx.x);
            for (int i = 0; i < 4; ++i) printf("%.2f, ", reg3[i]);
            printf("\n");
        }

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, "
            "%3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(rc[4]), "=r"(rc[5]), "=r"(rc[6]), "=r"(rc[7])
            : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[1]),
              "r"(rb[3]), "r"(rc[4]), "r"(rc[5]), "r"(rc[6]), "r"(rc[7]));

        if (threadIdx.x == 0) {
            const half* reg2 = reinterpret_cast<const half*>(b.data());
            const float* reg3 = reinterpret_cast<const float*>(c.data());

            printf("\npart 2, [B-tid = %d]:\t", threadIdx.x);
            printf("%.2f, %.2f, %.2f, %.2f\n", __half2float(reg2[2]),
                   __half2float(reg2[3]), __half2float(reg2[6]),
                   __half2float(reg2[7]));

            printf("part 2, [C-tid = %d]:\t", threadIdx.x);
            for (int i = 4; i < 8; ++i) printf("%.2f, ", reg3[i]);
        }
    }
};

}  // namespace detail

template <typename RegTileA, typename RegTileB, typename RegTileC>
DEVICE void wmma(const RegTileA& a, const RegTileB& b, RegTileC& c) {
    detail::Gemm<RegTileA, RegTileB, RegTileC, InstShape<16, 16, 16>> gemm;
    gemm(a, b, c);
}

}  // namespace tiledcuda::cell::compute
