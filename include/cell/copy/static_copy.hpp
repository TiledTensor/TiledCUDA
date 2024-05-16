#pragma once

#include "cuda_utils.hpp"

#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace tiledcuda::cell::copy {

using namespace cute;

template <typename Element, typename TiledMma_, typename DstLayout>
struct R2SCopy2D {
    using TiledMma = TiledMma_;
    using Dstlayout_ = DstLayout;
    using CopyAtom = Copy_Atom<DefaultCopy, Element>;

  public:
    template <typename Engine, typename Layout>
    DEVICE void copy(cute::Tensor<Engine, Layout> const& acc, Element* dst_data,
                     int tid) {
        // FIXME(haruhi): This implementation is specifically designed
        // for tcu WMMA and assumes that the ACC value has a
        // floating-point precision. The code converts the ACC value
        // to half-precision.
        auto src_tensor = convert_type<Element>(acc);
        auto dst_tensor = make_tensor(make_smem_ptr(dst_data), DstLayout{});

        auto tiled_copy = make_tiled_copy_C(CopyAtom{}, TiledMma{});
        auto thrd_copy = tiled_copy.get_thread_slice(tid);

        auto src = thrd_copy.retile_S(src_tensor);
        auto dst = thrd_copy.partition_D(dst_tensor);
        cute::copy(tiled_copy, src, dst);
    }

  private:
    template <typename To_type, typename Engine, typename Layout>
    DEVICE auto convert_type(cute::Tensor<Engine, Layout> const& tensor) {
        using From_type = typename Engine::value_type;
        constexpr int numel = decltype(size(tensor))::value;
        cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
        // HACK: this requires tensor to be "contiguous"
        auto frag = convert_op(
            *reinterpret_cast<const cutlass::Array<From_type, numel>*>(
                tensor.data()));
        return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
    }
};

#define DEBUG

/**
 * @brief (haruhi) a naive implementation to experiment with the interface.
 * Consider better implementations.
 *
 * @tparam src: the source data pointer.
 * @tparam s_layout: the layout of the source data.
 * @tparam dst: the destination data pointer.
 * @tparam d_layout: the layout of the destination data.
 * @tparam t_layout: the layout of the threads.
 */
template <typename SrcTile, typename DstTile>
DEVICE void copy_2d_tile_s2r(const SrcTile& src, DstTile& dst) {
    // #ifdef DEBUG_
    //     {
    //         if (threadIdx.x == 0) {
    //             half* data =
    //             reinterpret_cast<half*>(const_cast<Element*>(src)); for (int
    //             i = 0; i < 16; ++i) {
    //                 for (int j = 0; j < 16; ++j)
    //                     printf("%.0f, ", __half2float(data[i * 32 + k * 16 +
    //                     j]));
    //                 printf("\n");
    //             }
    //             printf("\n");
    //         }
    //     }
    // #endif

    // const size_t lane_id = threadIdx.x % warpSize;
    // const size_t i = lane_id % 16;
    // const size_t j = lane_id / 16;

    // // the shared memory address accessed by the current thread
    // uint32_t smem_addr =
    //     __cvta_generic_to_shared(src + i * 32 + k * 16 + j * 8);

    // uint32_t* reg_addr = reinterpret_cast<uint32_t*>(dst);

    // FIXME(haruhi): ldmatrix has more options to be explored.
    // the hard-coded implmentation here loads 4 x (8 x 8) numbers in halfs
    // simultaneously. Check the documents for more usages.
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix
    // asm volatile(
    //         "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3},
    //         [%4];\n" : "=r"(*reg_addr), "=r"(*(reg_addr + 1)),
    //         "=r"(*(reg_addr + 2)),
    //           "=r"(*(reg_addr + 3))
    //         : "r"(smem_addr));

    // #ifdef DEBUG
    //     {
    //         __half* data = reinterpret_cast<__half*>(dst);
    //         printf("[%d]: %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f,
    //         %.0f\n\n",
    //                lane_id, __half2float(data[0]), __half2float(data[1]),
    //                __half2float(data[2]), __half2float(data[3]),
    //                __half2float(data[4]), __half2float(data[5]),
    //                __half2float(data[6]), __half2float(data[7]));
    //     }
    // #endif

}  // namespace tiledcuda::cell::copy

DEVICE void copy_2d_tile_r2s() {}

}  // namespace tiledcuda::cell::copy
