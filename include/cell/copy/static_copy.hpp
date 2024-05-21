#pragma once

#include "cuda_utils.hpp"
#include "types/common.hpp"

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

/*
 * @brief FIXME(haruhi): a naive implementation to experiment with the
 * interface. Consider better implementations.
 *
 * @tparam pos: the shared memory positions accessed by the current thread.
 * @tparam dst: the dst data tile.
 */
template <typename SrcPtrs, typename DstTile>
DEVICE void copy_2d_tile_s2r(SrcPtrs& pos, DstTile& dst) {
    static_assert(SrcPtrs::kSize == DstTile::kExecCount,
                  "The data tile a single thread load from shared memory to "
                  "its local register should have a equal shape.");

    uint32_t smem_addr;
    // cast to pointer pointing to 128-bit register array
    uint32_t* reg = reinterpret_cast<uint32_t*>(dst.mutable_data());

    // issue the atomic memory access instruction in a temporal fashion
    for (int i = 0; i < SrcPtrs::kSize; ++i) {
        smem_addr = __cvta_generic_to_shared(pos[i]);

        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(*(reg)), "=r"(*(reg + 1)), "=r"(*(reg + 2)), "=r"(*(reg + 3))
            : "r"(smem_addr));

        if (threadIdx.x == 0) {
            half* r = (half*)(reg);
            printf("[%d]: %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, % .0f\n",
                   threadIdx.x, __half2float(r[0]), __half2float(r[1]),
                   __half2float(r[2]), __half2float(r[3]), __half2float(r[4]),
                   __half2float(r[5]), __half2float(r[6]), __half2float(r[7]));
        }

        // The magic number 4 here is computed as follows:
        // Base::kNumPerAccess / 8 / sizeof(uint32_t);
        // each thread access 128-bit data that equals to 4 uint32_t elements.
        reg += 4;  // advance pointer
    }
}

DEVICE void copy_2d_tile_r2s() {
    // TODO: not implemented yet
}

}  // namespace tiledcuda::cell::copy
