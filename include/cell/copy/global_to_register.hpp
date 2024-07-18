#pragma once

#include "cell/copy/constants.hpp"
#include "cell/copy/warp.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

using namespace tiledcuda::cell::traits;

template <typename Global_, typename Reg_, const int kRowExec_,
          const int kColExec_, const tl::Layout type>
struct GlobalToRegLoaderImpl {
    using Global = Global_;
    using Reg = Reg_;
    using DType = Global::type;

    DEVICE void operator()(const DType* src, Reg& dst);
};

template <typename Global_, typename Reg_, const int kRowExec_,
          const int kColExec_>
struct GlobalToRegLoaderImpl<Global_, Reg_, kRowExec_, kColExec_,
                             tl::Layout::RowMajor> {
    using Global = Global_;
    using Reg = Reg_;
    using DType = Global::DType;

    // strides to iterate over each 16x128-bits `BaseTile`.
    static constexpr int kTileRstride =
        BaseTileShape<DType>::row * Global::kRowStride;
    static constexpr int kTileCstride = BaseTileShape<DType>::col;

    // row stride and col stride for a thread in a warp
    static constexpr int kStride = traits::TraitsBase<DType>::kNumPerAccess;
    static constexpr int kLaneRstride = Global::kRowStride;
    static constexpr int kLaneCstride = kStride;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const DType* src, Reg& dst) {
        int lane_id = threadIdx.x % warpSize;

        const DType* data = src;
#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
            // TODO: magic number.
            int row = i * 16 + lane_id / 4;
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                // TODO: magic number.
                int col = j * 16 + (lane_id % 4) * 2;
                dst(i, j)(0, 0) =
                    data[(row + 0) * Global::kRowStride + col + 0];
                dst(i, j)(0, 1) =
                    data[(row + 0) * Global::kRowStride + col + 1];
                dst(i, j)(1, 0) =
                    data[(row + 0) * Global::kRowStride + col + 8];
                dst(i, j)(1, 1) =
                    data[(row + 0) * Global::kRowStride + col + 9];
                dst(i, j)(0, 2) =
                    data[(row + 8) * Global::kRowStride + col + 0];
                dst(i, j)(0, 3) =
                    data[(row + 8) * Global::kRowStride + col + 1];
                dst(i, j)(1, 2) =
                    data[(row + 8) * Global::kRowStride + col + 8];
                dst(i, j)(1, 3) =
                    data[(row + 8) * Global::kRowStride + col + 9];
            }
        }
    }
};

template <typename Global_, typename Reg_, typename WarpLayout_,
          const WarpReuse kMode_,
          typename Base = warp::CopyBase<WarpLayout_, kMode_>>
struct GlobalToRegLoader : public Base {
    using Global = Global_;
    using Reg = Reg_;
    using DType = typename Global::DType;

    using WarpLayout = WarpLayout_;
    static constexpr WarpReuse kMode = kMode_;

    DEVICE void operator()(const Global& src, Reg& dst) {
        const DType* src_ptr = src.data();

        // 1. advance the pointer to input data to the current warp according to
        // warp reuse mode.
        src_ptr += Base::template get_warp_offset<Global>();

        // how many times a `BaseTile` is executed along the row and column
        // direction.
        static constexpr int kRowExec =
            Base::template row_exec_count<DType, Global::kRows>();
        static constexpr int kColExec =
            Base::template col_exec_count<DType, Global::kCols>();

        // TODO: Load data from global memory to register.
        using Loader = GlobalToRegLoaderImpl<Global, Reg, kRowExec, kColExec,
                                             Global::type>;
        Loader loader;
        loader(src_ptr, dst);
    }
};

}  // namespace tiledcuda::cell::copy
