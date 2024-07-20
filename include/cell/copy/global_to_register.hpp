#pragma once

#include "cell/copy/constants.hpp"
#include "cell/copy/warp.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {

using namespace tiledcuda::cell::traits;

/**
 * @brief Load a BastTile Matrix from Global memory to Register.
 * @tparam Global_ Global memory tile type.
 * @tparam BaseTile_ BaseTile type.
 * @tparam type Global Layout type.
 */
template <typename Global_, typename BaseTile_, const tl::Layout type>
struct GlobalToRegMatLoader {
    using Global = Global_;
    using BaseTile = BaseTile_;
    using DType = Global::DType;

    DEVICE void operator()(const DType* src, BaseTile& dst);
};

template <typename Global_, typename BaseTile_>
struct GlobalToRegMatLoader<Global_, BaseTile_, tl::Layout::RowMajor> {
    using Global = Global_;
    using BaseTile = BaseTile_;
    using DType = Global::DType;

    static constexpr int kStride = Global::kRowStride;

    DEVICE void operator()(const DType* src, BaseTile& dst) {
        dst(0, 0) = src[0 * kStride + 0];
        dst(0, 1) = src[0 * kStride + 1];
        dst(1, 0) = src[0 * kStride + 8];
        dst(1, 1) = src[0 * kStride + 9];
        dst(0, 2) = src[8 * kStride + 0];
        dst(0, 3) = src[8 * kStride + 1];
        dst(1, 2) = src[8 * kStride + 8];
        dst(1, 3) = src[8 * kStride + 9];
    }
};

template <typename Global_, typename BaseTile_>
struct GlobalToRegMatLoader<Global_, BaseTile_, tl::Layout::ColMajor> {
    using Global = Global_;
    using BaseTile = BaseTile_;
    using DType = Global::DType;

    static constexpr int kStride = Global::kColStride;

    DEVICE void operator()(const DType* src, BaseTile& dst) {
        dst(0, 0) = src[0 * kStride + 0];
        dst(1, 0) = src[0 * kStride + 1];
        dst(0, 1) = src[0 * kStride + 8];
        dst(1, 1) = src[0 * kStride + 9];
        dst(2, 0) = src[8 * kStride + 0];
        dst(3, 0) = src[8 * kStride + 1];
        dst(2, 1) = src[8 * kStride + 8];
        dst(3, 1) = src[8 * kStride + 9];
    }
};

/**
 * @brief Load a RegTile from Global memory to Register.
 * @tparam Global_ Global memory tile type.
 * @tparam Reg_ Register tile type.
 * @tparam kRowExec_ Number of times a `RegTile` is executed along the row
 * @tparam kColExec_ Number of times a `RegTile` is executed along the column
 * @tparam type Global Layout type.
 */
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
    using DType = typename Global::DType;
    using BaseTile = typename Reg::DType;

    // The size of a `BaseTile`.
    static constexpr int kTileSize = BaseTileShape<DType>::kTileSize;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const DType* src, Reg& dst) {
        int lane_id = threadIdx.x % warpSize;

        const DType* data;
#pragma unroll
        for (int i = 0; i < kRowExec; ++i) {
            int row = i * kTileSize + lane_id / 4;
#pragma unroll
            for (int j = 0; j < kColExec; ++j) {
                int col = j * kTileSize + (lane_id % 4) * 2;

                data = src + row * Global::kRowStride + col;
                using Loader = GlobalToRegMatLoader<Global, BaseTile,
                                                    tl::Layout::RowMajor>;
                Loader loader;
                loader(data, dst(i, j));
            }
        }
    }
};

template <typename Global_, typename Reg_, const int kRowExec_,
          const int kColExec_>
struct GlobalToRegLoaderImpl<Global_, Reg_, kRowExec_, kColExec_,
                             tl::Layout::ColMajor> {
    using Global = Global_;
    using Reg = Reg_;
    using DType = typename Global::DType;
    using BaseTile = typename Reg::DType;

    // The size of a `BaseTile`.
    static constexpr int kTileSize = BaseTileShape<DType>::kTileSize;

    static constexpr int kRowExec = kRowExec_;
    static constexpr int kColExec = kColExec_;

    DEVICE void operator()(const DType* src, Reg& dst) {
        int lane_id = threadIdx.x % warpSize;

        const DType* data;

#pragma unroll
        for (int i = 0; i < kColExec; ++i) {
            int col = i * kTileSize + lane_id / 4;
            for (int j = 0; j < kRowExec; ++j) {
                int row = j * kTileSize + (lane_id % 4) * 2;
                data = src + col * Global::kColStride + row;
                using Loader = GlobalToRegMatLoader<Global, BaseTile,
                                                    tl::Layout::ColMajor>;
                Loader loader;
                loader(data, dst(j, i));
            }
        }
    }
};

/**
 * @brief Load a data tile from Global memory to Register based on the warp
 * reuse mode.
 * @tparam Global_ Global memory tile type.
 * @tparam Reg_ Register tile type.
 * @tparam WarpLayout_ Warp layout type.
 * @tparam kMode_ Warp reuse mode.
 * @tparam Base Copy base.
 */
template <typename Global_, typename Reg_, typename WarpLayout_,
          const WarpReuse kMode_,
          typename Base = warp::CopyBase<WarpLayout_, kMode_>>
struct GlobalToRegLoader : public Base {
    using Global = Global_;
    using Reg = Reg_;
    using DType = typename Global::DType;
    using BaseTile = typename Reg::DType;

    using WarpLayout = WarpLayout_;
    static constexpr WarpReuse kMode = kMode_;

    DEVICE void operator()(const Global& src, Reg& dst) {
        const DType* src_ptr = src.data();

        // 1. advance the pointer to input data to the current warp
        // according to warp reuse mode.
        src_ptr += Base::template get_warp_offset<Global>();

        // how many times a `BaseTile` is executed along the row and column
        // direction.
        static constexpr int kRowExec =
            Base::template row_exec_count<BaseTileShape<DType>,
                                          Global::kRows>();
        static constexpr int kColExec =
            Base::template col_exec_count<BaseTileShape<DType>,
                                          Global::kCols>();

        using Loader = GlobalToRegLoaderImpl<Global, Reg, kRowExec, kColExec,
                                             Global::type>;
        Loader loader;
        loader(src_ptr, dst);
    }
};

}  // namespace tiledcuda::cell::copy
