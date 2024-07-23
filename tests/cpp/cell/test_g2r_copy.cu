#include "cell/copy/constants.hpp"
#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"
#include "util/debug.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda::testing {
using namespace cell;

namespace {
template <typename Element, tl::Layout kType, typename WarpLayout,
          typename RegLayout, typename GlobalLayout, typename BaseTile,
          const copy::WarpReuse kMode, const size_t kHeight, size_t kWidth>
__global__ void load_g2r(Element* src) {
    using SrcTile = GlobalTile<Element, GlobalLayout>;
    using DstTile = RegTile<BaseTile, RegLayout>;
    SrcTile src_tile(src);
    DstTile dst_tile;

    copy::GlobalToRegLoader<SrcTile, DstTile, WarpLayout, kMode> loader;
    loader(src_tile, dst_tile);
    __syncthreads();

    if (thread0()) {
        printf("thread 0:\n");
        dst_tile.dump_value();
    }

    if (thread(1)) {
        printf("thread 1:\n");
        dst_tile.dump_value();
    }

    __syncthreads();

    if (thread(32)) {
        printf("thread 32:\n");
        dst_tile.dump_value();
    }

    __syncthreads();

    if (thread(64)) {
        printf("thread 64:\n");
        dst_tile.dump_value();
    }

    __syncthreads();

    if (thread(96)) {
        printf("thread 96:\n");
        dst_tile.dump_value();
    }
}

template <typename Element, tl::Layout kType, typename WarpLayout,
          typename RegLayout, typename GlobalLayout, typename BaseTile,
          const size_t kHeight, size_t kWidth>
__global__ void store_r2g(Element* dst) {
    using SrcLayout = RegTile<BaseTile, RegLayout>;
    using DstLayout = GlobalTile<Element, GlobalLayout>;

    SrcLayout src_tile;
    DstLayout dst_tile(dst);

    int lane_id = threadIdx.x % 32;

    switch (kType) {
        case tl::Layout::kRowMajor:
            // row major
            for (int i = 0; i < kHeight; ++i) {
                int row = i * 16 + lane_id / 4;
                for (int j = 0; j < kWidth; ++j) {
                    int col = j * 16 + (lane_id % 4) * 2;
                    src_tile(i, j)(0, 0) = row * DstLayout::kRowStride + col;
                    src_tile(i, j)(0, 1) =
                        row * DstLayout::kRowStride + col + 1;
                    src_tile(i, j)(1, 0) =
                        row * DstLayout::kRowStride + col + 8;
                    src_tile(i, j)(1, 1) =
                        row * DstLayout::kRowStride + col + 9;
                    src_tile(i, j)(0, 2) =
                        (row + 8) * DstLayout::kRowStride + col;
                    src_tile(i, j)(0, 3) =
                        (row + 8) * DstLayout::kRowStride + col + 1;
                    src_tile(i, j)(1, 2) =
                        (row + 8) * DstLayout::kRowStride + col + 8;
                    src_tile(i, j)(1, 3) =
                        (row + 8) * DstLayout::kRowStride + col + 9;
                }
            }
            break;
        case tl::Layout::kColMajor:
            // col major
            for (int i = 0; i < kWidth; ++i) {
                int col = i * 16 + lane_id / 4;
                for (int j = 0; j < kHeight; ++j) {
                    int row = j * 16 + (lane_id % 4) * 2;
                    src_tile(j, i)(0, 0) = col * DstLayout::kColStride + row;
                    src_tile(j, i)(1, 0) =
                        col * DstLayout::kColStride + row + 1;
                    src_tile(j, i)(0, 1) =
                        col * DstLayout::kColStride + row + 8;
                    src_tile(j, i)(1, 1) =
                        col * DstLayout::kColStride + row + 9;
                    src_tile(j, i)(2, 0) =
                        (col + 8) * DstLayout::kColStride + row;
                    src_tile(j, i)(3, 0) =
                        (col + 8) * DstLayout::kColStride + row + 1;
                    src_tile(j, i)(2, 1) =
                        (col + 8) * DstLayout::kColStride + row + 8;
                    src_tile(j, i)(3, 1) =
                        (col + 8) * DstLayout::kColStride + row + 9;
                }
            }
            break;
        default:
            break;
    }

    copy::RegToGlobalStorer<DstLayout, SrcLayout, WarpLayout> storer;
    storer(src_tile, dst_tile);

    __syncthreads();

    if (thread(0)) {
        dst_tile.dump_value();
    }
}

template <typename Element, tl::Layout kType, typename WarpLayout,
          typename RegLayout, typename GlobalLayout, typename BaseTile,
          const copy::WarpReuse kMode, const size_t kHeight,
          const size_t kWidth, const size_t kWarpSize>
void run_load_g2r_test() {
    int kNumel = 16 * 16 * kHeight * kWidth;
    thrust::host_vector<Element> h_src(kNumel);
    for (int i = 0; i < kNumel; ++i) {
        h_src[i] = (Element)i;
    }

    thrust::device_vector<Element> d_src = h_src;

    load_g2r<Element, kType, WarpLayout, RegLayout, GlobalLayout, BaseTile,
             kMode, kHeight, kWidth><<<1, 32 * kWarpSize>>>(d_src.data().get());
}

template <typename Element, tl::Layout kType, typename WarpLayout,
          typename RegLayout, typename GlobalLayout, typename BaseTile,
          const size_t kHeight, const size_t kWidth, const size_t kWarpSize>
void run_store_r2g_test() {
    int kNumel = 16 * 16 * kHeight * kWidth;
    thrust::host_vector<Element> h_dst(kNumel, 0);

    thrust::device_vector<Element> d_dst = h_dst;

    store_r2g<Element, kType, WarpLayout, RegLayout, GlobalLayout, BaseTile,
              kHeight, kWidth><<<1, 32 * kWarpSize>>>(d_dst.data().get());
}
}  // namespace

TEST(TestG2RegCopy, copy_2d_tile_g2r_row_major_0) {
    using Element = float;
    using WarpLayout = tl::RowMajor<1, 1>;
    using RegLayout = tl::RowMajor<1, 1>;

    const int kHeight = 1;
    const int kWidth = 1;
    const int kWarpSize = 1;
    const copy::WarpReuse kMode = copy::WarpReuse::kCont;

    using GlobalLayout = tl::RowMajor<16 * kHeight, 16 * kWidth>;

    run_load_g2r_test<Element, tl::Layout::kRowMajor, WarpLayout, RegLayout,
                      GlobalLayout, BaseTileRowMajor<Element>, kMode, kHeight,
                      kWidth, kWarpSize>();
}

TEST(TestG2RegCopy, load_2d_tile_g2r_row_major_1) {
    using Element = float;
    using WarpLayout = tl::RowMajor<1, 1>;
    using RegLayout = tl::RowMajor<2, 2>;

    const int kHeight = 2;
    const int kWidth = 2;
    const int kWarpSize = 1;
    const copy::WarpReuse kMode = copy::WarpReuse::kCont;

    using GlobalLayout = tl::RowMajor<16 * kHeight, 16 * kWidth>;

    run_load_g2r_test<Element, tl::Layout::kRowMajor, WarpLayout, RegLayout,
                      GlobalLayout, BaseTileRowMajor<Element>, kMode, kHeight,
                      kWidth, kWarpSize>();
}

TEST(TestG2RegCopy, load_2d_tile_g2r_row_major_2) {
    using Element = float;
    using WarpLayout = tl::RowMajor<2, 2>;
    using RegLayout = tl::RowMajor<1, 2>;

    const int kHeight = 2;
    const int kWidth = 2;
    const int kWarpSize = 4;
    const copy::WarpReuse kMode = copy::WarpReuse::kRowReuseCont;

    using GlobalLayout = tl::RowMajor<16 * kHeight, 16 * kWidth>;

    run_load_g2r_test<Element, tl::Layout::kRowMajor, WarpLayout, RegLayout,
                      GlobalLayout, BaseTileRowMajor<Element>, kMode, kHeight,
                      kWidth, kWarpSize>();
}

TEST(TestG2RegCopy, load_2d_tile_g2r_row_major_3) {
    using Element = float;
    using WarpLayout = tl::RowMajor<2, 2>;
    using RegLayout = tl::RowMajor<2, 1>;

    const int kHeight = 2;
    const int kWidth = 2;
    const int kWarpSize = 4;
    const copy::WarpReuse kMode = copy::WarpReuse::kColReuseCont;

    using GlobalLayout = tl::RowMajor<16 * kHeight, 16 * kWidth>;

    run_load_g2r_test<Element, tl::Layout::kRowMajor, WarpLayout, RegLayout,
                      GlobalLayout, BaseTileRowMajor<Element>, kMode, kHeight,
                      kWidth, kWarpSize>();
}

TEST(TestG2RegCopy, load_2d_tile_g2r_col_major_0) {
    using Element = float;
    using WarpLayout = tl::RowMajor<1, 1>;
    using RegLayout = tl::ColMajor<1, 1>;

    const int kHeight = 1;
    const int kWidth = 1;
    const int kWarpSize = 1;
    const copy::WarpReuse kMode = copy::WarpReuse::kCont;

    using GlobalLayout = tl::ColMajor<16 * kWidth, 16 * kHeight>;

    run_load_g2r_test<Element, tl::Layout::kColMajor, WarpLayout, RegLayout,
                      GlobalLayout, BaseTileColMajor<Element>, kMode, kHeight,
                      kWidth, kWarpSize>();
}

TEST(TestG2RegCopy, load_2d_tile_g2r_col_major_1) {
    using Element = float;
    using WarpLayout = tl::RowMajor<1, 1>;
    using RegLayout = tl::ColMajor<2, 2>;

    const int kHeight = 2;
    const int kWidth = 2;
    const int kWarpSize = 1;
    const copy::WarpReuse kMode = copy::WarpReuse::kCont;

    using GlobalLayout = tl::ColMajor<16 * kWidth, 16 * kHeight>;

    run_load_g2r_test<Element, tl::Layout::kColMajor, WarpLayout, RegLayout,
                      GlobalLayout, BaseTileColMajor<Element>, kMode, kHeight,
                      kWidth, kWarpSize>();
}

TEST(TestG2RegCopy, load_2d_tile_g2r_col_major_2) {
    using Element = float;
    using WarpLayout = tl::RowMajor<2, 2>;
    using RegLayout = tl::ColMajor<1, 2>;

    const int kHeight = 2;
    const int kWidth = 2;
    const int kWarpSize = 4;
    const copy::WarpReuse kMode = copy::WarpReuse::kRowReuseCont;

    using GlobalLayout = tl::ColMajor<16 * kWidth, 16 * kHeight>;

    run_load_g2r_test<Element, tl::Layout::kColMajor, WarpLayout, RegLayout,
                      GlobalLayout, BaseTileColMajor<Element>, kMode, kHeight,
                      kWidth, kWarpSize>();
}

TEST(TestG2RegCopy, load_2d_tile_g2r_col_major_3) {
    using Element = float;
    using WarpLayout = tl::RowMajor<2, 2>;
    using RegLayout = tl::ColMajor<2, 1>;

    const int kHeight = 2;
    const int kWidth = 2;
    const int kWarpSize = 4;
    const copy::WarpReuse kMode = copy::WarpReuse::kColReuseCont;

    using GlobalLayout = tl::ColMajor<16 * kWidth, 16 * kHeight>;

    run_load_g2r_test<Element, tl::Layout::kColMajor, WarpLayout, RegLayout,
                      GlobalLayout, BaseTileColMajor<Element>, kMode, kHeight,
                      kWidth, kWarpSize>();
}

TEST(TestG2RegCopy, store_2d_tile_r2g_row_major) {
    using Element = float;
    using WarpLayout = tl::RowMajor<1, 1>;
    using RegLayout = tl::RowMajor<1, 1>;

    const int kHeight = 1;
    const int kWidth = 1;
    const int kWarpSize = 1;

    using GlobalLayout = tl::RowMajor<16 * kHeight, 16 * kWidth>;

    run_store_r2g_test<Element, tl::Layout::kRowMajor, WarpLayout, RegLayout,
                       GlobalLayout, BaseTileRowMajor<Element>, kHeight, kWidth,
                       kWarpSize>();
}

TEST(TestG2RegCopy, store_2d_tile_r2g_col_major) {
    using Element = float;
    using WarpLayout = tl::ColMajor<1, 1>;
    using RegLayout = tl::ColMajor<1, 1>;

    const int kHeight = 1;
    const int kWidth = 1;
    const int kWarpSize = 1;

    using GlobalLayout = tl::ColMajor<16 * kWidth, 16 * kHeight>;

    run_store_r2g_test<Element, tl::Layout::kColMajor, WarpLayout, RegLayout,
                       GlobalLayout, BaseTileColMajor<Element>, kHeight, kWidth,
                       kWarpSize>();
}

}  // namespace tiledcuda::testing
