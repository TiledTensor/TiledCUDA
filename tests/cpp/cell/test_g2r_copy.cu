#include "cell/copy/constants.hpp"
#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"
#include "util/debug.hpp"
#include "util/print.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda {

using namespace cell;

template <typename Element, tl::Layout type, typename WarpLayout,
          typename RegLayout, const copy::WarpReuse kMode, const size_t kHeight,
          size_t kWidth>
__global__ void copy_g2r_row_major(Element* src) {
    using SrcTile =
        GlobalTile<Element, tl::RowMajor<16 * kHeight, 16 * kWidth>>;
    using DstTile = RegTile<BaseTileRowMajor<Element>, RegLayout>;
    SrcTile src_tile(src);
    DstTile dst_tile;

    cell::copy::GlobalToRegLoader<SrcTile, DstTile, WarpLayout, kMode> loader;
    loader(src_tile, dst_tile);
    __syncthreads();

    if (thread0()) {
        printf("thread 0:\n");
        for (int i = 0; i < DstTile::kRows; ++i) {
            for (int j = 0; j < DstTile::kCols; ++j) {
                print_tile(dst_tile(i, j).data(), dst_tile(i, j).layout());
            }
        }
    }

    if (thread(1)) {
        printf("thread 1:\n");
        for (int i = 0; i < DstTile::kRows; ++i) {
            for (int j = 0; j < DstTile::kCols; ++j) {
                print_tile(dst_tile(i, j).data(), dst_tile(i, j).layout());
            }
        }
    }

    __syncthreads();

    if (thread(32)) {
        printf("thread 32:\n");
        for (int i = 0; i < DstTile::kRows; ++i) {
            for (int j = 0; j < DstTile::kCols; ++j) {
                print_tile(dst_tile(i, j).data(), dst_tile(i, j).layout());
            }
        }
    }
}

namespace testing {
TEST(TestG2RegCopy, copy_2d_tile_g2r_row_major_0) {
    using Element = float;
    using WarpLayout = tl::RowMajor<1, 1>;

    const int height = 1;
    const int width = 1;

    int numel = height * width * 16 * 16;
    thrust::host_vector<Element> h_src(numel);
    for (int i = 0; i < numel; ++i) {
        h_src[i] = (float)i;
    }

    thrust::device_vector<Element> d_src = h_src;

    copy_g2r_row_major<Element, tl::Layout::RowMajor, WarpLayout,
                       tl::RowMajor<1, 1>, copy::WarpReuse::RowReuseCont,
                       height, width><<<1, 32>>>(d_src.data().get());
}

TEST(TestG2RegCopy, copy_2d_tile_g2r_row_major_1) {
    using Element = float;
    using WarpLayout = tl::RowMajor<1, 1>;

    const int height = 2;
    const int width = 2;

    int numel = height * width * 16 * 16;
    thrust::host_vector<Element> h_src(numel);
    for (int i = 0; i < numel; ++i) {
        h_src[i] = (float)i;
    }

    thrust::device_vector<Element> d_src = h_src;

    copy_g2r_row_major<Element, tl::Layout::RowMajor, WarpLayout,
                       tl::RowMajor<2, 2>, copy::WarpReuse::RowReuseCont,
                       height, width><<<1, 32>>>(d_src.data().get());
}

TEST(TestG2RegCopy, copy_2d_tile_g2r_row_major_2) {
    using Element = float;
    using WarpLayout = tl::RowMajor<2, 2>;

    const int height = 2;
    const int width = 2;

    int numel = height * width * 16 * 16;
    thrust::host_vector<Element> h_src(numel);
    for (int i = 0; i < numel; ++i) {
        h_src[i] = (float)i;
    }

    thrust::device_vector<Element> d_src = h_src;

    copy_g2r_row_major<Element, tl::Layout::RowMajor, WarpLayout,
                       tl::RowMajor<1, 1>, copy::WarpReuse::RowReuseCont,
                       height, width><<<1, 32 * 4>>>(d_src.data().get());
}

}  // namespace testing

}  // namespace tiledcuda
