#include "cell/copy/constants.hpp"
#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"
#include "util/debug.hpp"

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
          typename RegLayout, const copy::WarpReuse kMode, const size_t kHeight,
          const size_t kWidth, const size_t kWarpSize>
void run_load_g2r_test() {
    int kNumel = 16 * 16 * kHeight * kWidth;
    thrust::host_vector<Element> h_src(kNumel);
    for (int i = 0; i < kNumel; ++i) {
        h_src[i] = (Element)i;
    }

    thrust::device_vector<Element> d_src = h_src;

    copy_g2r_row_major<Element, kType, WarpLayout, RegLayout, kMode, kHeight,
                       kWidth><<<1, 32 * kWarpSize>>>(d_src.data().get());
}

namespace testing {
TEST(TestG2RegCopy, copy_2d_tile_g2r_row_major_0) {
    using Element = float;
    using WarpLayout = tl::RowMajor<1, 1>;
    using RegLayout = tl::RowMajor<1, 1>;

    const int kHeight = 1;
    const int kWidth = 1;
    const int kWarpSize = 1;
    const copy::WarpReuse kMode = copy::WarpReuse::RowReuseCont;

    run_load_g2r_test<Element, tl::Layout::RowMajor, WarpLayout, RegLayout,
                      kMode, kHeight, kWidth, kWarpSize>();
}

TEST(TestG2RegCopy, copy_2d_tile_g2r_row_major_1) {
    using Element = float;
    using WarpLayout = tl::RowMajor<1, 1>;
    using RegLayout = tl::RowMajor<2, 2>;

    const int kHeight = 2;
    const int kWidth = 2;
    const int kWarpSize = 1;
    const copy::WarpReuse kMode = copy::WarpReuse::RowReuseCont;

    run_load_g2r_test<Element, tl::Layout::RowMajor, WarpLayout, RegLayout,
                      kMode, kHeight, kWidth, kWarpSize>();
}

TEST(TestG2RegCopy, copy_2d_tile_g2r_row_major_2) {
    using Element = float;
    using WarpLayout = tl::RowMajor<2, 2>;
    using RegLayout = tl::RowMajor<1, 2>;

    const int kHeight = 2;
    const int kWidth = 2;
    const int kWarpSize = 4;
    const copy::WarpReuse kMode = copy::WarpReuse::RowReuseCont;

    run_load_g2r_test<Element, tl::Layout::RowMajor, WarpLayout, RegLayout,
                      kMode, kHeight, kWidth, kWarpSize>();
}

TEST(TestG2RegCopy, copy_2d_tile_g2r_row_major_3) {
    using Element = float;
    using WarpLayout = tl::RowMajor<2, 2>;
    using RegLayout = tl::RowMajor<2, 1>;

    const int kHeight = 2;
    const int kWidth = 2;
    const int kWarpSize = 4;
    const copy::WarpReuse kMode = copy::WarpReuse::ColReuseCont;

    run_load_g2r_test<Element, tl::Layout::RowMajor, WarpLayout, RegLayout,
                      kMode, kHeight, kWidth, kWarpSize>();
}

}  // namespace testing

}  // namespace tiledcuda
