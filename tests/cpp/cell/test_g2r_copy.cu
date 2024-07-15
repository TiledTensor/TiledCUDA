#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda {

using namespace cell;

template <typename Element, tl::Layout type, size_t height, size_t width>
__global__ void copy_g2r(const Element* src, const int row_stride) {
    using DstTile = RegTile<RegTile<Element, tl::RowMajor<2, 4>>,
                            tl::RowMajor<height, width>>;
    DstTile dst;

    cell::copy::GlobalToRegLoader<Element, DstTile, type> loader;
    loader(src, dst, row_stride);
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst(i, j).dump_value();
            }
        }
    }
}

namespace testing {
TEST(TestG2RegCopy, copy_2d_tile_g2r) {
    using Element = float;

    const int height = 1;
    const int width = 1;

    int numel = height * width * 16 * 16;
    thrust::host_vector<Element> h_src(numel);
    for (int i = 0; i < numel; ++i) {
        h_src[i] = (float)i;
    }

    thrust::device_vector<Element> d_src = h_src;

    int row_stride = width * 16;

    copy_g2r<Element, tl::Layout::RowMajor, height, width>
        <<<1, 32>>>(d_src.data().get(), row_stride);
}

TEST(TestG2RegCopy, copy_2d_tile_g2r_2) {
    using Element = float;

    const int height = 2;
    const int width = 2;

    int numel = height * width * 16 * 16;
    thrust::host_vector<Element> h_src(numel);
    for (int i = 0; i < numel; ++i) {
        h_src[i] = (float)i;
    }

    thrust::device_vector<Element> d_src = h_src;

    int row_stride = width * 16;

    copy_g2r<Element, tl::Layout::RowMajor, height, width>
        <<<1, 32>>>(d_src.data().get(), row_stride);
}
}  // namespace testing

}  // namespace tiledcuda
