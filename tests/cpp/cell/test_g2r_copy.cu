#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda {

using namespace cell;

template <typename Element, size_t height, size_t width>
__global__ void copy_g2r(const Element* src, const int row_stride) {
    RegTile<RegTile<Element, tl::RowMajor<2, 4>>, tl::RowMajor<height, width>>
        dst;
    cell::copy::copy_2d_tile_g2r<Element, height, width>(src, dst, row_stride);
    __syncthreads();
}

namespace testing {
TEST(TestG2RegCopy, copy_2d_tile_g2r) {
    using Element = float;
    const size_t height = 1;
    const size_t width = 1;

    int numel = height * width * 16 * 16;
    thrust::host_vector<Element> h_src(numel);
    for (int i = 0; i < numel; ++i) {
        h_src[i] = (float)i;
    }

    thrust::device_vector<Element> d_src = h_src;

    int row_stride = width * 16;

    copy_g2r<Element, height, width><<<1, 32>>>(d_src.data().get(), row_stride);
}
}  // namespace testing

}  // namespace tiledcuda
