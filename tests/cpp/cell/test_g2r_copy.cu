#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda {

using namespace cell;

template <typename Element, tl::Layout type, size_t height, size_t width>
__global__ void copy_g2r_row_major(Element* src) {
    using SrcTile = GlobalTile<Element, tl::RowMajor<16 * height, 16 * width>>;
    using DstTile = RegTile<RegTile<Element, tl::RowMajor<2, 4>>,
                            tl::RowMajor<height, width>>;
    SrcTile src_tile(src);
    DstTile dst_tile;

    cell::copy::GlobalToRegLoader<SrcTile, DstTile, type> loader;
    loader(src_tile, dst_tile);
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst_tile(i, j).dump_value();
            }
        }
    }
}

template <typename Element, tl::Layout type, size_t height, size_t width>
__global__ void copy_g2r_col_major(Element* src) {
    using SrcTile = GlobalTile<Element, tl::ColMajor<16 * width, 16 * height>>;
    using DstTile = RegTile<RegTile<Element, tl::ColMajor<4, 2>>,
                            tl::ColMajor<width, height>>;
    SrcTile src_tile(src);
    DstTile dst_tile;

    cell::copy::GlobalToRegLoader<SrcTile, DstTile, type> loader;
    loader(src_tile, dst_tile);
    __syncthreads();

    if (threadIdx.x == 0) {
        // printf("src(0, 0) = %f\n", src_tile(0, 0));
        // printf("src(0, 1) = %f\n", src_tile(0, 1));
        // printf("src(1, 0) = %f\n", src_tile(1, 0));
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                dst_tile(i, j).dump_value();
            }
        }
    }
}

namespace testing {
TEST(TestG2RegCopy, copy_2d_tile_g2r_row_major_0) {
    using Element = float;

    const int height = 1;
    const int width = 1;

    int numel = height * width * 16 * 16;
    thrust::host_vector<Element> h_src(numel);
    for (int i = 0; i < numel; ++i) {
        h_src[i] = (float)i;
    }

    thrust::device_vector<Element> d_src = h_src;

    copy_g2r_row_major<Element, tl::Layout::RowMajor, height, width>
        <<<1, 32>>>(d_src.data().get());
}

TEST(TestG2RegCopy, copy_2d_tile_g2r_row_major_1) {
    using Element = float;

    const int height = 2;
    const int width = 2;

    int numel = height * width * 16 * 16;
    thrust::host_vector<Element> h_src(numel);
    for (int i = 0; i < numel; ++i) {
        h_src[i] = (float)i;
    }

    thrust::device_vector<Element> d_src = h_src;

    copy_g2r_row_major<Element, tl::Layout::RowMajor, height, width>
        <<<1, 32>>>(d_src.data().get());
}

TEST(TestG2RegCopy, copy_2d_tile_g2r_col_major_0) {
    using Element = float;
    const int height = 1;
    const int width = 1;
    int numel = height * width * 16 * 16;

    thrust::host_vector<Element> h_src(numel);
    // Assign values in column major order.
    // for (int i = 0; i < 16; ++i) {
    //     for (int j = 0; j < 16; ++j) {
    //         h_src[j * 16 + i] = (float)(j * 16 + i);
    //     }
    // }
    for (int i = 0; i < numel; ++i) {
        h_src[i] = (float)i;
    }

    copy_g2r_col_major<Element, tl::Layout::ColMajor, height, width>
        <<<1, 32>>>(thrust::device_vector<Element>(h_src).data().get());
}

}  // namespace testing

}  // namespace tiledcuda
