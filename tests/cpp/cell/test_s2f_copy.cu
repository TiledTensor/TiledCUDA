#include "cell/copy/static_copy.hpp"
#include "common/test_utils.hpp"
#include "types/types.hpp"

namespace tiledcuda {

using namespace cell::copy;
using namespace cute;

namespace tl = tile_layout;

namespace {
/// utility functions
template <typename Tile>
__device__ void init(Tile& tile) {
    if (threadIdx.x) return;

    for (int i = 0; i < Tile::kRows; ++i) {
        for (int j = 0; j < Tile::kCols; ++j) {
            int2 idx = make_int2(i, j);
            *tile[idx] =
                static_cast<typename Tile::Element>(i * Tile::kCols + j);
        }
    }
}

/// utility functions
template <typename Tile>
__device__ void print_tile(const Tile& tile) {
    if (threadIdx.x) return;

    for (int i = 0; i < Tile::kRows; ++i) {
        for (int j = 0; j < Tile::kCols; ++j) {
            int2 idx = make_int2(i, j);
            printf("%.0f, ", __half2float(static_cast<float>(*tile[idx])));
        }
        printf("\n");
    }
    printf("\n");
}

/// unittest for copy_s2r
template <typename Element, typename SrcLayout, typename DstLayout,
          typename ThreadLayout>
__global__ void copy_s2r(int rows, int cols) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);

    SharedTile<Element, SrcLayout> s_tile(buf);
    RegTile<Element, DstLayout> r_tile;

    init(s_tile);
    // print_tile(s_tile);

    for (int k = 0; k < 2; ++k) {
        // TODO(haruhi): the index operator returns the pointer to the
        // underlying data, r_tile[0] is a trick to get the pointer to the
        // first element.
        copy_2d_tile_s2r(buf, k, SrcLayout{}, r_tile[0], DstLayout{},
                         ThreadLayout{});
    }
}
}  // namespace

namespace testing {
TEST(TestShm2Rf, copy_2d_tile_s2r) {
    using Element = cutlass::half_t;

    const int warp_size = 32;

    // To fully utilize ldmatrix, a minimum tile shape of 16 x 16 (in half
    // precision) is required.
    const int kRows = 16;
    const int kCols = 32;

    // TODO(haruhi): Test swizzled row major layout for shared memory.
    // using Swizzled = tl::SwizzledRowMajor<Element, kRows, kCols, 0>;
    // using SrcLayout = typename Swizzled::SmemLayout;

    // the layout of a single shared memory tile accessed by 32 threads
    using SrcLayout = tl::RowMajor<kRows, kCols>;

    // During one execution of ldmatrix by 32 threads in a warp, they can read
    // 16 x 16 halfs from shared memory. At the destination, each thread's
    // register file stores a partition with a shape of 1 x 8 tile halfs.
    using DstLayout = tl::RowMajor<1, 8>;

    // to use ldmatrix, 32 threads forms a 16 x 2 tile,
    // each tile has a shape of 8 x 1
    using ThreadLayout = cute::Layout<Shape<_16, _2>, Stride<_16, _8>>;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(warp_size, 1, 1);

    int shm_size = kRows * kCols * sizeof(Element);

    copy_s2r<Element, SrcLayout, DstLayout, ThreadLayout>
        <<<dim_grid, dim_block, shm_size>>>(kRows, kCols);

    cudaDeviceSynchronize();
}

}  // namespace testing
}  // namespace tiledcuda
