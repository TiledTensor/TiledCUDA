#include "cell/copy/static_copy.hpp"
#include "common/test_utils.hpp"
#include "types/types.hpp"

namespace tiledcuda {

using namespace cell::copy;
using namespace cute;

namespace tl = tile_layout;

namespace {
/// utility functions
template <typename Element, const int rows, const int cols>
__device__ void init(Element* data) {
    if (threadIdx.x) return;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i * cols + j] = static_cast<Element>(i * cols + j);
        }
    }
}

/// utility functions
template <typename Element, const int rows, const int cols>
__device__ void print_tile(const Element* data) {
    if (threadIdx.x) return;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%.0f, ",
                   __half2float(static_cast<float>(data[i * cols + j])));
        }
        printf("\n");
    }
    printf("\n");
}

/// unittest for copy_s2r
template <typename Element, typename SrcLayout, typename DstLayout,
          typename ThreadLayout, const int rows, const int cols>
__global__ void copy_s2r() {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init<Element, rows, cols>(buf);
    // print_tile<Element, rows, cols>(buf);

    SharedTile<Element, SrcLayout, ThreadLayout> s_tile(buf);
    RegTile<Element, DstLayout> r_tile;  // register tile for a single thread
    for (int k = 0; k < s_tile.count(); ++k) {
        copy_2d_tile_s2r(s_tile[k], r_tile);
    }
}
}  // namespace

namespace testing {
TEST(TestShm2Rf, copy_2d_tile_s2r) {
    using Element = cutlass::half_t;

    const int warp_size = 32;

    // total data tile stored in shared memory.
    const int kRows = 32;
    const int kCols = 64;

    // TODO(haruhi): Test swizzled row major layout for shared memory.
    // using Swizzled = tl::SwizzledRowMajor<Element, kRows, kCols, 0>;
    // using SrcLayout = typename Swizzled::SmemLayout;

    // To fully utilize ldmatrix, a minimum tile shape of 16 x 16 (in half
    // precision) is required.

    // Since shared memory can be accessed by all threads in a CTA.
    // The layout of the shared memory tile defines the data tile accessed by
    // threads in a thread blocks.
    using SrcLayout = cute::Layout<Shape<Shape<_2, _16>, Shape<_2, _32>>,
                                   Stride<Stride<_1024, _64>, Stride<_32, _1>>>;
    // to use ldmatrix, 32 threads forms a 2 x 2 tile,
    // each tile has a shape of 8 x 1.
    using ThreadLayout =
        cute::Layout<Shape<Shape<_2, _8>, _2>, Stride<Stride<_16, _1>, _8>>;

    // TODO(haruhi): The layout of the destination tile defines how the data is
    // stored in a single thread's register file, which is DIFFERENT from the
    // source tile. Consider wheather this will be a problem.

    // During one execution of ldmatrix by 32 threads in a warp, they can read
    // 16 x 16 halfs from shared memory. At the destination, each thread's
    // register file stores a partition with a shape of 1 x 8 tile halfs.
    // executing ldmatrix 2 times results in a 1 x 16 tile in the register file.
    using DstLayout = cute::Layout<Shape<_1, _16>>;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(warp_size, 1, 1);

    int shm_size = kRows * kCols * sizeof(Element);
    copy_s2r<Element, SrcLayout, DstLayout, ThreadLayout, kRows, kCols>
        <<<dim_grid, dim_block, shm_size>>>();

    cudaDeviceSynchronize();
}

}  // namespace testing
}  // namespace tiledcuda
