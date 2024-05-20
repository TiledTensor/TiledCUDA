#include "cell/copy/static_copy.hpp"
#include "common/test_utils.hpp"
#include "types/types.hpp"

#include <glog/logging.h>

namespace tiledcuda {

using namespace cell;
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
        for (int j = 0; j < cols; ++j)
            printf("%.0f, ",
                   __half2float(static_cast<float>(data[i * cols + j])));

        printf("\n");
    }
    printf("\n");
}

/// unittest for copy_s2r
template <typename Element, typename Shared, typename Reg, const int rows,
          const int cols>
__global__ void copy_s2r() {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init<Element, rows, cols>(buf);
    // print_tile<Element, rows, cols>(buf);

    Shared s_tiles(buf);
    Reg r_tile;
    for (auto i = 0; i < 2; ++i) {
        for (auto j = 0; j < 2; ++j) {
            copy::copy_2d_tile_s2r(s_tiles[make_int2(i, j)], r_tile);
        }
    }
}

}  // namespace

namespace testing {
TEST(TestShm2Rf, copy_2d_tile_s2r) {
    using Element = cutlass::half_t;

    // TODO(haruhi): Test swizzled row major layout for shared memory.
    // using Swizzled = tl::SwizzledRowMajor<Element, kRows, kCols, 0>;
    // using SrcLayout = typename Swizzled::SmemLayout;

    using TemporalExecShared = TileShape<2, 2>;
    using TemporalExecReg = TileShape<2, 1>;

    // ======== configurated by internal tunner =========
    // how warps are laied out in a CTA
    using WarpLayout = TileShape<1, 4>;
    // how threads are laid out in a single warp.
    // this configuration is fixed when using ldmatrix.
    using ThreadLayout = TileShape<16, 2>;
    // the shape of an elementary data file for a single thread.
    using ElemDataTile = TileShape<2, 16>;

    // for register tile
    using ElemDataTileReg = TileShape<1, 8>;

    using Shared = SharedTile<Element, TemporalExecShared, WarpLayout,
                              ThreadLayout, ElemDataTile>;
    using Reg = RegTile<Element, TemporalExecReg, ElemDataTileReg>;

    const int kRows = Shared::kRows;
    const int kCols = Shared::kCols;
    const int kThreads = Shared::kThreads;

    LOG(INFO) << "kRows: " << kRows << ", kCols: " << kCols
              << "; kThreads: " << kThreads;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);

    int shm_size = kRows * kCols * sizeof(Element);
    copy_s2r<Element, Shared, Reg, kRows, kCols>
        <<<dim_grid, dim_block, shm_size>>>();

    cudaDeviceSynchronize();
}

}  // namespace testing
}  // namespace tiledcuda
