#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/types.hpp"

#include <glog/logging.h>

namespace tiledcuda {

using namespace cell;
using namespace cute;
namespace tl = tile_layout;

namespace testing {

namespace {
/// utility function
template <typename Element, const int rows, const int cols>
__device__ void init(Element* data) {
    if (threadIdx.x) return;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i * cols + j] = static_cast<Element>(i * cols + j);
        }
    }
}

/// utility function
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
template <typename Element, typename Shared, typename Reg>
__global__ void copy_s2r() {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init<Element, Shared::kRows, Shared::kCols>(buf);
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

TEST(TestShm2Rf, copy_2d_tile_s2r) {
    using Element = cutlass::half_t;

    // TODO(haruhi): Test swizzled row major layout for shared memory.
    // using Swizzled = tl::SwizzledRowMajor<Element, kRows, kCols, 0>;
    // using SrcLayout = typename Swizzled::SmemLayout;

    // The shared memory tile access is made up of elementary data tile
    // accesses, each of which is performed by a single thread. The following
    // configurations describe how the elementary data tiles combine to form a
    // shared memory tile data tile.
    using TemporalExecShared = TileShape<2, 2>;
    using WarpLayout = TileShape<1, 4>;  // how warps are laied out in a CTA
    // how threads are laid out in a single warp.
    using ThreadLayout = TileShape<16, 2>;  // fixed when using ldmatrix.
    // the shape of an elementary data tile accessed by a single thread.
    using ElemDataTile = TileShape<2, 16>;

    // configuration for register tile
    using TemporalExecReg = TileShape<2, 1>;
    // shape of the accessed data by executing the atomic instruction
    using ElemDataTileReg = TileShape<1, 8>;  // fixed when using ldmatrix

    // the final copy plan
    using Shared = SharedTile<Element, TemporalExecShared, WarpLayout,
                              ThreadLayout, ElemDataTile>;
    using Reg = RegTile<Element, TemporalExecReg, ElemDataTileReg>;

    const int kThreads = Shared::kThreads;
    LOG(INFO) << "kThreads: " << kThreads;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);

    int shm_size = Shared::kNumel * sizeof(Element);
    copy_s2r<Element, Shared, Reg><<<dim_grid, dim_block, shm_size>>>();

    cudaDeviceSynchronize();
}

}  // namespace testing
}  // namespace tiledcuda
