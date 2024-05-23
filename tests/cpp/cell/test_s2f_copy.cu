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
template <typename Element, const int kNumel>
__device__ void init(Element* data) {
    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    for (int i = 0; i < kNumel; ++i) data[i] = static_cast<Element>(i % 2048);
}

/// utility function
template <typename Element, const int kRows, const int kCols>
__device__ void print_tile(const Element* data) {
    const int delimeter = 64;

    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    for (int i = 0; i < kRows * kCols; ++i) {
        printf("%.0f, ", float(data[i] * 1.0_hf));  // print cutlass::half_t
        if ((i + 1) % delimeter == 0) printf("\n");
    }
    printf("\n");
}

#define DEBUG_
/// unittest for copy_s2r
template <typename Element, typename Shared, typename Reg>
__global__ void copy_s2r() {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init<Element, Shared::kRows * Shared::kCols>(buf);

#ifdef DEBUG
    print_tile<Element, Shared::kRows, Shared::kCols>(buf);
#endif

    Shared s_tiles(buf);
    Reg r_tile;
    for (auto i = 0; i < Shared::sc0; ++i) {
        for (auto j = 0; j < Shared::sc1; ++j) {
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

    // configuration for shared memory tile
    using TemporalExecShared = TileShape<2, 2>;
    using WarpLayout = TileShape<1, 2>;
    using ThreadLayout = TileShape<16, 2>;  // fixed when using ldmatrix.
    using ElemDataTileShared = TileShape<2, 16>;
    // the final copy plan for accessing shared memory
    using Shared = SharedTile<Element, TemporalExecShared, WarpLayout,
                              ThreadLayout, ElemDataTileShared>;

    // configuration for register tile
    using TemporalExecReg = TileShape<2, 2>;
    using ElemDataTileReg = TileShape<1, 8>;  // fixed when using ldmatrix
    // the final copy plan for accessing local register file
    using Reg = RegTile<Element, TemporalExecReg, ElemDataTileReg>;

    const int kThreads = Shared::kThreads;
    LOG(INFO) << "kThreads: " << kThreads << std::endl
              << "Shared memory tile shape: [" << Shared::kRows << ", "
              << Shared::kCols << "];" << std::endl
              << "Spatial tile shape: [" << Shared::kRowsPerExec << ", "
              << Shared::kColsPerExec << "]; " << std::endl;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);

    int shm_size = Shared::kNumel * sizeof(Element);
    copy_s2r<Element, Shared, Reg><<<dim_grid, dim_block, shm_size>>>();

    cudaDeviceSynchronize();
}

}  // namespace testing
}  // namespace tiledcuda
