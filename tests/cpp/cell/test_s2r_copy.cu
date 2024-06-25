#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <glog/logging.h>

namespace tiledcuda {

using namespace cell;
using namespace copy;
namespace tl = tile_layout;

namespace {

template <typename Element>
__device__ void init_value(Element* data, int numel) {
    for (int i = 0; i < numel; ++i) data[i] = static_cast<Element>(i % 2048);
}

template <typename Element, typename Shared, typename Reg, typename Copy>
__global__ void run_test(Copy& copy) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init_value(buf, Shared::kNumel);

    Shared s_tile(buf);
    Reg r_tile;

    copy(s_tile, r_tile);
}

}  // namespace

namespace testing {

TEST(TestShared2Reg, operand_A) {
    // the load mode for loading operand B in gemm
    using Element = cutlass::half_t;

    using WarpLayout = tl::RowMajor<2, 2>;
    const int kThreads = tl::get_numel<WarpLayout> * 32;

    using Shared = SharedTile<Element, tl::RowMajor<64, 32>>;
    using Reg = RegTile<Element, tl::RowMajor<4, 8>>;
    using Copy = SharedToRegLoader<Reg, WarpLayout, WarpReuse::RowReuseCont,
                                   CopyInst::LoadMat>;
    Copy copy;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = Shared::kNumel * sizeof(Element);

    run_test<Element, Shared, Reg, Copy>
        <<<dim_grid, dim_block, shm_size>>>(copy);
    cudaDeviceSynchronize();
}

TEST(TestShared2Reg, operand_B) {
    // the load mode for loading operand B in gemm
    using Element = cutlass::half_t;

    using WarpLayout = tl::RowMajor<2, 2>;
    const int kThreads = tl::get_numel<WarpLayout> * 32;

    // a 32x64 row-major shared tile is equivalent to a 64x32 col-major tile
    using Shared = SharedTile<Element, tl::RowMajor<32, 64>>;
    using Reg = RegTile<Element, tl::RowMajor<4, 8>>;

    using Copy = SharedToRegLoader<Reg, WarpLayout, WarpReuse::ColReuseCont,
                                   CopyInst::LoadMat>;
    Copy copy;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = Shared::kNumel * sizeof(Element);

    run_test<Element, Shared, Reg, Copy>
        <<<dim_grid, dim_block, shm_size>>>(copy);
    cudaDeviceSynchronize();
}

}  // namespace testing
}  // namespace tiledcuda
