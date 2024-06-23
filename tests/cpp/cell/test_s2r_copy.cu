#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <glog/logging.h>

namespace tiledcuda {

using namespace cell;
namespace tl = tile_layout;

namespace {

template <typename Element>
__device__ void init_value(Element* data, int numel) {
    for (int i = 0; i < numel; ++i) data[i] = static_cast<Element>(i);
}

template <typename Element, typename Shared, typename Reg, typename Copy>
__global__ void test1(Copy& copy) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init_value(buf, Shared::kNumel);

    Shared s_tile(buf);
    Reg r_tile;

    copy(s_tile, r_tile, copy::WarpReuse::RowCir);
}

}  // namespace

namespace testing {

TEST(TestShared2Reg, shape1) {
    using Element = cutlass::half_t;

    using WarpLayout = TileShape<2, 2>;
    const int kThreads = get_numel<WarpLayout> * 32;

    using Shared = SharedTile<Element, tl::RowMajor<32, 48>>;
    using Reg = RegTile<Element, tl::RowMajor<2, 12>>;
    using Copy = copy::detail::CopyShared2Reg<Shared, Reg, WarpLayout,
                                              copy::CopyInst::LoadMat>;
    Copy copy;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = Shared::kNumel * sizeof(Element);

    test1<Element, Shared, Reg, Copy><<<dim_grid, dim_block, shm_size>>>(copy);
    cudaDeviceSynchronize();
}

}  // namespace testing
}  // namespace tiledcuda
