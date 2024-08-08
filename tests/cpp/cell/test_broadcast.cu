#include "cell/compute/broadcast.hpp"
#include "cell/compute/reduce.hpp"
#include "cell/copy/constants.hpp"
#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"
#include "util/debug.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda::testing {
using namespace cell;

template <typename Element, typename RegLayout, typename GlobalLayout,
          typename ReduceLayout, typename BaseTile, typename WarpLayout,
          const tl::Layout kLayout, const copy::WarpReuse kMode,
          const int kHeight, const int kWidth>
__global__ void reg_broadcast(Element* src) {
    using SrcLoadTile = GlobalTile<Element, GlobalLayout>;
    using DstLoadTile = RegTile<BaseTile, RegLayout>;
    using SrcReduceTile = DstLoadTile;
    using DstReduceTile = RegTile<Element, tl::RowMajor<kHeight, 2>>;
    using SrcBroadcastTile = DstReduceTile;
    using DstBroadcastTile = SrcReduceTile;

    SrcLoadTile src_load_tile(src);
    DstLoadTile dst_load_tile;
    DstReduceTile dst_reduce_tile;
    DstBroadcastTile dst_broadcast_tile;

    // Load data from global memory to register file
    copy::GlobalToRegLoader<DstLoadTile, WarpLayout, kMode> loader;
    loader(src_load_tile, dst_load_tile);
    __syncthreads();

    // Execute reduce operation.
    compute::MaxReduce<SrcReduceTile, kLayout> row_max;
    row_max(dst_load_tile, dst_reduce_tile);

    __syncthreads();

    compute::Broadcast<SrcBroadcastTile, DstBroadcastTile, kLayout>
        broadcast_reduce;

    broadcast_reduce(dst_reduce_tile, dst_broadcast_tile);

    __syncthreads();

    if (thread(0)) {
        printf("Row Max:\n");
        printf("Thread 0:\n");
        dst_broadcast_tile.dump_value();
    }
}

template <typename Element, typename RegLayout, typename GlobalLayout,
          typename BaseTile, typename WarpLayout, const tl::Layout kLayout,
          const copy::WarpReuse kMode, const int kHeight, const int kWidth>
void run_row_major_reg_broadcast() {
    int kNumel = 16 * 16 * kHeight * kWidth;
    int kWarpSize = tl::get_numel<WarpLayout>;

    using ReduceLayout = tl::RowMajor<kHeight, 2>;

    thrust::host_vector<Element> h_src(kNumel);
    for (int i = 0; i < kNumel; ++i) {
        h_src[i] = (Element)i;
    }

    thrust::device_vector<Element> d_src = h_src;

    reg_broadcast<Element, RegLayout, GlobalLayout, ReduceLayout, BaseTile,
                  WarpLayout, kLayout, kMode, kHeight, kWidth>
        <<<1, 32 * kWarpSize>>>(thrust::raw_pointer_cast(d_src.data()));
}

TEST(TestRegBroadcast, row_major_reg_broadcast) {
    const int kHeight = 1;
    const int kWidth = 1;
    using Element = float;
    using WarpLayout = tl::RowMajor<1, 1>;
    using RegLayout = tl::RowMajor<kHeight, kWidth>;

    const copy::WarpReuse kMode = copy::WarpReuse::kCont;

    using GlobalLayout = tl::RowMajor<16 * kHeight, 16 * kWidth>;

    run_row_major_reg_broadcast<
        Element, RegLayout, GlobalLayout, BaseTileRowMajor<Element>, WarpLayout,
        tl::Layout::kRowMajor, kMode, kHeight, kWidth>();
}

}  // namespace tiledcuda::testing
