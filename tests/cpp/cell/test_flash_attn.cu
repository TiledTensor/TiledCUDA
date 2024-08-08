#include "cell/compute/broadcast.hpp"
#include "cell/compute/map.hpp"
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
__global__ void flash_attn_reg_reduce(Element* src) {
    using SrcLoadTile = GlobalTile<Element, GlobalLayout>;
    using DstLoadTile = RegTile<BaseTile, RegLayout>;
    using SrcReduceTile = DstLoadTile;
    using DstReduceTile = RegTile<Element, tl::RowMajor<kHeight, 2>>;
    using SrcBroadcastTile = DstReduceTile;
    using DstBroadcastTile = SrcReduceTile;

    SrcLoadTile src_load_tile(src);
    DstLoadTile attn_block;
    DstReduceTile max_vec;
    DstBroadcastTile max_broadcast_tile;

    // Load data from global memory to register file
    copy::GlobalToRegLoader<DstLoadTile, WarpLayout, kMode> loader;
    loader(src_load_tile, attn_block);
    __syncthreads();

    // Execute reduce operation.
    compute::MaxReduce<SrcReduceTile, kLayout> row_max;
    // accumulate onto the max_vec
    row_max(attn_block, max_vec);

    __syncthreads();

    compute::Broadcast<SrcBroadcastTile, DstBroadcastTile, kLayout>
        broadcast_max;

    broadcast_max(max_vec, max_broadcast_tile);

    __syncthreads();

    compute::RegTileSub<DstBroadcastTile> sub_row_max;
    sub_row_max(attn_block, max_broadcast_tile, attn_block);
}

}  // namespace tiledcuda::testing