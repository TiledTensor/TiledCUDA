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

/**
 * @brief Reduce/Map operation for the flash attention.
 */
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
    DstReduceTile last_max_vec;
    DstReduceTile max_vec;
    DstReduceTile last_norm_vec;
    DstReduceTile norm_vec;
    DstBroadcastTile max_broadcast_tile;
    DstBroadcastTile norm_broadcast_tile;

    // Load data from global memory to register file
    copy::GlobalToRegLoader<DstLoadTile, WarpLayout, kMode> loader;
    loader(src_load_tile, attn_block);

    // Copy `max_vec` into `last_max_vec`
    copy::BaseTileCopy<DstReduceTile> copy_max_reg;
    copy_max_reg(max_vec, last_max_vec);
    // Copy `norm_vec` into `last_norm_vec`
    copy::BaseTileCopy<DstReduceTile> copy_norm_reg;
    copy_norm_reg(norm_vec, last_norm_vec);

    // Execute reduce operation.
    compute::MaxReduce<SrcReduceTile, kLayout> row_max;
    // accumulate onto the max_vec
    row_max(attn_block, max_vec);

    compute::Broadcast<SrcBroadcastTile, DstBroadcastTile, kLayout>
        broadcast_max;

    broadcast_max(max_vec, max_broadcast_tile);

    if (thread(0)) {
        printf("Thread 0:\n");
        max_vec.dump_value();
        max_broadcast_tile.dump_value();
        attn_block.dump_value();
    }

    // subtract max from attention -- now all <= 0.
    compute::RegTileSub<DstBroadcastTile> sub_row_max;
    sub_row_max(attn_block, max_broadcast_tile, attn_block);

    if (thread(0)) {
        printf("Thread 0:\n");
        attn_block.dump_value();
    }

    // exponentiate the block in-place.
    compute::RegTileExp<DstBroadcastTile> exp_attn;
    exp_attn(attn_block, attn_block);

    if (thread(0)) {
        printf("Thread 0:\n");
        attn_block.dump_value();
    }

    // subtract new max from old max to find the new normalization.
    compute::BaseTileSub<DstReduceTile> sub_new_max;
    sub_new_max(last_max_vec, max_vec, last_max_vec);

    // exponentiate this vector -- this is what we need to normalize by.
    compute::BaseTileExp<DstReduceTile> exp_max;
    exp_max(last_max_vec, last_max_vec);

    // and the norm vec is now normalized.
    compute::BaseTileMul<DstReduceTile> mul_norm;
    mul_norm(last_max_vec, norm_vec, norm_vec);

    // Accumulate the new attention block onto the now-rescaled norm-vec.
    // Reduce Sum + Add
    DstReduceTile sum_vec;
    compute::SumReduce<SrcReduceTile, kLayout> row_sum;
    row_sum(attn_block, sum_vec);
    compute::BaseTileAdd<DstReduceTile> add_sum;
    add_sum(sum_vec, norm_vec, norm_vec);

    // Now the attention block is correctly normalized.
    // Broadcast + Divide
    compute::Broadcast<SrcBroadcastTile, DstBroadcastTile, kLayout>
        broadcast_norm;
    broadcast_norm(norm_vec, norm_broadcast_tile);
    compute::RegTileDiv<DstBroadcastTile> div_norm;
    div_norm(attn_block, norm_broadcast_tile, attn_block);

    // Normalize the previous norm vec accorfing to the new max.
    compute::BaseTileMul<DstReduceTile> mul_norm_new;
    mul_norm_new(last_max_vec, last_norm_vec, last_norm_vec);

    // Normalize the previous norm vec according to the new norm.
    compute::BaseTileDiv<DstReduceTile> div_norm_new;
    div_norm_new(last_norm_vec, norm_vec, last_norm_vec);
}

template <typename Element, typename RegLayout, typename GlobalLayout,
          typename BaseTile, typename WarpLayout, const tl::Layout kLayout,
          const copy::WarpReuse kMode, const int kHeight, const int kWidth>
void run_row_major_reg_flash_attn() {
    int kNumel = 16 * 16 * kHeight * kWidth;
    int kWarpSize = tl::get_numel<WarpLayout>;

    using ReduceLayout = tl::RowMajor<kHeight, 2>;

    thrust::host_vector<Element> h_src(kNumel);
    for (int i = 0; i < kNumel; ++i) {
        h_src[i] = (Element)i;
    }

    thrust::device_vector<Element> d_src = h_src;

    flash_attn_reg_reduce<Element, RegLayout, GlobalLayout, ReduceLayout,
                          BaseTile, WarpLayout, kLayout, kMode, kHeight, kWidth>
        <<<1, 32 * kWarpSize>>>(thrust::raw_pointer_cast(d_src.data()));
}

TEST(TestRegBroadcast, row_major_reg_flash_attn_0) {
    const int kHeight = 1;
    const int kWidth = 1;
    using Element = float;
    using WarpLayout = tl::RowMajor<1, 1>;
    using RegLayout = tl::RowMajor<kHeight, kWidth>;

    const copy::WarpReuse kMode = copy::WarpReuse::kCont;

    using GlobalLayout = tl::RowMajor<16 * kHeight, 16 * kWidth>;

    run_row_major_reg_flash_attn<
        Element, RegLayout, GlobalLayout, BaseTileRowMajor<Element>, WarpLayout,
        tl::Layout::kRowMajor, kMode, kHeight, kWidth>();
}

}  // namespace tiledcuda::testing
