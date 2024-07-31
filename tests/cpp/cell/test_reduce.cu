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
          typename BaseTile, typename WarpLayout, const tl::Layout kLayout,
          const copy::WarpReuse kMode, const int kHeight, const int kWidth>
__global__ void reg_reduce(Element* src) {
    using SrcLoadTile = GlobalTile<Element, GlobalLayout>;
    using DstLoadTile = RegTile<BaseTile, RegLayout>;
    using SrcReduceTile = DstLoadTile;
    using DstReduceTile = RegTile<Element, tl::RowMajor<kHeight, 2>>;

    SrcLoadTile src_load_tile(src);
    DstLoadTile dst_load_tile;
    DstReduceTile dst_reduce_tile;

    // Load data from global memory to register file
    copy::GlobalToRegLoader<DstLoadTile, WarpLayout, kMode> loader;
    loader(src_load_tile, dst_load_tile);
    __syncthreads();

    // Execute reduce operation.
    compute::SumReduce<SrcReduceTile, kLayout> row_sum;
    row_sum(dst_load_tile, dst_reduce_tile);

    __syncthreads();
}

}  // namespace tiledcuda::testing
