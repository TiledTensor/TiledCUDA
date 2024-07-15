#include "cell/copy/mod.hpp"
#include "types/mod.hpp"

namespace tiledcuda {

using namespace cell;

template <typename Element, size_t height, size_t width>
__global__ void copy_g2r(const Element* src,
                         RegTile<RegTile<Element, tl::RowMajor<2, 4>>,
                                 tl::RowMajor<height, width>>& dst,
                         const int row_stride) {
    cell::copy::copy_2d_tile_g2r(src, dst, row_stride);
    __syncthreads();
}

}  // namespace tiledcuda