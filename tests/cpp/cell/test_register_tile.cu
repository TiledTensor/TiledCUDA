#include "common/test_utils.hpp"
#include "types/base_tile.hpp"
#include "types/mod.hpp"
#include "types/register_tile.hpp"

#include <glog/logging.h>

namespace tiledcuda {

using namespace cell;
namespace tl = tile_layout;

namespace {
template <typename RegTile>
__global__ void test_reg_tile() {
    RegTile tile;

    int count = 0;
    for (int i = 0; i < RegTile::kRows; ++i) {
        for (int j = 0; j < RegTile::kCols; ++j) {
            for (int m = 0; m < RegTile::DType::kNumelPerThread; ++m) {
                tile(i, j)(m) = ++count;
            }
        }
    }

    for (int i = 0; i < RegTile::kRows; ++i) {
        for (int j = 0; j < RegTile::kCols; ++j) {
            for (int m = 0; m < RegTile::DType::kNumelPerThread; ++m) {
                printf("%.0f, ", tile(i, j)(m));
            }
            printf("\n");
        }
    }
}

}  // namespace

namespace testing {

TEST(TestRegTile, test1) {
    using Element = float;
    using RegTile = RegisterTile<Element, tl::RowMajor<4, 4>>;

    test_reg_tile<RegTile><<<1, 1>>>();

    cudaDeviceSynchronize();
}

}  // namespace testing
}  // namespace tiledcuda
