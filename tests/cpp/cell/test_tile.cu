#include "common/test_utils.hpp"
#include "types/register.hpp"
#include "types/shared.hpp"

namespace tiledcuda {
namespace testing {

using namespace cell;
using namespace cute;
namespace tl = tile_layout;

namespace {

/// utility function
template <typename Shared>
__device__ void init_tile(Shared& data) {
    if (threadIdx.x) return;

    int v = 0;
    for (int i = 0; i < Shared::kRows; ++i) {
        for (int j = 0; j < Shared::kCols; ++j) {
            data(i, j) = static_cast<typename Shared::DType>(++v % 2048);
        }
    }
}

template <typename Shared>
__global__ void test_shared_tile() {
    using DType = typename Shared::DType;

    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<typename Shared::DType*>(buf_);

    Shared s_tile(buf);
    init_tile(s_tile);

    if (threadIdx.x == 0) {
        printf("\nShared tile:\n");
        for (int i = 0; i < Shared::kRows; ++i) {
            for (int j = 0; j < Shared::kCols; ++j) {
                // print cutlass::half_t
                printf("%.0f, ", float(s_tile(i, j) * 1.0_hf));
            }
            printf("\n");
        }
    }

    using Reg = RegTile<DType, tl::RowMajor<2, 4>>;
    Reg r_tile;

    if (threadIdx.x == 0) {
        for (int i = 0; i < Reg::kRows; ++i) {
            for (int j = 0; j < Reg::kCols; ++j) {
                r_tile(i, j) = static_cast<DType>(i * Reg::kCols + j);
            }
        }

        printf("RegTile\n");
        for (int i = 0; i < Reg::kRows; ++i) {
            for (int j = 0; j < Reg::kCols; ++j) {
                // print cutlass::half_t
                printf("%.0f, ", float(r_tile(i, j) * 1.0_hf));
            }
            printf("\n");
        }
    }
}

}  // namespace

TEST(TestTile, test_shared_tile) {
    using Element = cutlass::half_t;

    const int rows = 16;
    const int cols = 8;

    using Shared1 = SharedTile<Element, tl::RowMajor<rows, cols>>;
    int shm_size = Shared1::kNumel * sizeof(Element);
    test_shared_tile<Shared1><<<1, 32, shm_size>>>();
    cudaDeviceSynchronize();
}

}  // namespace testing
}  // namespace tiledcuda
