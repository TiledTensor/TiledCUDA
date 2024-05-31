#include "cell/compute/mod.hpp"
#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/types.hpp"

#include <glog/logging.h>

namespace tiledcuda {

using namespace cell;
using namespace cute;
namespace tl = tile_layout;

namespace {

/// utility function
template <typename Element, const int kNumel>
__device__ void init(Element* data) {
    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    for (int i = 0; i < kNumel; ++i) {
        data[i] = static_cast<Element>(i % 2048 / 100.);
    }
}

/// utility function
template <typename Element, const int kRows, const int kCols>
__device__ void print_tile(const Element* data, int delimeter = kCols) {
    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    for (int i = 0; i < kRows * kCols; ++i) {
        printf("%.2f, ", float(data[i] * 1.0_hf));  // print cutlass::half_t
        if ((i + 1) % delimeter == 0) printf("\n");
    }
    printf("\n");
}

template <typename Element, typename Shared, typename Reg>
__global__ void test_wmma1() {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init<Element, 2 * Shared::kRows * Shared::kCols>(buf);

    Shared sA(buf);
    Shared sB(buf + Shared::kNumel);
    Reg rA, rB, rC;

    copy::copy_2d_tile_s2r(sA[make_int2(0, 0)], rA);
    copy::copy_2d_tile_s2r(sB[make_int2(0, 0)], rB);
    compute::wmma(rA, rB, rC);

    if (threadIdx.x == 0) {
        const half* reg1 = reinterpret_cast<const half*>(rA.data());
        const half* reg2 = reinterpret_cast<const half*>(rB.data());
        const half* reg3 = reinterpret_cast<const half*>(rC.data());

        printf("[A-tid = %d]:\t", threadIdx.x);
        for (int i = 0; i < 8; ++i) printf("%.2f, ", __half2float(reg1[i]));
        printf("\n[B-tid = %d]:\t", threadIdx.x);
        for (int i = 0; i < 8; ++i) printf("%.2f, ", __half2float(reg2[i]));
        printf("\n[C-tid = %d]:\t", threadIdx.x);
        for (int i = 0; i < 8; ++i) printf("%.2f, ", __half2float(reg3[i]));
        printf("\n");
    }
}

}  // namespace

namespace testing {

TEST(TestWmma, shape1) {
    using Element = cutlass::half_t;

    using TemporalExecShared = TileShape<1, 1>;
    using WarpLayout = TileShape<1, 1>;
    using ThreadLayout = TileShape<16, 2>;  // fixed when using ldmatrix.
    using ElemDataTileShared = TileShape<1, 8>;
    using Shared = SharedTile<Element, TemporalExecShared, WarpLayout,
                              ThreadLayout, ElemDataTileShared>;

    using TemporalExecReg = TileShape<1, 1>;
    using ElemDataTileReg = TileShape<1, 8>;  // fixed when using ldmatrix
    using Reg = RegTile<Element, TemporalExecReg, ElemDataTileReg>;

    const int kThreads = Shared::kThreads;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = 2 * Shared::kNumel * sizeof(Element);
    test_wmma1<Element, Shared, Reg><<<dim_grid, dim_block, shm_size>>>();
    cudaDeviceSynchronize();
}

}  // namespace testing
}  // namespace tiledcuda
