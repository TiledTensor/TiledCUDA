#include "cell/compute/mod.hpp"
#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/types.hpp"

#include <glog/logging.h>
#include <thrust/host_vector.h>

namespace tiledcuda {

using namespace cell;
using namespace cute;
namespace tl = tile_layout;

namespace {

void ground_truth1(const float* a, const float* b, float* c, int M, int N,
                   int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.;
            for (int k = 0; k < K; ++k) s += a[i * K + k] * b[k * N + j];
            c[i * N + j] = s;
        }
    }

    printf("\n\nGround truth:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) printf("%.2f, ", float(c[i * N + j]));
        printf("\n");
    }
}

/// utility function
template <typename Element, const int kRows, const int kCols>
__device__ void init_a(Element* data) {
    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    int idx;
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kCols; ++j) {
            idx = i * kCols + j;
            data[idx] = static_cast<Element>(idx % 2048 / 100.);
        }
    }
}

template <typename Element, const int kRows, const int kCols>
__device__ void init_b(Element* data) {
    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    int idx;
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kCols; ++j) {
            idx = i * kCols + j;
            data[idx] = static_cast<Element>((j * kRows + i) % 2048 / 100.);
        }
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

template <typename Element, typename SharedA, typename SharedB, typename RegA,
          typename RegB, typename RegC>
__global__ void test_wmma1() {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    init_a<Element, SharedA::kRows, SharedA::kCols>(buf);
    init_b<Element, SharedB::kRows, SharedB::kCols>(buf + SharedA::kNumel);

#ifdef DEBUG
    print_tile<Element, SharedA::kRows, SharedA::kCols>(buf);
    print_tile<Element, SharedB::kRows, SharedB::kCols>(buf + SharedA::kNumel);
#endif

    SharedA sA(buf);
    SharedB sB(buf + SharedA::kNumel);
    RegA rA;
    RegB rB;
    RegC rC;

    copy::copy_2d_tile_s2r(sA[make_int2(0, 0)], rA);
    copy::copy_2d_tile_s2r(sB[make_int2(0, 0)], rB);
    compute::wmma(rA, rB, rC);

    copy::copy_2d_tile_r2s(rC, sA[make_int2(0, 0)]);
}

}  // namespace

namespace testing {

TEST(TestWmma, shape1) {
    using Element = cutlass::half_t;

    // operand A: shared memory
    using TemporalExecShared = TileShape<1, 1>;
    using WarpLayout = tl::RowMajor<1, 1>;
    using ThreadLayout = tl::RowMajor<16, 2>;  // fixed when using ldmatrix.
    using ElemDataTileShared = tl::RowMajor<1, 8>;
    using Shared = SharedTile<Element, TemporalExecShared, WarpLayout,
                              ThreadLayout, ElemDataTileShared>;

    // operand A: register
    using TemporalExecReg = TileShape<1, 1>;
    using ElemDataTileReg = tl::RowMajor<1, 8>;  // fixed when using ldmatrix
    using Reg = RegTile<Element, TemporalExecReg, ElemDataTileReg>;

    // operand C: register
    using TemporalExecReg = TileShape<1, 1>;
    using ElemDataTileReg = tl::RowMajor<1, 8>;
    using RegC = RegTile<float, TemporalExecReg, ElemDataTileReg>;

    const int kThreads = Shared::kThreads;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);
    int shm_size = (Shared::kNumel + Shared::kNumel) * sizeof(Element);
    test_wmma1<Element, Shared, Shared, Reg, Reg, RegC>
        <<<dim_grid, dim_block, shm_size>>>();
    cudaDeviceSynchronize();

    {  // ground truth
        int M = 16, N = 16, K = 16;

        thrust::host_vector<float> h_a(M * K);
        thrust::host_vector<float> h_b(K * N);
        thrust::host_vector<float> h_c(M * N);

        for (int i = 0; i < h_a.size(); ++i) h_a[i] = i % 2048 / 100.;
        // printf("\n\nB:\n");
        for (int i = 0; i < h_b.size(); ++i) {
            h_b[i] = i % 2048 / 100.;

            // printf("%.2f, ", h_b[i]);
            // if ((i + 1) % N == 0) printf("\n");
        }
        thrust::fill(h_c.begin(), h_c.end(), 0.);

        ground_truth1(thrust::raw_pointer_cast(h_a.data()),
                      thrust::raw_pointer_cast(h_b.data()),
                      thrust::raw_pointer_cast(h_c.data()), M, N, K);
    }
}

}  // namespace testing
}  // namespace tiledcuda
