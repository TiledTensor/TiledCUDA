#include "cell/compute/mod.hpp"
#include "cell/copy/mod.hpp"
#include "common/test_utils.hpp"
#include "types/types.hpp"

#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace tiledcuda {

using namespace cell;
using namespace cute;
namespace tl = tile_layout;

namespace {

void check_result1(const float* a, const float* b, float* c, int M, int N,
                   int K, const float* result) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.;
            for (int k = 0; k < K; ++k) s += a[i * K + k] * b[k * N + j];
            c[i * N + j] = s;
        }
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            EXPECT_NEAR(c[i * N + j], result[i * N + j], 1e-2);
        }
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

template <typename Element, typename SharedA, typename RegA, typename SharedB,
          typename RegB, typename ElementC, typename SharedC, typename RegC,
          typename CopyTraits>
__global__ void test_wmma1(ElementC* gC) {
    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    auto* st_buf = reinterpret_cast<ElementC*>(buf);

    init_a<Element, SharedA::kRows, SharedA::kCols>(buf);
    init_b<Element, SharedB::kRows, SharedB::kCols>(buf + SharedA::kNumel);

#ifdef DEBUG
    print_tile<Element, SharedA::kRows, SharedA::kCols>(buf);
    print_tile<Element, SharedB::kRows, SharedB::kCols>(buf + SharedA::kNumel);
#endif

    // declare shared memory tile
    SharedA sA(buf);
    SharedB sB(buf + SharedA::kNumel);
    SharedC sC(st_buf);

    // declare register tile
    RegA rA;
    RegB rB;
    RegC rC;

    copy::copy_2d_tile_s2r(sA[make_int2(0, 0)], rA);
    copy::copy_2d_tile_s2r(sB[make_int2(0, 0)], rB);
    compute::wmma(rA, rB, rC);
    copy::copy_2d_tile_r2s(rC, sC[make_int2(0, 0)]);
    __syncthreads();

    copy::copy_2d_tile_s2g(st_buf, gC, typename CopyTraits::SmemLayout{},
                           typename CopyTraits::GmemLayout{},
                           typename CopyTraits::TiledCopy{});
}

}  // namespace

namespace testing {

template <typename Element, int kRows, int kCols>
struct CopyTraits {
    using SmemLayout = tl::RowMajor<kRows, kCols>;
    using GmemLayout = tl::RowMajor<kRows, kCols>;
    using CopyInst = Copy_Atom<DefaultCopy, Element>;

    using TiledCopy = decltype(make_tiled_copy(CopyInst{}, tl::RowMajor<8, 4>{},
                                               Layout<Shape<_1, _4>>{}));
};

TEST(TestWmma, shape1) {
    using Element = cutlass::half_t;

    // operand A/B: shared memory
    using TemporalExecShared = TileShape<1, 1>;
    using WarpLayout = tl::RowMajor<1, 1>;
    using ThreadLayout = tl::RowMajor<16, 2>;  // fixed when using ldmatrix.
    using ElemTileShared = tl::RowMajor<1, 8>;
    using Shared = SharedTile<Element, TemporalExecShared, WarpLayout,
                              ThreadLayout, ElemTileShared>;

    // operand A/B: register
    using ExecShapeReg = TileShape<1, 1>;
    using ElemTileReg = tl::RowMajor<1, 8>;  // fixed when using ldmatrix
    using Reg = RegTile<Element, ExecShapeReg, ElemTileReg>;

    using ElementC = float;
    // operand C: register
    using ExecShapeRegC = TileShape<1, 1>;
    // wmma's register tile is 1 x 2 nested with 2 x 2
    using ElemTileRegC = tl::RowMajor<2, 4>;
    using RegC = RegTile<ElementC, ExecShapeRegC, ElemTileRegC>;

    // operand C: shared memory
    // wmma's register tile is 1 x 2 nested with 2 x 2
    using ElemTileSharedC = tl::RowMajor<2, 4>;
    using ThreadLayoutC = tl::RowMajor<8, 4>;
    using WarpLayoutC = tl::RowMajor<1, 1>;
    using TemporalExecSharedC = TileShape<1, 1>;
    using SharedC = SharedTile<ElementC, TemporalExecSharedC, WarpLayoutC,
                               ThreadLayoutC, ElemTileSharedC>;

    thrust::device_vector<float> d_c(SharedC::kNumel);
    thrust::fill(d_c.begin(), d_c.end(), 0.);

    const int kThreads = Shared::kThreads;
    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);

    using CopyS2G = CopyTraits<ElementC, Shared::kRows, Shared::kCols>;

    int shm_size = (Shared::kNumel + Shared::kNumel) * sizeof(Element);
    test_wmma1<Element, Shared, Reg, Shared, Reg, ElementC, SharedC, RegC,
               CopyS2G><<<dim_grid, dim_block, shm_size>>>(
        thrust::raw_pointer_cast(d_c.data()));
    cudaDeviceSynchronize();

    thrust::host_vector<float> h_c1;
    h_c1 = d_c;

    {  // ground truth
        int M = 16, N = 16, K = 16;

        thrust::host_vector<float> h_a(M * K);
        thrust::host_vector<float> h_b(K * N);
        thrust::host_vector<float> h_c(M * N);

        for (int i = 0; i < h_a.size(); ++i) h_a[i] = i % 2048 / 100.;

        for (int i = 0; i < h_b.size(); ++i) {
            h_b[i] = i % 2048 / 100.;
        }
        thrust::fill(h_c.begin(), h_c.end(), 0.);

        check_result1(thrust::raw_pointer_cast(h_a.data()),
                      thrust::raw_pointer_cast(h_b.data()),
                      thrust::raw_pointer_cast(h_c.data()), M, N, K,
                      thrust::raw_pointer_cast(h_c1.data()));
    }
}

}  // namespace testing
}  // namespace tiledcuda
