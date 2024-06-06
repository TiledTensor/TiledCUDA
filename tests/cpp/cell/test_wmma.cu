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

template <typename S2R, typename R2S, typename S2G>
__global__ void test_wmma1(typename R2S::Element* g_buf) {
    using Element = typename S2R::Element;
    using SharedA = typename S2R::Shared;
    using RegA = typename S2R::Reg;

    using SharedB = typename S2R::Shared;
    using RegB = typename S2R::Reg;

    using ElementAcc = typename R2S::Element;
    using SharedC = typename R2S::Shared;
    using RegC = typename R2S::Reg;

    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* buf = reinterpret_cast<Element*>(buf_);
    auto* st_buf = reinterpret_cast<ElementAcc*>(buf);

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

    copy::copy_2d_tile_s2g(st_buf, g_buf, typename S2G::SmemLayout{},
                           typename S2G::GmemLayout{},
                           typename S2G::TiledCopy{});
}

}  // namespace

namespace testing {

template <typename Element_>
struct S2RTraits {
    using Element = Element_;

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
};

template <typename Element_>
struct R2STraits {
    using Element = Element_;

    // operand C: register
    using ExecShapeReg = TileShape<1, 1>;
    // wmma's register tile is 1 x 2 nested with 2 x 2
    using ElemTileReg = tl::RowMajor<2, 4>;
    using Reg = RegTile<Element, ExecShapeReg, ElemTileReg>;

    // operand C: shared memory
    // wmma's register tile is 1 x 2 nested with 2 x 2
    using ElemTileShared = tl::RowMajor<2, 4>;
    using ThreadLayout = tl::RowMajor<8, 4>;
    using WarpLayout = tl::RowMajor<1, 1>;
    using TemporalExecShared = TileShape<1, 1>;
    using Shared = SharedTile<Element, TemporalExecShared, WarpLayout,
                              ThreadLayout, ElemTileShared>;
};

template <typename Element, int kRows, int kCols>
struct S2GTraits {
    using SmemLayout = tl::RowMajor<kRows, kCols>;
    using GmemLayout = tl::RowMajor<kRows, kCols>;
    using CopyInst = Copy_Atom<DefaultCopy, Element>;

    using TiledCopy = decltype(make_tiled_copy(CopyInst{}, tl::RowMajor<8, 4>{},
                                               Layout<Shape<_1, _4>>{}));
};

TEST(TestWmma, shape1) {
    using Element = cutlass::half_t;
    // transfer operand A/B from the shared memory to register
    using LoadS2R = S2RTraits<Element>;
    using SharedA = LoadS2R::Shared;
    using SharedB = LoadS2R::Shared;

    // transfer operand C from register to shared memory
    using StoreR2S = R2STraits<float>;
    using SharedC = StoreR2S::Shared;
    // transfer operand C from shared memory to global memory
    using StoreS2G = S2GTraits<float, SharedA::kRows, SharedB::kCols>;

    const int kThreads = SharedA::kThreads;
    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);

    thrust::device_vector<float> d_c(SharedC::kNumel);
    thrust::fill(d_c.begin(), d_c.end(), 0.);

    int shm_size = (SharedA::kNumel + SharedB::kNumel) * sizeof(Element);
    test_wmma1<LoadS2R, StoreR2S, StoreS2G><<<dim_grid, dim_block, shm_size>>>(
        thrust::raw_pointer_cast(d_c.data()));
    cudaDeviceSynchronize();

    thrust::host_vector<float> h_c1;
    h_c1 = d_c;

    {  // ground truth
        int M = SharedA::kRows, N = SharedB::kCols, K = SharedA::kCols;

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
