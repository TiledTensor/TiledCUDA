#include "cell/mod.hpp"
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

template <typename DType1, typename DType2>
void check_result1(const DType1* a, const DType1* b, DType2* c, int M, int N,
                   int K, const float* result) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            DType2 s = 0.;
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
__device__ void print_tile(const Element* data, int delimeter = kCols) {
    if (threadIdx.x || blockIdx.x || blockIdx.y) return;

    for (int i = 0; i < kRows * kCols; ++i) {
        printf("%.2f, ", float(data[i] * 1.0_hf));  // print cutlass::half_t
        if ((i + 1) % delimeter == 0) printf("\n");
    }
    printf("\n");
}

template <typename Element, typename ElementAcc, typename LoadSharedA,
          typename LoadSharedB, typename LoadRegA, typename LoadRegB,
          typename StoreRegC, typename StoreSharedC>
__global__ void test_wmma1(const Element* ga, const Element* gb,
                           ElementAcc* gc) {
    using SharedA = typename LoadRegA::Shared;
    using RegA = typename LoadRegA::Reg;

    using SharedB = typename LoadRegB::Shared;
    using RegB = typename LoadRegB::Reg;

    using SharedC = typename StoreRegC::Shared;
    using RegC = typename StoreRegC::Reg;

    extern __shared__ __align__(sizeof(double)) unsigned char buf_[];
    auto* shared_a = reinterpret_cast<Element*>(buf_);
    auto* shared_b = shared_a + SharedA::kNumel;
    auto* shared_c = reinterpret_cast<ElementAcc*>(buf_);

    // transfer data tiles from global to shared
    copy::copy_2d_tile_g2s(ga, shared_a, typename LoadSharedA::SrcLayout{},
                           typename LoadSharedA::DstLayout{},
                           typename LoadSharedA::TiledCopy{});
    copy::copy_2d_tile_g2s(gb, shared_b, typename LoadSharedB::SrcLayout{},
                           typename LoadSharedB::DstLayout{},
                           typename LoadSharedB::TiledCopy{});
    __copy_async();
    __syncthreads();

    // #ifdef DEBUG
    //     print_tile<Element, SharedA::kRows, SharedA::kCols * 2>(shared_a);
    //     print_tile<Element, SharedB::kRows, SharedB::kCols * 2>(shared_b);
    // #endif

    // declare shared memory tile
    SharedA sa_tiles(shared_a);
    SharedB sb_tiles(shared_b);
    SharedC sc_tile(shared_c);

    // declare register tile
    RegA ra;
    RegB rb;
    RegC acc;

    for (int k = 0; k < 2; ++k) {  // iterate over register tile along K
        // transfer data tiles from shared to register
        copy::copy_2d_tile_s2r(sa_tiles[make_int2(0, 0)], ra);
        copy::copy_2d_tile_s2r(sb_tiles[make_int2(0, 0)], rb);

        compute::gemm2(ra, rb, acc);  // compute at register
    }

    // transfer data tile from register to shared
    copy::copy_2d_tile_r2s(acc, sc_tile[make_int2(0, 0)]);
    __syncthreads();

    // transfer data tile from shared to global
    copy::copy_2d_tile_s2g(shared_c, gc, typename StoreSharedC::SrcLayout{},
                           typename StoreSharedC::DstLayout{},
                           typename StoreSharedC::TiledCopy{});
    __copy_async();
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

TEST(TestWmma, shape1) {
    using Element = cutlass::half_t;

    // transfer operand A/B from the shared memory to register
    using LoadRegAB = S2RTraits<Element>;
    using SharedA = LoadRegAB::Shared;
    using SharedB = LoadRegAB::Shared;

    const int kThreads = SharedA::kThreads;
    const int M = SharedA::kRows;
    const int K = SharedA::kCols * 2;
    const int N = SharedB::kCols;

    LOG(INFO) << "[M, N, K]: " << M << ", " << N << ", " << K << std::endl
              << "kThreads: " << kThreads;

    using LoadSharedA = traits::G2S2DCopyTraits<Element, M, K, M, K, kThreads,
                                                false /*use swizzle*/>;

    using LoadSharedB = traits::G2S2DCopyTraits<Element, N, K, N, K, kThreads,
                                                false /*use swizzle*/>;

    using ElementAcc = float;
    // transfer operand C from register to shared memory
    using StoreRegC = R2STraits<ElementAcc>;
    using SharedC = StoreRegC::Shared;

    // transfer operand C from shared memory to global memory
    using StoreSharedC =
        traits::S2G2DCopyTraits<ElementAcc, M, N, M, N, kThreads,
                                false /*use swizzle*/>;

    thrust::host_vector<Element> h_a(M * K);
    thrust::host_vector<Element> h_b(K * N);
    thrust::host_vector<ElementAcc> h_c(M * N);

    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<Element>(i % 2048 / 100.);

    for (int i = 0; i < h_b.size(); ++i) {
        h_b[i] = static_cast<Element>(i % 2048 / 100.);
    }
    thrust::fill(h_c.begin(), h_c.end(), 0.);

    thrust::device_vector<Element> d_a = h_a;
    thrust::device_vector<Element> d_b = h_b;
    thrust::device_vector<ElementAcc> d_c = h_c;

    dim3 dim_grid(1, 1, 1);
    dim3 dim_block(kThreads, 1, 1);

    int size_ab = (M + N) * K * sizeof(Element);
    int size_c = M * N * sizeof(ElementAcc);
    int shm_size = size_ab > size_c ? size_ab : size_c;

    test_wmma1<Element, float, LoadSharedA, LoadSharedB, LoadRegAB, LoadRegAB,
               StoreRegC, StoreSharedC><<<dim_grid, dim_block, shm_size>>>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_c.data()));

    cudaDeviceSynchronize();

    thrust::host_vector<float> h_c1;
    h_c1 = d_c;

    // {  // ground truth

    //     check_result1(thrust::raw_pointer_cast(h_a.data()),
    //                   thrust::raw_pointer_cast(h_b.data()),
    //                   thrust::raw_pointer_cast(h_c.data()), M, N, K,
    //                   thrust::raw_pointer_cast(h_c1.data()));
    // }
}

}  // namespace testing
}  // namespace tiledcuda
